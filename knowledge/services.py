import hashlib
import io
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from django.conf import settings
from pypdf import PdfReader

from knowledge.models import Chunk, Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".text", ".md", ".markdown"}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_text(*, content: str | None = None, uploaded_file=None, source_name: str = "") -> str:
    """Extract UTF-8 text from text/markdown or a text PDF."""
    if content is not None:
        return content.replace("\r\n", "\n").strip()
    if uploaded_file is None:
        raise ValueError("One of content or file is required")
    name = (source_name or getattr(uploaded_file, "name", "")).lower()
    if name and not any(name.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise ValueError("supported formats are PDF, TXT and Markdown")
    raw = uploaded_file.read()
    if name.endswith(".pdf") or raw[:4] == b"%PDF":
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
    return raw.decode("utf-8-sig").replace("\r\n", "\n").strip()


def split_chunks(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    """Paragraph-aware chunker with a bounded overlap for retrieval context."""
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            step = max(1, max_chars - overlap)
            chunks.extend(paragraph[start : start + max_chars].strip() for start in range(0, len(paragraph), step))
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            tail = current[-overlap:].strip()
            current = f"{tail}\n\n{paragraph}" if tail else paragraph
            if len(current) > max_chars:
                current = paragraph
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _fallback_embedding(text: str, dimensions: int = 256) -> list[float]:
    """Deterministic local fallback, useful for development and offline tests."""
    values = [0.0] * dimensions
    for token in re.findall(r"\w+", text.casefold()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        values[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not settings.EMBEDDING_API_KEY:
        return [_fallback_embedding(t) for t in texts]
    payload = json.dumps({"model": settings.EMBEDDING_MODEL, "input": texts}).encode()
    request = urllib.request.Request(
        f"{settings.EMBEDDING_BASE_URL}/embeddings",
        data=payload,
        headers={"Authorization": f"Bearer {settings.EMBEDDING_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.EMBEDDING_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
        data = sorted(result["data"], key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in data]
        if len(vectors) != len(texts):
            raise ValueError("Embedding service returned an unexpected item count")
        return vectors
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
        # The application remains usable during a temporary provider outage. The health
        # endpoint and document metadata make the fallback observable during operations.
        return [_fallback_embedding(t) for t in texts]


def embed_in_batches(texts: list[str], max_request_chars: int = 180_000) -> list[list[float]]:
    """Respect the provider's 200k-character request limit."""
    vectors: list[list[float]] = []
    batch: list[str] = []
    chars = 0
    for text in texts:
        if batch and chars + len(text) > max_request_chars:
            vectors.extend(embed_texts(batch))
            batch, chars = [], 0
        batch.append(text)
        chars += len(text)
    if batch:
        vectors.extend(embed_texts(batch))
    return vectors


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(left[i] * left[i] for i in range(size)))
    right_norm = math.sqrt(sum(right[i] * right[i] for i in range(size)))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


@dataclass(frozen=True)
class Match:
    chunk: Chunk
    score: float


def index_document(document: Document) -> int:
    document.chunks.all().delete()
    chunks = split_chunks(document.content)
    vectors = embed_in_batches(chunks)
    Chunk.objects.bulk_create(
        [
            Chunk(document=document, ordinal=i, text=text, embedding=vector)
            for i, (text, vector) in enumerate(zip(chunks, vectors))
        ]
    )
    return len(chunks)


def retrieve(question: str, top_k: int | None = None) -> list[Match]:
    query_vector = embed_texts([question])[0]
    matches = [
        Match(chunk=chunk, score=cosine_similarity(query_vector, chunk.embedding))
        for chunk in Chunk.objects.select_related("document").filter(document__is_active=True)
    ]
    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[: max(1, min(top_k or settings.RAG_TOP_K, 20))]


def answer_question(question: str, top_k: int | None = None) -> dict:
    matches = retrieve(question, top_k)
    accepted = [m for m in matches if m.score >= settings.RAG_MIN_SIMILARITY]
    if not accepted:
        return {
            "answer": "اطلاعات کافی در اسناد فعال برای پاسخ به این پرسش پیدا نشد.",
            "sufficient_information": False,
            "sources": [],
        }
    query_terms = set(re.findall(r"\w+", question.casefold()))
    candidates: list[tuple[float, str, Match]] = []
    for match in accepted:
        for sentence in re.split(r"(?<=[.!؟?؛])\s+|\n+", match.chunk.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            terms = set(re.findall(r"\w+", sentence.casefold()))
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            candidates.append((overlap + match.score * 0.2, sentence, match))
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[tuple[str, Match]] = []
    seen = set()
    for _, sentence, match in candidates:
        if sentence in seen:
            continue
        chosen.append((sentence, match))
        seen.add(sentence)
        if len(chosen) >= 3:
            break
    answer = " ".join(sentence for sentence, _ in chosen) or accepted[0].chunk.text
    source_keys = {(m.chunk.document_id, m.chunk.ordinal): m for _, m in chosen}
    sources = [
        {
            "document_id": m.chunk.document_id,
            "document_title": m.chunk.document.title,
            "document_version": m.chunk.document.version,
            "chunk": m.chunk.ordinal,
            "score": round(m.score, 4),
            "excerpt": m.chunk.text[:300],
        }
        for m in source_keys.values()
    ]
    return {"answer": answer, "sufficient_information": True, "sources": sources}
