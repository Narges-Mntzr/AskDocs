import hashlib
import io
import math
import re
from dataclasses import dataclass

from django.conf import settings
from pypdf import PdfReader

from knowledge.models import Chunk, Document
from knowledge.services.openai import create_embeddings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".text", ".md", ".markdown"}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_and_validate_text(*, content: str | None = None, uploaded_file=None) -> str:
    """Extract UTF-8 text from text/markdown or a text PDF."""
    if content is not None:
        return content.replace("\r\n", "\n").strip()
    if uploaded_file is None:
        raise ValueError("One of content or file is required")
    name = (getattr(uploaded_file, "name", "") or "").lower()
    if name and not any(name.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise ValueError("supported formats are PDF, TXT and Markdown")
    raw = uploaded_file.read()
    if name.endswith(".pdf") or raw[:4] == b"%PDF":
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
    return raw.decode("utf-8-sig").replace("\r\n", "\n").strip()


def split_chunks(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    """One chunk per non-empty line. Lines longer than max_chars are windowed."""
    lines = [
        re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines() if line.strip()
    ]
    if not lines:
        return []
    step = max(1, max_chars - overlap)
    chunks: list[str] = []
    for line in lines:
        if len(line) <= max_chars:
            chunks.append(line)
            continue
        for start in range(0, len(line), step):
            piece = line[start : start + max_chars].strip()
            if piece:
                chunks.append(piece)
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return create_embeddings(texts)


def embed_in_batches(
    texts: list[str], max_request_chars: int = 180_000
) -> list[list[float]]:
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
        for chunk in Chunk.objects.select_related("document").filter(
            document__is_active=True
        )
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
