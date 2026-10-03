import hashlib
import io
import math
import re
import unicodedata
from dataclasses import dataclass

from django.conf import settings
from pypdf import PdfReader

from knowledge.models import Chunk, Document
from knowledge.services.openai import create_embeddings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".text", ".md", ".markdown"}
# Same visual line in a PDF can jitter by a fraction of the font size.
_PDF_LINE_Y_TOLERANCE = 0.45
# A larger jump is a paragraph break; ordinary leading stays one chunk.
_PDF_PARAGRAPH_Y_GAP = 1.8


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
        return _extract_pdf_text(raw)
    return raw.decode("utf-8-sig").replace("\r\n", "\n").strip()


def _extract_pdf_text(raw: bytes) -> str:
    """Join glyphs that share a line. pypdf otherwise splits one line into many."""
    reader = PdfReader(io.BytesIO(raw))
    pages = [_extract_pdf_page(page) for page in reader.pages]
    return "\n\n".join(page for page in pages if page).strip()


def _extract_pdf_page(page) -> str:
    fragments: list[tuple[float, float, str]] = []

    def visitor(text, cm, tm, font_dict, font_size):
        cleaned = unicodedata.normalize("NFKC", text).strip()
        if not cleaned:
            return
        positioned = abs(float(tm[4])) > 0.01 or abs(float(tm[5])) > 0.01
        if positioned:
            y = float(cm[1]) * float(tm[4]) + float(cm[3]) * float(tm[5]) + float(cm[5])
            size = abs(float(cm[3]) * float(tm[3]) * float(font_size)) or 12.0
        elif fragments:
            y, size, _ = fragments[-1]
        else:
            y, size = 0.0, 12.0
        fragments.append((y, size, cleaned))

    page.extract_text(visitor_text=visitor)
    return _join_pdf_lines(fragments)


def _join_pdf_lines(fragments: list[tuple[float, float, str]]) -> str:
    lines: list[tuple[float, float, str]] = []
    line_y: float | None = None
    tolerance = 0.0
    for y, size, text in fragments:
        if line_y is None or abs(y - line_y) > tolerance:
            lines.append((y, size, text))
            line_y = y
            tolerance = max(size, 1.0) * _PDF_LINE_Y_TOLERANCE
        else:
            prev_y, prev_size, prev_text = lines[-1]
            lines[-1] = (prev_y, prev_size, f"{prev_text} {text}")

    paragraphs: list[str] = []
    previous_y: float | None = None
    for y, size, text in lines:
        gap_limit = max(size, 1.0) * _PDF_PARAGRAPH_Y_GAP
        if previous_y is None or abs(y - previous_y) > gap_limit:
            paragraphs.append(text)
        else:
            paragraphs[-1] = f"{paragraphs[-1]} {text}"
        previous_y = y
    return "\n".join(paragraphs)


def split_chunks(text: str, max_chars: int = 2200, overlap: int = 180) -> list[str]:
    """One chunk per non-empty line. Lines longer than max_chars are windowed."""
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
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
    texts: list[str],
    max_request_chars: int = 180_000,
    max_items: int = 4,
) -> list[list[float]]:
    """Keep each request under the provider character cap and a small input count.

    More than a handful of Embedding-3-Large inputs do not return within the HTTP timeout.
    """
    vectors: list[list[float]] = []
    batch: list[str] = []
    chars = 0
    for text in texts:
        overflows = chars + len(text) > max_request_chars or len(batch) >= max_items
        if batch and overflows:
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
    chunks = split_chunks(
        document.content,
        max_chars=settings.RAG_CHUNK_MAX_CHARS,
        overlap=settings.RAG_CHUNK_OVERLAP,
    )
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
