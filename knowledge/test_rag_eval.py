import csv
import hashlib
import json
import os
import re
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient

from knowledge.models import Chunk, Document
from knowledge.services.general import retrieve
from knowledge.services.openai import OpenAIRequestError
from knowledge.services.openai import create_embeddings as request_embeddings

EVAL_DIR = Path(__file__).resolve().parent / "eval"
DOCUMENTS_DIR = EVAL_DIR / "documents"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
REPORT_PATH = EVAL_DIR / "report.md"
RESULTS_PATH = EVAL_DIR / "results.csv"
CACHE_DIR = EVAL_DIR / ".embedding-cache"

MODELS = (
    ("Bge-m3", 1024),
    ("Embedding-3-Small", 1536),
    ("Embedding-3-Large", 3072),
    ("Gemini-embedding-001", 3072),
)
MAX_CHARS = (2200, 1400, 800, 400)
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
EMBED_BATCH_ITEMS = 8
OVERLAP = 180
TOP_K = 5


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).casefold().strip()


def case_is_correct(case: dict, payload: dict) -> bool:
    """Score one question against an /ask/ response."""
    if not case["is_relevant"]:
        return payload["sufficient_information"] is False
    if payload["sufficient_information"] is not True:
        return False
    titles = {source["document_title"] for source in payload["sources"]}
    if case["document_title"] not in titles:
        return False
    return normalize_text(case["relevant_passage"]) in normalize_text(payload["answer"])


def document_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise ValueError("markdown document has no H1 title")


def load_questions() -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if len(questions) != 100:
        raise ValueError(f"expected 100 questions, found {len(questions)}")
    return questions


def _cache_path(model: str, text: str) -> Path:
    digest = hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _request_embeddings(texts: list[str], model: str) -> list[list[float]]:
    delay = 3
    last_error: OpenAIRequestError | None = None
    for attempt in range(3):
        try:
            return request_embeddings(texts, model=model)
        except OpenAIRequestError as exc:
            last_error = exc
            if attempt == 2:
                break
            _log(f"retry {model} ({len(texts)} texts) after: {exc}")
            time.sleep(delay)
            delay *= 2
    assert last_error is not None
    raise last_error


def cached_create_embeddings(texts, model=None):
    if not texts:
        return []
    model_name = model or settings.EMBEDDING_MODEL
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vectors: list[list[float] | None] = [None] * len(texts)
    missing: list[tuple[int, str]] = []
    for index, text in enumerate(texts):
        path = _cache_path(model_name, text)
        if path.exists():
            vectors[index] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append((index, text))
    unique_missing = list(dict.fromkeys(text for _, text in missing))
    fetched: dict[str, list[float]] = {}
    for start in range(0, len(unique_missing), EMBED_BATCH_ITEMS):
        batch = unique_missing[start : start + EMBED_BATCH_ITEMS]
        _log(f"embedding {model_name}: {len(batch)} texts")
        for text, vector in zip(batch, _request_embeddings(batch, model_name)):
            _cache_path(model_name, text).write_text(
                json.dumps(vector), encoding="utf-8"
            )
            fetched[text] = vector
    for index, text in missing:
        vectors[index] = fetched[text]
    return vectors


class RetrieveCache:
    def __init__(self):
        self.store: dict[tuple, list] = {}

    def __call__(self, question: str, top_k: int | None = None):
        key = (question, top_k)
        if key not in self.store:
            self.store[key] = retrieve(question, top_k)
        return list(self.store[key])

    def clear(self) -> None:
        self.store.clear()


def upload_eval_documents(client: APIClient) -> int:
    Document.objects.all().delete()
    paths = sorted(DOCUMENTS_DIR.glob("*.md"))
    if len(paths) != 10:
        raise ValueError(f"expected 10 markdown documents, found {len(paths)}")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        upload = SimpleUploadedFile(
            path.name, text.encode(), content_type="text/markdown"
        )
        response = client.post(
            "/knowledge/documents/create/",
            {"title": document_title(text), "file": upload},
            format="multipart",
        )
        if response.status_code != 201:
            raise RuntimeError(
                f"upload failed for {path.name}: {response.status_code} {response.content!r}"
            )
    return Chunk.objects.count()


def score_questions(client: APIClient, questions: list[dict], threshold: float) -> dict:
    relevant_correct = 0
    relevant_total = 0
    refusal_correct = 0
    refusal_total = 0
    with override_settings(RAG_MIN_SIMILARITY=threshold):
        for case in questions:
            response = client.post(
                "/knowledge/ask/",
                {"question": case["question"], "top_k": TOP_K},
                format="json",
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"ask failed: {response.status_code} {response.content!r}"
                )
            correct = case_is_correct(case, response.data)
            if case["is_relevant"]:
                relevant_total += 1
                relevant_correct += int(correct)
            else:
                refusal_total += 1
                refusal_correct += int(correct)
    correct = relevant_correct + refusal_correct
    total = relevant_total + refusal_total
    return {
        "correct": correct,
        "total": total,
        "score": correct / total,
        "relevant_correct": relevant_correct,
        "relevant_total": relevant_total,
        "relevant_hit": relevant_correct / relevant_total,
        "refusal_correct": refusal_correct,
        "refusal_total": refusal_total,
        "refusal_hit": refusal_correct / refusal_total,
    }


def write_report(rows: list[dict], failures: list[str]) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (row["score"], row["relevant_hit"], row["refusal_hit"]),
        reverse=True,
    )
    lines = [
        "# گزارش مقایسه تنظیم‌ها",
        "",
        "هر حالت سندهای `knowledge/eval/documents` را آپلود می‌کند و هر ۱۰۰ سؤال را از `/knowledge/ask/` می‌پرسد.",
        "",
        "سؤال مرتبط وقتی درست است که پاسخ، اطلاعات کافی اعلام کند، عنوان همان سند در منابع باشد، و عبارت مرتبط داخل متن پاسخ باشد.",
        "سؤال نامرتبط وقتی درست است که سیستم بگوید اطلاعات کافی پیدا نشد.",
        "امتیاز کل، تعداد پاسخ درست تقسیم بر ۱۰۰ است.",
        f"`top_k` برابر {TOP_K} و overlap برابر {OVERLAP} است و در این مقایسه عوض نشده‌اند.",
        "",
    ]
    if ranked:
        best = ranked[0]
        lines.extend(
            [
                "## بهترین حالت",
                "",
                (
                    f"- مدل: `{best['model']}` (بعد {best['dimensions']})"
                    f"\n- حداکثر کاراکتر چانک: {best['max_chars']}"
                    f"\n- آستانه شباهت: {best['min_similarity']:.2f}"
                    f"\n- امتیاز کل: {best['score']:.1%}"
                    f" ({best['correct']} از {best['total']})"
                    f"\n- سؤال‌های مرتبط: {best['relevant_hit']:.1%}"
                    f" ({best['relevant_correct']} از {best['relevant_total']})"
                    f"\n- سؤال‌های نامرتبط: {best['refusal_hit']:.1%}"
                    f" ({best['refusal_correct']} از {best['refusal_total']})"
                ),
                "",
            ]
        )
    else:
        lines.extend(["## بهترین حالت", "", "هیچ حالتی تا آخر اجرا نشد.", ""])
    lines.extend(
        [
            "## همهٔ حالت‌ها",
            "",
            "| مدل | حداکثر کاراکتر | آستانه | امتیاز کل | مرتبط | نامرتبط | تعداد چانک |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in ranked:
        lines.append(
            "| {model} | {max_chars} | {min_similarity:.2f} | {score:.1%} | {relevant_hit:.1%} | {refusal_hit:.1%} | {chunk_count} |".format(
                **row
            )
        )
    if failures:
        lines.extend(["", "## خطاها", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            extrasaction="ignore",
            fieldnames=[
                "model",
                "dimensions",
                "max_chars",
                "overlap",
                "min_similarity",
                "top_k",
                "score",
                "correct",
                "relevant_hit",
                "relevant_correct",
                "refusal_hit",
                "refusal_correct",
                "chunk_count",
            ],
        )
        writer.writeheader()
        writer.writerows(ranked)


class ScoreCaseTests(SimpleTestCase):
    def test_relevant_answer_must_include_passage_and_title(self):
        case = {
            "is_relevant": True,
            "relevant_passage": "هستهٔ محصول",
            "document_title": "راهنما",
        }
        self.assertTrue(
            case_is_correct(
                case,
                {
                    "answer": "هستهٔ محصول همراه داده است.",
                    "sufficient_information": True,
                    "sources": [{"document_title": "راهنما"}],
                },
            )
        )
        self.assertFalse(
            case_is_correct(
                case,
                {
                    "answer": "پاسخ دیگری است.",
                    "sufficient_information": True,
                    "sources": [{"document_title": "راهنما"}],
                },
            )
        )

    def test_irrelevant_question_must_be_refused(self):
        case = {
            "is_relevant": False,
            "relevant_passage": "",
            "document_title": "",
        }
        self.assertTrue(
            case_is_correct(
                case,
                {
                    "answer": "اطلاعات کافی پیدا نشد.",
                    "sufficient_information": False,
                    "sources": [],
                },
            )
        )
        self.assertFalse(
            case_is_correct(
                case,
                {
                    "answer": "یک پاسخ ساختگی.",
                    "sufficient_information": True,
                    "sources": [],
                },
            )
        )


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@unittest.skipUnless(
    os.getenv("RUN_RAG_EVAL") == "1",
    "set RUN_RAG_EVAL=1 to score chunk size, threshold, and model",
)
@override_settings(RAG_TOP_K=TOP_K, EMBEDDING_TIMEOUT_SECONDS=180)
class EvalSweepTests(TestCase):
    def test_sweep_uploads_documents_and_writes_report(self):
        questions = load_questions()
        client = APIClient()
        retrieve_cache = RetrieveCache()
        rows: list[dict] = []
        failures: list[str] = []
        with (
            patch(
                "knowledge.services.general.create_embeddings",
                new=cached_create_embeddings,
            ),
            patch("knowledge.services.general.retrieve", new=retrieve_cache),
        ):
            try:
                for model, dimensions in MODELS:
                    for max_chars in MAX_CHARS:
                        label = f"{model} max_chars={max_chars}"
                        try:
                            rows.extend(
                                self._run_config(
                                    client,
                                    questions,
                                    retrieve_cache,
                                    model,
                                    dimensions,
                                    max_chars,
                                )
                            )
                        except Exception as exc:
                            failures.append(f"{label}: {exc}")
                            _log(f"failed {label}: {exc}")
                        finally:
                            write_report(rows, failures)
            finally:
                write_report(rows, failures)
        self.assertTrue(REPORT_PATH.exists())
        self.assertGreater(len(rows), 0, "\n".join(failures))
        if failures:
            self.fail("some configs failed:\n" + "\n".join(failures))

    def _run_config(
        self,
        client,
        questions,
        retrieve_cache: RetrieveCache,
        model: str,
        dimensions: int,
        max_chars: int,
    ) -> list[dict]:
        retrieve_cache.clear()
        _log(f"indexing {model} max_chars={max_chars}")
        with override_settings(
            EMBEDDING_MODEL=model,
            RAG_CHUNK_MAX_CHARS=max_chars,
            RAG_CHUNK_OVERLAP=OVERLAP,
        ):
            chunk_count = upload_eval_documents(client)
            sample = Chunk.objects.values_list("embedding", flat=True).first()
            if sample is None or len(sample) != dimensions:
                found = 0 if sample is None else len(sample)
                raise RuntimeError(
                    f"{model} returned dimension {found}, expected {dimensions}"
                )
            cached_create_embeddings([case["question"] for case in questions])
            rows = []
            for threshold in THRESHOLDS:
                _log(f"asking {model} max_chars={max_chars} threshold={threshold:.2f}")
                scored = score_questions(client, questions, threshold)
                row = {
                    "model": model,
                    "dimensions": dimensions,
                    "max_chars": max_chars,
                    "overlap": OVERLAP,
                    "min_similarity": threshold,
                    "top_k": TOP_K,
                    "chunk_count": chunk_count,
                    **scored,
                }
                rows.append(row)
                _log(
                    f"score {model} max_chars={max_chars} "
                    f"threshold={threshold:.2f} -> {scored['score']:.1%}"
                )
            return rows
