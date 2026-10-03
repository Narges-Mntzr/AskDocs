# AskDocs

AskDocs is a citation-first document Q&A API. It indexes PDF, plain text, and Markdown, retrieves the closest passages with embeddings, and answers only from those passages. If nothing clears the similarity threshold, it says the active documents do not contain enough information.

Answers are extracted sentences from the indexed text. The service does not call a chat model and does not invent wording.

## Stack

- Python 3.12, Django 5, Django REST Framework
- SQLite for documents, chunks, and embedding vectors
- An OpenAI-compatible embeddings API (`POST /embeddings`)
- pypdf for text extraction from PDFs
- Gunicorn and WhiteNoise in the container
- Swagger UI at `/api/docs/`

## How it works

1. A document is created from raw text or an uploaded `.pdf`, `.txt`, or `.md` file. PDF pages are extracted with pypdf. Other files are read as UTF-8.
2. The text is split into chunks: one chunk per non-empty line. A line longer than `RAG_CHUNK_MAX_CHARS` is cut into overlapping windows (`RAG_CHUNK_OVERLAP` characters).
3. Chunks are embedded in batches and stored with the document. Replacing the content deletes the old chunks, increments the document version, and indexes the new text. Deleting a document deactivates it so its chunks are no longer searched.
4. A question is embedded with the same model. Every chunk of an active document is ranked by cosine similarity, and the top `top_k` matches are kept (default 5, maximum 20).
5. Matches below `RAG_MIN_SIMILARITY` are dropped. From what remains, the API picks up to three sentences, preferring sentences that share terms with the question and come from a high-scoring chunk.
6. The response includes `sufficient_information`, the extracted `answer`, and `sources` (document id, title, version, chunk index, score, and a short excerpt). With no accepted match, `sufficient_information` is `false` and `sources` is empty.

## Run

Copy `.env.example` to `.env` and set `EMBEDDING_API_KEY`. `EMBEDDING_BASE_URL` is required. The defaults below are the configuration selected by the evaluation.

| Variable | Default | Role |
| --- | --- | --- |
| `EMBEDDING_BASE_URL` | — | Embeddings API origin, without a trailing path |
| `EMBEDDING_API_KEY` | — | Bearer token |
| `EMBEDDING_MODEL` | `Embedding-3-Large` | Model name sent to `/embeddings` |
| `EMBEDDING_TIMEOUT_SECONDS` | `20` | HTTP timeout |
| `RAG_TOP_K` | `5` | Passages considered per question |
| `RAG_MIN_SIMILARITY` | `0.50` | Minimum cosine similarity |
| `RAG_CHUNK_MAX_CHARS` | `2200` | Maximum characters in a chunk |
| `RAG_CHUNK_OVERLAP` | `180` | Overlap used when a line is windowed |

Docker Compose reads `.env` and starts the API on port 8000. The container runs migrations and collects static files on startup.

```bash
docker compose up --build
```

Locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
set -a && source .env && set +a
python manage.py migrate
python manage.py runserver
```

Interactive docs: [http://localhost:8000/api/docs/](http://localhost:8000/api/docs/). Health check: `GET /health/`.

Create a document, then ask:

```bash
curl -s -X POST http://localhost:8000/knowledge/documents/create/ \
  -H 'Content-Type: application/json' \
  -d '{"title":"Guide","content":"The core product is called Didban.\nIt collects operational events."}'

curl -s -X POST http://localhost:8000/knowledge/ask/ \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the core product called?"}'
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/knowledge/documents/?active=true` | List documents |
| `POST` | `/knowledge/documents/create/` | Create and index (`title` plus `content` or `file`) |
| `GET` | `/knowledge/documents/<id>/` | Fetch one document |
| `POST`, `PUT` | `/knowledge/documents/<id>/` | Edit title or content and re-index |
| `DELETE` | `/knowledge/documents/<id>/` | Deactivate |
| `POST` | `/knowledge/ask/` | Ask a question (`question`, optional `top_k`) |
| `GET` | `/knowledge/stats/` | Active and total document counts |

`python manage.py createsuperuser` enables `/admin/`, which can upload the same file types and re-index on save.

## Evaluation

Retrieval settings were compared on a fixed Persian knowledge base, not by changing the answer extractor. The sweep lives in `knowledge/test_rag_eval.py` and is skipped unless `RUN_RAG_EVAL=1`.

```bash
RUN_RAG_EVAL=1 python manage.py test knowledge.test_rag_eval.EvalSweepTests
```

Each configuration uploads the corpus through `POST /knowledge/documents/create/` and asks all 100 questions through `POST /knowledge/ask/`. Embeddings are cached under `knowledge/eval/.embedding-cache/` so repeated text is not sent to the API again. `top_k` stayed at 5 and chunk overlap stayed at 180.

The grid is 4 models × 4 chunk sizes × 6 thresholds = 96 runs:

- Models: `Bge-m3` (1024), `Embedding-3-Small` (1536), `Embedding-3-Large` (3072), `Gemini-embedding-001` (3072)
- Maximum chunk length: 2200, 1400, 800, 400 characters
- Similarity threshold: 0.20, 0.30, 0.40, 0.50, 0.60, 0.70

### Eval files

| File | Contents |
| --- | --- |
| `knowledge/eval/documents/` | 10 Markdown guides for a fictional company (product, pricing, limits, support, leave, security, incidents, API, billing, usage rules) |
| `knowledge/eval/questions.json` | 100 questions: 70 answerable from one guide, 30 unrelated. Each item has `question`, `is_relevant`, `document_title`, and `relevant_passage` |
| `knowledge/eval/results.csv` | One row per configuration: model, dimensions, chunk size, overlap, threshold, top-k, scores, and chunk count |
| `knowledge/eval/report.md` | The same rows ranked, with the winning configuration called out |

A relevant question is correct only when the API reports sufficient information, cites that document's title, and includes the expected passage in the answer. An unrelated question is correct only when the API reports that it does not have enough information. The overall score is correct answers divided by 100.

### Result

`Embedding-3-Large` at a 0.50 threshold scored **97% (97/100)** at every chunk size: **97.1%** on relevant questions (68/70) and **96.7%** on refusals (29/30). The 2200-character setting is the one shipped in the defaults because it matches that score with the smallest index (230 chunks, versus 463 at 400 characters).

Threshold moved the score much more than chunk size. Lower thresholds answered relevant questions and also answered unrelated ones. Higher thresholds refused unrelated questions and dropped relevant ones. Best score per model:

| Model | Chunk size | Threshold | Overall | Relevant | Refusal | Chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Embedding-3-Large | 2200 | 0.50 | 97% | 97.1% | 96.7% | 230 |
| Bge-m3 | 2200 | 0.60 | 94% | 98.6% | 83.3% | 230 |
| Gemini-embedding-001 | 2200 | 0.70 | 91% | 100% | 70.0% | 230 |
| Embedding-3-Small | 2200 | 0.40 | 88% | 92.9% | 76.7% | 230 |

`Embedding-3-Small` at 400 characters and the same 0.40 threshold fell to 85%. The full ranking is in `knowledge/eval/report.md` and `knowledge/eval/results.csv`.

API behavior is also covered by `python manage.py test knowledge.tests`, which stubs the embedding client and does not call the remote API.
