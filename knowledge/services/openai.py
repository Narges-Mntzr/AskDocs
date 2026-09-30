import json
import urllib.error
import urllib.request

from django.conf import settings


class OpenAIRequestError(Exception):
    pass


def _request(*, method: str, path: str, payload: dict | None = None) -> dict:
    if not settings.EMBEDDING_API_KEY:
        raise OpenAIRequestError("OpenAI API key is not configured")
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{settings.EMBEDDING_BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(
            request, timeout=settings.EMBEDDING_TIMEOUT_SECONDS
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OpenAIRequestError(str(exc)) from exc


def create_embeddings(texts: list[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    result = _request(
        method="POST",
        path="/embeddings",
        payload={"model": model or settings.EMBEDDING_MODEL, "input": texts},
    )
    try:
        data = sorted(result["data"], key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in data]
    except (KeyError, TypeError) as exc:
        raise OpenAIRequestError(
            "Embedding service returned an unexpected payload"
        ) from exc
    if len(vectors) != len(texts):
        raise OpenAIRequestError("Embedding service returned an unexpected item count")
    return vectors


def list_models() -> list[dict]:
    result = _request(method="GET", path="/models")
    try:
        return list(result["data"])
    except (KeyError, TypeError) as exc:
        raise OpenAIRequestError(
            "Models service returned an unexpected payload"
        ) from exc
