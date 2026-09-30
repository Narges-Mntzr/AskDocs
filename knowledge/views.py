import json

from django.conf import settings
from django.db import transaction
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from knowledge.models import Document
from knowledge.serializers import DocumentSerializer
from knowledge.services import answer_question, extract_text, index_document, sha256_text


def health(request: HttpRequest) -> JsonResponse:
    return JsonResponse({"status": "ok", "embedding_configured": bool(settings.EMBEDDING_API_KEY)})


def api_schema(request: HttpRequest) -> JsonResponse:
    return JsonResponse(OPENAPI_SCHEMA)


def api_docs(request: HttpRequest):
    html = """<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Knowledge API</title>
    <link rel='stylesheet' href='https://unpkg.com/swagger-ui-dist/swagger-ui.css'></head><body>
    <div id='swagger-ui'></div><script src='https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js'></script>
    <script>window.ui=SwaggerUIBundle({url:'/api/schema/',dom_id:'#swagger-ui',deepLinking:true,presets:[SwaggerUIBundle.presets.apis,SwaggerUIBundle.SwaggerUIStandalonePreset]});</script>
    </body></html>"""
    from django.http import HttpResponse

    return HttpResponse(html)


class DocumentListCreateView(APIView):
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def get(self, request):
        active_only = request.query_params.get("active", "true").lower() != "false"
        queryset = Document.objects.all()
        if active_only:
            queryset = queryset.filter(is_active=True)
        return Response(DocumentSerializer(queryset, many=True).data)

    @transaction.atomic
    def post(self, request):
        payload = request.data
        title = str(payload.get("title") or "").strip()
        uploaded = payload.get("file")
        content = payload.get("content")
        source_name = str(payload.get("source_name") or getattr(uploaded, "name", "")).strip()
        if not title:
            return Response({"detail": "title is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            text = extract_text(content=str(content) if content is not None else None, uploaded_file=uploaded, source_name=source_name)
        except Exception as exc:
            # pypdf raises several parser-specific exceptions; return a stable API error.
            return Response({"detail": f"could not read document: {exc}"}, status=status.HTTP_400_BAD_REQUEST)
        if not text:
            return Response({"detail": "document has no extractable text"}, status=status.HTTP_400_BAD_REQUEST)
        document = Document.objects.create(title=title, source_name=source_name, content=text, content_hash=sha256_text(text))
        index_document(document)
        return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)


class DocumentDetailView(APIView):
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def get_object(self, pk):
        return Document.objects.get(pk=pk)

    def get(self, request, pk):
        try:
            document = self.get_object(pk)
        except Document.DoesNotExist:
            return Response({"detail": "document not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(DocumentSerializer(document).data)

    @transaction.atomic
    def patch(self, request, pk):
        try:
            document = self.get_object(pk)
        except Document.DoesNotExist:
            return Response({"detail": "document not found"}, status=status.HTTP_404_NOT_FOUND)
        payload = request.data
        uploaded = payload.get("file")
        has_new_content = uploaded is not None or "content" in payload
        if has_new_content:
            try:
                text = extract_text(content=str(payload.get("content")) if uploaded is None else None, uploaded_file=uploaded, source_name=getattr(uploaded, "name", document.source_name))
            except Exception as exc:
                return Response({"detail": f"could not read document: {exc}"}, status=status.HTTP_400_BAD_REQUEST)
            if not text:
                return Response({"detail": "document has no extractable text"}, status=status.HTTP_400_BAD_REQUEST)
            document.content = text
            document.content_hash = sha256_text(text)
            document.version += 1
            document.source_name = str(payload.get("source_name") or getattr(uploaded, "name", document.source_name))
            document.is_active = True
        if "title" in payload:
            document.title = str(payload["title"]).strip() or document.title
        document.save()
        if has_new_content:
            index_document(document)
        return Response(DocumentSerializer(document).data)

    def put(self, request, pk):
        return self.patch(request, pk)

    def delete(self, request, pk):
        try:
            document = self.get_object(pk)
        except Document.DoesNotExist:
            return Response({"detail": "document not found"}, status=status.HTTP_404_NOT_FOUND)
        document.is_active = False
        document.save(update_fields=["is_active", "updated_at"])
        return Response(status=status.HTTP_204_NO_CONTENT)


class AskView(APIView):
    parser_classes = [JSONParser, FormParser]

    def post(self, request):
        question = str(request.data.get("question") or "").strip()
        if not question:
            return Response({"detail": "question is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            top_k = int(request.data.get("top_k", settings.RAG_TOP_K))
        except (TypeError, ValueError):
            return Response({"detail": "top_k must be an integer"}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"question": question, **answer_question(question, top_k)})


@api_view(["GET"])
def stats(request):
    return Response({"active_documents": Document.objects.filter(is_active=True).count(), "documents": Document.objects.count()})


OPENAPI_SCHEMA = {
    "openapi": "3.0.3",
    "info": {"title": "Document Q&A API", "version": "1.0.0", "description": "Citation-first RAG over active PDF, text and Markdown documents."},
    "servers": [{"url": "/"}],
    "paths": {
        "/api/documents/": {
            "get": {"summary": "List documents", "responses": {"200": {"description": "Document list"}}},
            "post": {"summary": "Create and index a document", "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/DocumentCreate"}}, "multipart/form-data": {"schema": {"$ref": "#/components/schemas/DocumentUpload"}}}}, "responses": {"201": {"description": "Created"}, "400": {"description": "Invalid document"}}},
        },
        "/api/documents/{id}/": {
            "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
            "get": {"summary": "Get a document", "responses": {"200": {"description": "Document"}, "404": {"description": "Not found"}}},
            "put": {"summary": "Replace document content and re-index", "responses": {"200": {"description": "Updated"}}},
            "patch": {"summary": "Edit and re-index a document", "responses": {"200": {"description": "Updated"}}},
            "delete": {"summary": "Deactivate a document", "responses": {"204": {"description": "Deactivated"}}},
        },
        "/api/ask/": {"post": {"summary": "Ask a citation-backed question", "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/AskRequest"}}}}, "responses": {"200": {"description": "Answer with sources"}}}},
        "/health/": {"get": {"summary": "Health check", "responses": {"200": {"description": "Healthy"}}}},
    },
    "components": {"schemas": {
        "DocumentCreate": {"type": "object", "required": ["title", "content"], "properties": {"title": {"type": "string"}, "content": {"type": "string"}, "source_name": {"type": "string"}}},
        "DocumentUpload": {"type": "object", "required": ["title", "file"], "properties": {"title": {"type": "string"}, "file": {"type": "string", "format": "binary"}, "source_name": {"type": "string"}}},
        "AskRequest": {"type": "object", "required": ["question"], "properties": {"question": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}}},
    }},
}
