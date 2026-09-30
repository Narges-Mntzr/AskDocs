from django.conf import settings
from django.db import transaction
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from knowledge.models import Document
from knowledge.serializers import (
    AskRequestSerializer,
    DocumentListQuerySerializer,
    DocumentSerializer,
    DocumentUpdateSerializer,
    DocumentWriteSerializer,
)
from knowledge.services.general import (
    answer_question,
    extract_and_validate_text,
    index_document,
    sha256_text,
)


class DocumentListView(APIView):
    def get(self, request):
        query = DocumentListQuerySerializer(data=request.query_params)
        query.is_valid(raise_exception=True)
        queryset = Document.objects.all()
        if query.validated_data["active"] is not None:
            is_active = query.validated_data["active"]
            queryset = queryset.filter(is_active=is_active)
        return Response(DocumentSerializer(queryset, many=True).data)


class DocumentCreateView(APIView):
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    @transaction.atomic
    def post(self, request):
        serializer = DocumentWriteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        uploaded = data.get("file")
        content = data.get("content") or None
        source_name = data.get("source_name") or getattr(uploaded, "name", "")
        try:
            text = extract_and_validate_text(
                content=content,
                uploaded_file=None if content else uploaded,
                source_name=source_name,
            )
        except Exception as exc:
            # pypdf raises several parser-specific exceptions; return a stable API error.
            return Response(
                {"detail": f"could not read document: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not text:
            return Response(
                {"detail": "document has no extractable text"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        document = Document.objects.create(
            title=data["title"],
            source_name=source_name,
            content=text,
            content_hash=sha256_text(text),
        )
        index_document(document)
        return Response(
            DocumentSerializer(document).data, status=status.HTTP_201_CREATED
        )


class DocumentDetailView(APIView):
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def get_object(self, pk):
        return Document.objects.get(pk=pk)

    def get(self, request, pk):
        try:
            document = self.get_object(pk)
        except Document.DoesNotExist:
            return Response(
                {"detail": "document not found"}, status=status.HTTP_404_NOT_FOUND
            )
        return Response(DocumentSerializer(document).data)

    @transaction.atomic
    def patch(self, request, pk):
        try:
            document = self.get_object(pk)
        except Document.DoesNotExist:
            return Response(
                {"detail": "document not found"}, status=status.HTTP_404_NOT_FOUND
            )
        serializer = DocumentUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        uploaded = data.get("file")
        content = data.get("content") or None
        has_new_content = uploaded is not None or bool(content)
        if has_new_content:
            source_name = data.get("source_name") or getattr(
                uploaded, "name", document.source_name
            )
            try:
                text = extract_and_validate_text(
                    content=None if uploaded is not None else content,
                    uploaded_file=uploaded,
                    source_name=source_name,
                )
            except Exception as exc:
                return Response(
                    {"detail": f"could not read document: {exc}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if not text:
                return Response(
                    {"detail": "document has no extractable text"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            document.content = text
            document.content_hash = sha256_text(text)
            document.version += 1
            document.source_name = source_name
            document.is_active = True
        if "title" in data:
            document.title = data["title"]
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
            return Response(
                {"detail": "document not found"}, status=status.HTTP_404_NOT_FOUND
            )
        document.is_active = False
        document.save(update_fields=["is_active", "updated_at"])
        return Response(status=status.HTTP_204_NO_CONTENT)


class AskView(APIView):
    parser_classes = [JSONParser, FormParser]

    def post(self, request):
        serializer = AskRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        question = serializer.validated_data["question"].strip()
        top_k = serializer.validated_data.get("top_k", settings.RAG_TOP_K)
        return Response({"question": question, **answer_question(question, top_k)})


class StatsView(APIView):
    def get(self, request):
        return Response(
            {
                "active_documents": Document.objects.filter(is_active=True).count(),
                "documents": Document.objects.count(),
            }
        )
