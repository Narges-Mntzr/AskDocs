from rest_framework import serializers

from knowledge.models import Document

CONTENT_MAX_LENGTH = 1_000_000


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.SerializerMethodField()

    def get_chunk_count(self, obj):
        return obj.chunks.count()

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "source_name",
            "content",
            "version",
            "is_active",
            "chunk_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "version",
            "is_active",
            "chunk_count",
            "created_at",
            "updated_at",
        ]


class DocumentListQuerySerializer(serializers.Serializer):
    active = serializers.BooleanField(required=False, allow_null=True, default=None)


class DocumentWriteSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255)
    content = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, max_length=CONTENT_MAX_LENGTH
    )
    file = serializers.FileField(required=False, allow_empty_file=False)
    source_name = serializers.CharField(
        required=False, allow_blank=True, max_length=255
    )

    def validate(self, attrs):
        content = attrs.get("content")
        if isinstance(content, str):
            content = content.strip()
            attrs["content"] = content
        has_text = bool(content)
        has_file = attrs.get("file") is not None
        if not has_text and not has_file:
            raise serializers.ValidationError("One of content or file is required.")
        source_name = attrs.get("source_name", "")
        attrs["source_name"] = source_name.strip()
        return attrs


class DocumentUpdateSerializer(serializers.Serializer):
    title = serializers.CharField(required=False, max_length=255)
    content = serializers.CharField(
        required=False, allow_blank=True, allow_null=True, max_length=CONTENT_MAX_LENGTH
    )
    file = serializers.FileField(required=False, allow_empty_file=False)
    source_name = serializers.CharField(
        required=False, allow_blank=True, max_length=255
    )

    def validate(self, attrs):
        if "title" in attrs:
            title = attrs["title"].strip()
            if not title:
                raise serializers.ValidationError(
                    {"title": "This field may not be blank."}
                )
            attrs["title"] = title
        if "content" in attrs:
            content = attrs["content"]
            attrs["content"] = content.strip() if isinstance(content, str) else None
        if "source_name" in attrs:
            attrs["source_name"] = attrs["source_name"].strip()
        content = attrs.get("content")
        has_file = attrs.get("file") is not None
        if "content" in attrs and not content and not has_file:
            raise serializers.ValidationError("document has no extractable text")
        return attrs


class AskRequestSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=2_000)
    top_k = serializers.IntegerField(required=False, min_value=1, max_value=20)
