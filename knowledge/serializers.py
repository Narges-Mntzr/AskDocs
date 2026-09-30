from rest_framework import serializers

from knowledge.models import Document


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.SerializerMethodField()

    def get_chunk_count(self, obj):
        return obj.chunks.count()

    class Meta:
        model = Document
        fields = ["id", "title", "source_name", "content", "version", "is_active", "chunk_count", "created_at", "updated_at"]
        read_only_fields = ["id", "version", "is_active", "chunk_count", "created_at", "updated_at"]
