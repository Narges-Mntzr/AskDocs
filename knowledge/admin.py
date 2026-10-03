from django.contrib import admin
from django.db.models import Count

from knowledge.models import Chunk, Document


class ChunkInline(admin.TabularInline):
    model = Chunk
    extra = 0
    fields = ("ordinal", "text", "created_at")
    readonly_fields = ("ordinal", "text", "created_at")
    can_delete = False

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "version",
        "is_active",
        "chunk_count",
        "updated_at",
    )
    list_filter = ("is_active",)
    search_fields = ("title", "content")
    readonly_fields = ("content", "content_hash", "version", "created_at", "updated_at")
    inlines = (ChunkInline,)

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(chunk_total=Count("chunks"))

    @admin.display(description="تعداد قطعه‌ها", ordering="chunk_total")
    def chunk_count(self, obj: Document) -> int:
        return obj.chunk_total


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "ordinal", "created_at")
    list_select_related = ("document",)
    list_filter = ("document",)
    search_fields = ("text", "document__title")
    readonly_fields = ("document", "ordinal", "text", "embedding", "created_at")

    def has_add_permission(self, request):
        return False
