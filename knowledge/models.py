from django.db import models


class Document(models.Model):
    title = models.TextField(max_length=255)
    source_name = models.TextField(max_length=255, blank=True)
    content = models.TextField(max_length=1_000_000)
    content_hash = models.TextField(max_length=64)
    version = models.PositiveIntegerField(default=1)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "سند"
        verbose_name_plural = "اسناد"

    def __str__(self) -> str:
        return self.title


class Chunk(models.Model):
    document = models.ForeignKey(
        Document, related_name="chunks", on_delete=models.CASCADE
    )
    ordinal = models.PositiveIntegerField()
    text = models.TextField(max_length=5_000)
    embedding = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "قطعه سند"
        verbose_name_plural = "قطعه‌های سند"
        constraints = [
            models.UniqueConstraint(
                fields=["document", "ordinal"], name="unique_document_chunk_ordinal"
            )
        ]

    def __str__(self) -> str:
        return f"{self.document.title}:{self.ordinal}"
