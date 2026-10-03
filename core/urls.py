from django.urls import path
from rest_framework.renderers import JSONOpenAPIRenderer
from rest_framework.schemas import get_schema_view

from core.views import DocsView, HealthView

schema_view = get_schema_view(
    title="Document Q&A API",
    description="Citation-first RAG over active PDF, text and Markdown documents.",
    version="1.0.0",
    public=True,
    renderer_classes=[JSONOpenAPIRenderer],
)

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("api/schema/", schema_view, name="api-schema"),
    path("api/docs/", DocsView.as_view(), name="api-docs"),
]
