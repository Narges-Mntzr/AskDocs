from django.urls import path

from core.views import DocsView, HealthView, SchemaView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("api/schema/", SchemaView.as_view(), name="api-schema"),
    path("api/docs/", DocsView.as_view(), name="api-docs"),
]
