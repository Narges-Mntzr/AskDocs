from django.contrib import admin
from django.urls import include, path

from knowledge.views import DocsView, HealthView, SchemaView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("health/", HealthView.as_view(), name="health"),
    path("api/schema/", SchemaView.as_view(), name="api-schema"),
    path("api/docs/", DocsView.as_view(), name="api-docs"),
    path("api/", include("knowledge.urls")),
]
