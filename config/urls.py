from django.urls import include, path

from knowledge.views import api_docs, api_schema, health

urlpatterns = [
    path("health/", health, name="health"),
    path("api/health/", health, name="api-health"),
    path("api/schema/", api_schema, name="api-schema"),
    path("api/docs/", api_docs, name="api-docs"),
    path("api/", include("knowledge.urls")),
]
