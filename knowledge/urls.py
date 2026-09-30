from django.urls import path

from knowledge.views import (
    AskView,
    DocumentDetailView,
    DocumentListCreateView,
    StatsView,
)

urlpatterns = [
    path("documents/", DocumentListCreateView.as_view(), name="document-list"),
    path("documents/<int:pk>/", DocumentDetailView.as_view(), name="document-detail"),
    path("ask/", AskView.as_view(), name="ask"),
    path("stats/", StatsView.as_view(), name="stats"),
]
