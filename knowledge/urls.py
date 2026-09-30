from django.urls import path

from knowledge.views import (
    AskView,
    DocumentCreateView,
    DocumentDetailView,
    DocumentListView,
    StatsView,
)

urlpatterns = [
    path("documents/", DocumentListView.as_view(), name="document-list"),
    path("documents/create/", DocumentCreateView.as_view(), name="document-create"),
    path("documents/<int:pk>/", DocumentDetailView.as_view(), name="document-detail"),
    path("ask/", AskView.as_view(), name="ask"),
    path("stats/", StatsView.as_view(), name="stats"),
]
