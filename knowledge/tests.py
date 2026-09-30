from unittest.mock import patch

from django.contrib import admin
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from knowledge.models import Chunk, Document


def fake_embeddings(texts, model=None):
    return [[1.0, 0.0] for _ in texts]


@override_settings(EMBEDDING_API_KEY="")
class DocumentApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        embed = patch(
            "knowledge.services.general.create_embeddings", side_effect=fake_embeddings
        )
        embed.start()
        self.addCleanup(embed.stop)

    def test_create_ask_update_and_delete_lifecycle(self):
        created = self.client.post(
            "/knowledge/documents/create/",
            {
                "title": "راهنما",
                "content": "Django برای ساخت API استفاده می‌شود.\n\nDocker برنامه را اجرا می‌کند.",
            },
            format="json",
        )
        self.assertEqual(created.status_code, 201)
        document_id = created.data["id"]
        answer = self.client.post(
            "/knowledge/ask/",
            {"question": "برنامه با چه چیزی اجرا می‌شود؟"},
            format="json",
        )
        self.assertEqual(answer.status_code, 200)
        self.assertTrue(answer.data["sufficient_information"])
        self.assertEqual(answer.data["sources"][0]["document_id"], document_id)
        updated = self.client.patch(
            f"/knowledge/documents/{document_id}/",
            {"content": "این متن جدید است."},
            format="json",
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.data["version"], 2)
        self.assertEqual(
            self.client.delete(f"/knowledge/documents/{document_id}/").status_code, 204
        )
        self.assertEqual(Document.objects.get(pk=document_id).is_active, False)

    def test_unknown_question_is_explicit(self):
        response = self.client.post(
            "/knowledge/ask/", {"question": "چیزی که در سند نیست؟"}, format="json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["sufficient_information"])

    def test_rejects_invalid_document_and_question_input(self):
        missing_title = self.client.post(
            "/knowledge/documents/create/", {"content": "متن"}, format="json"
        )
        self.assertEqual(missing_title.status_code, 400)
        missing_body = self.client.post(
            "/knowledge/documents/create/", {"title": "عنوان"}, format="json"
        )
        self.assertEqual(missing_body.status_code, 400)
        blank_question = self.client.post(
            "/knowledge/ask/", {"question": "   "}, format="json"
        )
        self.assertEqual(blank_question.status_code, 400)
        bad_top_k = self.client.post(
            "/knowledge/ask/", {"question": "سلام", "top_k": 0}, format="json"
        )
        self.assertEqual(bad_top_k.status_code, 400)

    def test_create_document_from_text_file(self):
        upload = SimpleUploadedFile("note.txt", "متن داخل فایل.".encode())
        response = self.client.post(
            "/knowledge/documents/create/",
            {"title": "فایل", "file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, 201)
        self.assertIn("متن داخل فایل.", response.data["content"])
        self.assertEqual(response.data["source_name"], "note.txt")

    def test_models_are_registered_in_admin(self):
        self.assertIn(Document, admin.site._registry)
        self.assertIn(Chunk, admin.site._registry)
