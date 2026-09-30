from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from knowledge.models import Document


@override_settings(EMBEDDING_API_KEY="")
class DocumentApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_create_ask_update_and_delete_lifecycle(self):
        created = self.client.post("/api/documents/", {"title": "راهنما", "content": "Django برای ساخت API استفاده می‌شود.\n\nDocker برنامه را اجرا می‌کند."}, format="json")
        self.assertEqual(created.status_code, 201)
        document_id = created.data["id"]
        answer = self.client.post("/api/ask/", {"question": "برنامه با چه چیزی اجرا می‌شود؟"}, format="json")
        self.assertEqual(answer.status_code, 200)
        self.assertTrue(answer.data["sufficient_information"])
        self.assertEqual(answer.data["sources"][0]["document_id"], document_id)
        updated = self.client.patch(f"/api/documents/{document_id}/", {"content": "این متن جدید است."}, format="json")
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.data["version"], 2)
        self.assertEqual(self.client.delete(f"/api/documents/{document_id}/").status_code, 204)
        self.assertEqual(Document.objects.get(pk=document_id).is_active, False)

    def test_unknown_question_is_explicit(self):
        response = self.client.post("/api/ask/", {"question": "چیزی که در سند نیست؟"}, format="json")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["sufficient_information"])
