import io
from unittest.mock import patch

from django.contrib import admin
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from rest_framework.test import APIClient

from knowledge.models import Chunk, Document
from knowledge.services.general import embed_in_batches, split_chunks


def fake_embeddings(texts, model=None):
    return [[1.0, 0.0] for _ in texts]


def pdf_with_jittered_line() -> bytes:
    """Two glyphs of one line, plus a second line. Y differs by a fraction of a point."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=200)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 16 Tf "
        b"1 0 0 1 40 100 Tm (Hello) Tj "
        b"1 0 0 1 0 0 Tm (year) Tj "
        b"1 0 0 1 120 100.6 Tm (world) Tj "
        b"1 0 0 1 40 70 Tm (Second) Tj ET"
    )
    page[NameObject("/Contents")] = stream
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class SplitChunksTests(SimpleTestCase):
    def test_each_nonempty_line_is_its_own_chunk(self):
        chunks = split_chunks("خط اول\nخط دوم\n\nخط سوم")
        self.assertEqual(chunks, ["خط اول", "خط دوم", "خط سوم"])

    def test_long_line_is_split(self):
        chunks = split_chunks("ا" * 3000)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 2200 for chunk in chunks))

    def test_embedding_requests_stay_small(self):
        with patch(
            "knowledge.services.general.create_embeddings",
            side_effect=lambda texts, model=None: [[1.0] for _ in texts],
        ) as embed:
            vectors = embed_in_batches(["متن"] * 10)
        self.assertEqual(len(vectors), 10)
        self.assertEqual([len(call.args[0]) for call in embed.call_args_list], [4, 4, 2])


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
        updated = self.client.post(
            f"/knowledge/documents/{document_id}/",
            {"content": "این متن جدید است."},
            format="json",
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.data["version"], 2)
        self.assertEqual(
            self.client.delete(f"/knowledge/documents/{document_id}/").status_code, 200
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

    def test_markdown_upload(self):
        upload = SimpleUploadedFile(
            "input1.md",
            "# عنوان\n\nمتن اظهارنامه.".encode(),
            content_type="text/markdown",
        )
        response = self.client.post(
            "/knowledge/documents/create/",
            {"title": "اظهارنامه قضایی", "file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, 201)
        self.assertIn("متن اظهارنامه.", response.data["content"])
        self.assertNotIn("source_name", response.data)

    def test_pdf_upload_joins_words_on_the_same_line(self):
        upload = SimpleUploadedFile(
            "note.pdf", pdf_with_jittered_line(), content_type="application/pdf"
        )
        response = self.client.post(
            "/knowledge/documents/create/",
            {"title": "حافظ", "content": "", "file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["content"], "Hello year world\nSecond")
        self.assertEqual(response.data["chunk_count"], 2)

    def test_update_document_from_file(self):
        created = self.client.post(
            "/knowledge/documents/create/",
            {"title": "قدیمی", "content": "متن اولیه."},
            format="json",
        )
        upload = SimpleUploadedFile(
            "input1.md",
            "متن جایگزین از فایل.".encode(),
            content_type="text/markdown",
        )
        response = self.client.post(
            f"/knowledge/documents/{created.data['id']}/",
            {"file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["content"], "متن جایگزین از فایل.")
        self.assertEqual(response.data["version"], 2)
        self.assertEqual(response.data["chunk_count"], 1)

    def test_create_document_from_text_file(self):
        upload = SimpleUploadedFile("note.txt", "متن داخل فایل.".encode())
        response = self.client.post(
            "/knowledge/documents/create/",
            {"title": "فایل", "file": upload},
            format="multipart",
        )
        self.assertEqual(response.status_code, 201)
        self.assertIn("متن داخل فایل.", response.data["content"])
        self.assertNotIn("source_name", response.data)

    def test_admin_session_does_not_require_csrf_token(self):
        User.objects.create_user(username="admin", password="pass")
        client = APIClient(enforce_csrf_checks=True)
        client.login(username="admin", password="pass")
        response = client.post(
            "/knowledge/documents/create/",
            {"title": "راهنما", "content": "متن سند برای ایندکس."},
            format="json",
        )
        self.assertEqual(response.status_code, 201)

    def test_schema_comes_from_serializers(self):
        schema = self.client.get("/api/schema/").json()
        ask = schema["components"]["schemas"]["AskRequest"]
        self.assertEqual(ask["required"], ["question"])
        self.assertEqual(ask["properties"]["top_k"]["maximum"], 20)
        self.assertEqual(ask["properties"]["question"]["maxLength"], 2000)
        document_write = schema["components"]["schemas"]["DocumentWrite"]
        self.assertIn("file", document_write["properties"])
        self.assertEqual(document_write["properties"]["file"]["format"], "binary")
        ask_path = schema["paths"]["/knowledge/ask/"]["post"]
        self.assertEqual(
            ask_path["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/AskRequest",
        )
        self.assertIn(
            "active",
            {
                parameter["name"]
                for parameter in schema["paths"]["/knowledge/documents/"]["get"][
                    "parameters"
                ]
            },
        )

    def test_models_are_registered_in_admin(self):
        self.assertIn(Document, admin.site._registry)
        self.assertIn(Chunk, admin.site._registry)
