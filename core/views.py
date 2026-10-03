from django.conf import settings
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView


class HealthView(APIView):
    def get(self, request):
        return Response(
            {"status": "ok", "embedding_configured": bool(settings.EMBEDDING_API_KEY)}
        )


class SchemaView(APIView):
    def get(self, request):
        return Response(OPENAPI_SCHEMA)


class DocsView(APIView):
    def get(self, request):
        html = """<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Knowledge API</title>
        <link rel='stylesheet' href='https://unpkg.com/swagger-ui-dist/swagger-ui.css'></head><body>
        <div id='swagger-ui'></div><script src='https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js'></script>
        <script>window.ui=SwaggerUIBundle({url:'/api/schema/',dom_id:'#swagger-ui',deepLinking:true,presets:[SwaggerUIBundle.presets.apis,SwaggerUIBundle.SwaggerUIStandalonePreset]});</script>
        </body></html>"""
        return HttpResponse(html)


OPENAPI_SCHEMA = {
    "openapi": "3.0.3",
    "info": {
        "title": "Document Q&A API",
        "version": "1.0.0",
        "description": "Citation-first RAG over active PDF, text and Markdown documents.",
    },
    "servers": [{"url": "/"}],
    "paths": {
        "/knowledge/documents/": {
            "get": {
                "summary": "List documents",
                "responses": {"200": {"description": "Document list"}},
            },
        },
        "/knowledge/documents/create/": {
            "post": {
                "summary": "Create and index a document",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DocumentCreate"}
                        },
                        "multipart/form-data": {
                            "schema": {"$ref": "#/components/schemas/DocumentUpload"}
                        },
                    },
                },
                "responses": {
                    "201": {"description": "Created"},
                    "400": {"description": "Invalid document"},
                },
            },
        },
        "/knowledge/documents/{id}/": {
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "integer"},
                }
            ],
            "get": {
                "summary": "Get a document",
                "responses": {
                    "200": {"description": "Document"},
                    "404": {"description": "Not found"},
                },
            },
            "put": {
                "summary": "Replace document content and re-index",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DocumentUpdate"}
                        },
                        "multipart/form-data": {
                            "schema": {
                                "$ref": "#/components/schemas/DocumentUpdateUpload"
                            }
                        },
                    },
                },
                "responses": {"200": {"description": "Updated"}},
            },
            "post": {
                "summary": "Edit and re-index a document",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DocumentUpdate"}
                        },
                        "multipart/form-data": {
                            "schema": {
                                "$ref": "#/components/schemas/DocumentUpdateUpload"
                            }
                        },
                    },
                },
                "responses": {"200": {"description": "Updated"}},
            },
            "delete": {
                "summary": "Deactivate a document",
                "responses": {"200": {"description": "Deactivated"}},
            },
        },
        "/knowledge/ask/": {
            "post": {
                "summary": "Ask a citation-backed question",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AskRequest"}
                        }
                    },
                },
                "responses": {"200": {"description": "Answer with sources"}},
            }
        },
        "/health/": {
            "get": {
                "summary": "Health check",
                "responses": {"200": {"description": "Healthy"}},
            }
        },
    },
    "components": {
        "schemas": {
            "DocumentCreate": {
                "type": "object",
                "required": ["title", "content"],
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
            "DocumentUpload": {
                "type": "object",
                "required": ["title", "file"],
                "properties": {
                    "title": {"type": "string"},
                    "file": {"type": "string", "format": "binary"},
                },
            },
            "DocumentUpdate": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
            "DocumentUpdateUpload": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "file": {"type": "string", "format": "binary"},
                },
            },
            "AskRequest": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                },
            },
        }
    },
}
