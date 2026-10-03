from django.conf import settings
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView


class HealthView(APIView):
    def get(self, request):
        """Health check."""
        return Response(
            {"status": "ok", "embedding_configured": bool(settings.EMBEDDING_API_KEY)}
        )


class DocsView(APIView):
    schema = None

    def get(self, request):
        html = """<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Knowledge API</title>
        <link rel='stylesheet' href='https://unpkg.com/swagger-ui-dist/swagger-ui.css'></head><body>
        <div id='swagger-ui'></div><script src='https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js'></script>
        <script>window.ui=SwaggerUIBundle({url:'/api/schema/',dom_id:'#swagger-ui',deepLinking:true,presets:[SwaggerUIBundle.presets.apis,SwaggerUIBundle.SwaggerUIStandalonePreset]});</script>
        </body></html>"""
        return HttpResponse(html)
