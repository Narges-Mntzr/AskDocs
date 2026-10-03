import re

from rest_framework import serializers
from rest_framework.schemas.openapi import AutoSchema


class SerializerSchema(AutoSchema):
    """Build OpenAPI operations from serializer classes declared on the view."""

    def get_operation_id(self, path, method):
        name = self.view.__class__.__name__
        for suffix in ("APIView", "View"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return f"{method.lower()}{name}"

    def get_description(self, path, method):
        method_func = getattr(self.view, method.lower(), None)
        docstring = getattr(method_func, "__doc__", None)
        if not docstring:
            return ""
        return docstring.strip()

    def get_path_parameters(self, path, method):
        parameters = []
        for name in re.findall(r"\{(\w+)\}", path):
            schema = {"type": "integer"} if name in {"pk", "id"} else {"type": "string"}
            parameters.append(
                {
                    "name": name,
                    "in": "path",
                    "required": True,
                    "schema": schema,
                }
            )
        return parameters

    def allows_filters(self, path, method):
        if self._query_serializer_class(method) is not None:
            return True
        return super().allows_filters(path, method)

    def get_filter_parameters(self, path, method):
        serializer_class = self._query_serializer_class(method)
        if serializer_class is None:
            return super().get_filter_parameters(path, method)
        return self._query_parameters(serializer_class())

    def get_request_serializer(self, path, method):
        if method not in ("PUT", "PATCH", "POST"):
            return None
        serializer_class = getattr(self.view, "request_serializer_class", None)
        if serializer_class is None:
            return None
        return serializer_class()

    def get_response_serializer(self, path, method):
        if method == "DELETE":
            return None
        serializer_class = getattr(self.view, "response_serializer_class", None)
        if serializer_class is None:
            return None
        return serializer_class()

    def get_request_body(self, path, method):
        body = super().get_request_body(path, method)
        serializer = self.get_request_serializer(path, method)
        if isinstance(serializer, serializers.Serializer) and body:
            body["required"] = any(
                field.required for field in serializer.fields.values()
            )
        return body

    def get_responses(self, path, method):
        status_overrides = getattr(self.view, "schema_status_codes", {})
        description = self.get_description(path, method)
        if method == "DELETE":
            return {status_overrides.get("DELETE", "204"): {"description": description}}

        self.response_media_types = self.map_renderers(path, method)
        serializer = self.get_response_serializer(path, method)
        if not isinstance(serializer, serializers.Serializer):
            response_schema = {}
        else:
            response_schema = self.get_reference(serializer)
        if getattr(self.view, "response_many", False):
            response_schema = {"type": "array", "items": response_schema}

        status_code = status_overrides.get(method)
        if status_code is None:
            status_code = "201" if method == "POST" else "200"
        return {
            status_code: {
                "content": {
                    content_type: {"schema": response_schema}
                    for content_type in self.response_media_types
                },
                "description": description,
            }
        }

    def map_field(self, field):
        if isinstance(field, serializers.SerializerMethodField):
            method = getattr(field.parent, field.method_name, None)
            return_type = getattr(method, "__annotations__", {}).get("return")
            if return_type is int:
                return {"type": "integer"}
        return super().map_field(field)

    def _query_serializer_class(self, method):
        if method.lower() != "get":
            return None
        return getattr(self.view, "query_serializer_class", None)

    def _query_parameters(self, serializer):
        parameters = []
        for field in serializer.fields.values():
            if isinstance(field, serializers.HiddenField):
                continue
            schema = self.map_field(field)
            if field.allow_null:
                schema["nullable"] = True
            parameter = {
                "name": field.field_name,
                "required": bool(field.required),
                "in": "query",
                "schema": schema,
            }
            if field.help_text:
                parameter["description"] = str(field.help_text)
            parameters.append(parameter)
        return parameters
