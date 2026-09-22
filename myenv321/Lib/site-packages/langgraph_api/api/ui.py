import html
import json
import os
import re
from urllib.parse import quote

from anyio import open_file
from orjson import loads
from starlette.responses import Response
from starlette.routing import BaseRoute, Mount
from starlette.staticfiles import StaticFiles
from typing_extensions import TypedDict

from langgraph_api.js.ui import UI_PUBLIC_DIR, UI_SCHEMAS_FILE
from langgraph_api.route import ApiRequest, ApiRoute


class UiSchema(TypedDict):
    name: str
    assets: list[str]


_UI_SCHEMAS_CACHE: dict[str, UiSchema] | None = None


def _quote_path_segment(value: str) -> str:
    """Encode a dynamic URL path segment for safe HTML attribute use."""
    return quote(value, safe="")


def _html_safe_json_arg(value: object) -> str:
    """Serialize a value for a JavaScript argument inside an HTML attribute."""
    return html.escape(json.dumps(value), quote=True)


async def load_ui_schemas() -> dict[str, UiSchema]:
    """Load and cache UI schema mappings from JSON file."""
    global _UI_SCHEMAS_CACHE

    if _UI_SCHEMAS_CACHE is not None:
        return _UI_SCHEMAS_CACHE

    if not UI_SCHEMAS_FILE.exists():
        _UI_SCHEMAS_CACHE = {}
    else:
        async with await open_file(UI_SCHEMAS_FILE, mode="r") as f:
            _UI_SCHEMAS_CACHE = loads(await f.read())

    return _UI_SCHEMAS_CACHE


async def handle_ui(request: ApiRequest) -> Response:
    """Serve UI HTML with appropriate script/style tags."""
    graph_id = request.path_params["graph_id"]
    message = await request.json(schema=None)

    # Load UI file paths from schema
    schemas = await load_ui_schemas()

    if graph_id not in schemas:
        return Response(f"UI not found for graph '{graph_id}'", status_code=404)

    result = []
    for filepath in schemas[graph_id]["assets"]:
        basename = os.path.basename(filepath)
        ext = os.path.splitext(basename)[1]

        valid_js_name = re.sub(r"[^a-zA-Z0-9]", "_", graph_id)
        asset_url = (
            f"/ui/{_quote_path_segment(graph_id)}/{_quote_path_segment(basename)}"
        )
        safe_asset_url = html.escape(asset_url, quote=True)

        if ext == ".css":
            result.append(f'<link rel="stylesheet" href="{safe_asset_url}" />')
        elif ext == ".js":
            safe_name = _html_safe_json_arg(message["name"])
            result.append(
                f'<script src="{safe_asset_url}" '
                f"onload='__LGUI_{valid_js_name}.render("
                f'{safe_name}, "{{{{shadowRootId}}}}")\'>'
                "</script>"
            )

    return Response(content="\n".join(result), headers={"Content-Type": "text/html"})


ui_routes: list[BaseRoute] = [
    ApiRoute("/ui/{graph_id}", handle_ui, methods=["POST"]),
    Mount("/ui", StaticFiles(directory=UI_PUBLIC_DIR, check_dir=False)),
]
