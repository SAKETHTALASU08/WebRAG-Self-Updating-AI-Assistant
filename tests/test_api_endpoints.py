import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.models.schemas import SourceItem
from app.storage.site_manager import SiteManager


@pytest.fixture(autouse=True)
def _isolate_api(tmp_path):
    """Give the API its own temp data directory so FAISS state never leaks."""
    iso_manager = SiteManager(base_data_dir=tmp_path)
    with patch("app.api.routes.site_manager", iso_manager), \
         patch("app.api.routes.rag_pipeline") as mock_pipeline:
        # Wire the mock pipeline's site_manager so CRUD routes work
        mock_pipeline.site_manager = iso_manager
        mock_pipeline.ollama.check_health.return_value = False
        mock_pipeline.ollama.list_models.return_value = []
        mock_pipeline.embedder._device = "cpu"

        # Mock query_stream for /sites/{id}/chat endpoint
        def mock_query_stream(site_id, query, **kwargs):
            mock_sources = [
                SourceItem(
                    url="https://testsite.org/docs/auth",
                    title="Auth Guide",
                    snippet="Bearer tokens are required for authentication.",
                    page_no=1,
                )
            ]
            def token_gen():
                tokens = ["Authentication ", "requires ", "Bearer ", "tokens ", "[1]."]
                for t in tokens:
                    yield t
            return mock_sources, token_gen()

        mock_pipeline.query_stream.side_effect = mock_query_stream

        from app.main import app
        yield TestClient(app)


def test_health_endpoint(_isolate_api):
    client = _isolate_api
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ollama" in data
    assert "embedding" in data


def test_openapi_and_cors(_isolate_api):
    """Verify OpenAPI schema generation and CORS preflight headers."""
    client = _isolate_api

    # OpenAPI schema check
    res_openapi = client.get("/openapi.json")
    assert res_openapi.status_code == 200
    schema = res_openapi.json()
    assert schema["info"]["title"] == "WebRAG API"
    paths = schema["paths"]

    # Verify requested endpoint routes exist in OpenAPI documentation
    assert "/sites" in paths
    assert "/sites/{site_id}/status" in paths
    assert "/sites/{site_id}/refresh" in paths
    assert "/sites/{site_id}/chat" in paths
    assert "/sites/{site_id}" in paths

    # CORS OPTIONS preflight check
    res_cors = client.options(
        "/sites",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res_cors.status_code == 200
    assert res_cors.headers.get("access-control-allow-origin") in ["*", "http://localhost:3000"]


def test_site_input_validation(_isolate_api):
    """Verify robust input validation on site creation."""
    client = _isolate_api

    # 1. Invalid scheme (ftp://)
    res_bad_scheme = client.post("/sites", json={"root_url": "ftp://files.example.com"})
    assert res_bad_scheme.status_code == 422
    assert "http:// or https://" in str(res_bad_scheme.json())

    # 2. Invalid missing host/netloc
    res_no_host = client.post("/sites", json={"root_url": "https://"})
    assert res_no_host.status_code == 422

    # 3. Blank / empty URL
    res_empty = client.post("/sites", json={"root_url": "   "})
    assert res_empty.status_code == 422


def test_sites_endpoints_lifecycle(_isolate_api):
    """
    Test full lifecycle of requested endpoints:
    - POST /sites (root URL, starts crawl+index background job)
    - GET /sites/{id}/status
    - POST /sites/{id}/refresh (incremental update)
    - POST /sites/{id}/chat (streaming answer plus sources)
    - DELETE /sites/{id}
    """
    client = _isolate_api

    # 1. POST /sites (starts background job)
    create_payload = {
        "root_url": "https://my-docs.dev/guide",
        "name": "Dev Guide",
        "max_depth": 2,
        "max_pages": 20,
    }
    res_create = client.post("/sites", json=create_payload)
    assert res_create.status_code == 201
    site = res_create.json()
    site_id = site["id"]
    assert site["root_url"] == "https://my-docs.dev/guide"
    assert site["name"] == "Dev Guide"

    # 2. GET /sites/{id}/status
    res_status = client.get(f"/sites/{site_id}/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["site_id"] == site_id
    assert "status" in status_data
    assert "is_crawling" in status_data

    # 3. POST /sites/{id}/refresh (trigger incremental update)
    res_refresh = client.post(f"/sites/{site_id}/refresh")
    assert res_refresh.status_code == 202
    refresh_data = res_refresh.json()
    assert "Incremental refresh initiated" in refresh_data["message"]
    assert refresh_data["site_id"] == site_id

    # 4. POST /sites/{id}/chat (streaming answer plus sources)
    chat_payload = {
        "query": "How do I authenticate?",
        "top_k": 3,
        "similarity_threshold": 0.0,
    }
    res_chat = client.post(f"/sites/{site_id}/chat", json=chat_payload)
    assert res_chat.status_code == 200
    assert "text/event-stream" in res_chat.headers["content-type"]

    # Read SSE events from the response
    sse_text = res_chat.text
    assert "event: sources" in sse_text
    assert "event: token" in sse_text
    assert "event: done" in sse_text

    # Verify sources event payload contains {url, title, snippet, page_no}
    lines = sse_text.splitlines()
    sources_line = None
    for i, line in enumerate(lines):
        if line == "event: sources":
            sources_line = lines[i + 1]
            break

    assert sources_line is not None
    assert sources_line.startswith("data: ")
    sources_json = json.loads(sources_line[6:])
    assert len(sources_json) == 1
    src = sources_json[0]
    assert src["url"] == "https://testsite.org/docs/auth"
    assert src["title"] == "Auth Guide"
    assert "Bearer tokens are required" in src["snippet"]
    assert src["page_no"] == 1

    # Verify tokens stream
    token_events = [l[6:] for l in lines if l.startswith("data: {\"token\":")]
    assert len(token_events) > 0

    # 5. Chat validation error on blank question
    res_blank_chat = client.post(f"/sites/{site_id}/chat", json={"query": "   "})
    assert res_blank_chat.status_code == 422

    # 6. DELETE /sites/{id}
    res_delete = client.delete(f"/sites/{site_id}")
    assert res_delete.status_code == 200
    assert res_delete.json()["message"] == f"Site {site_id} deleted successfully"

    # 7. Check 404 after deletion on all site endpoints
    assert client.get(f"/sites/{site_id}").status_code == 404
    assert client.get(f"/sites/{site_id}/status").status_code == 404
    assert client.post(f"/sites/{site_id}/refresh").status_code == 404
    assert client.post(f"/sites/{site_id}/chat", json={"query": "test"}).status_code == 404
    assert client.delete(f"/sites/{site_id}").status_code == 404
