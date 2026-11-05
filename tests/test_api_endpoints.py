import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
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
        from app.main import app
        yield TestClient(app)


def test_health_endpoint(_isolate_api):
    client = _isolate_api
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ollama" in data
    assert "embedding" in data


def test_sites_crud_api(_isolate_api):
    client = _isolate_api

    # 1. Create a site
    create_payload = {
        "root_url": "https://testsite.org/docs",
        "name": "Test Site Docs",
        "max_depth": 2,
        "max_pages": 10,
    }
    res_create = client.post("/api/sites", json=create_payload)
    assert res_create.status_code == 201
    site_data = res_create.json()
    site_id = site_data["id"]
    assert site_data["name"] == "Test Site Docs"
    assert site_data["root_url"] == "https://testsite.org/docs"

    # 2. Get site details
    res_get = client.get(f"/api/sites/{site_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == site_id

    # 3. List sites
    res_list = client.get("/api/sites")
    assert res_list.status_code == 200
    site_ids = [s["id"] for s in res_list.json()]
    assert site_id in site_ids

    # 4. Check crawl status endpoint
    res_status = client.get(f"/api/sites/{site_id}/crawl/status")
    assert res_status.status_code == 200
    assert res_status.json()["site_id"] == site_id

    # 5. Check chunks endpoint
    res_chunks = client.get(f"/api/sites/{site_id}/chunks")
    assert res_chunks.status_code == 200
    assert isinstance(res_chunks.json(), list)

    # 6. Delete site
    res_del = client.delete(f"/api/sites/{site_id}")
    assert res_del.status_code == 200

    # 7. Verify 404 after delete
    res_not_found = client.get(f"/api/sites/{site_id}")
    assert res_not_found.status_code == 404
