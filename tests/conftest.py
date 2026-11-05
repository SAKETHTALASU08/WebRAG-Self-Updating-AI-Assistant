import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator
import pytest

# Ensure OpenMP handles multiple runtimes without aborting on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import faiss
    faiss.omp_set_num_threads(1)
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

from app.storage.site_manager import SiteManager


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    temp_dir = Path(tempfile.mkdtemp(prefix="webrag_test_"))
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_site_manager(temp_data_dir: Path) -> SiteManager:
    return SiteManager(base_data_dir=temp_data_dir)
