"""WebRAG - Self-updating AI assistant generated from a website's root URL."""
import os

# Prevent OpenMP runtime conflict between PyTorch and FAISS on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import faiss
    faiss.omp_set_num_threads(1)
except Exception:
    pass

__version__ = "1.0.0"

