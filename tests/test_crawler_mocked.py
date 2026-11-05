import time
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest
import requests

from app.crawler.crawler import CrawlItem, WebCrawler


class MockResponse:
    """Mock HTTP response for testing crawler without external network calls."""

    def __init__(
        self,
        text: str = "",
        content: Optional[bytes] = None,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}


@pytest.fixture
def mock_session():
    """Returns a requests.Session mock where .get() can be wired per URL."""
    session = MagicMock(spec=requests.Session)
    session.headers = {}
    return session


def test_crawl_output_format_and_pdf_discovery():
    """Verify crawler outputs list of (url, content_type, raw_bytes) and discovers linked PDFs."""
    html_root = """
    <html>
      <body>
        <a href="/about">About Us</a>
        <a href="/files/whitepaper.pdf">Download PDF</a>
      </body>
    </html>
    """
    html_about = "<html><body><h1>About Us</h1></body></html>"
    pdf_bytes = b"%PDF-1.4 sample pdf binary data"

    responses = {
        "https://example.com/robots.txt": MockResponse(status_code=404),
        "https://example.com": MockResponse(
            text=html_root,
            headers={"Content-Type": "text/html", "ETag": "root-etag", "Last-Modified": "Mon, 01 Jan 2026 00:00:00 GMT"},
        ),
        "https://example.com/about": MockResponse(
            text=html_about,
            headers={"Content-Type": "text/html", "ETag": "about-etag"},
        ),
        "https://example.com/files/whitepaper.pdf": MockResponse(
            content=pdf_bytes,
            headers={"Content-Type": "application/pdf", "ETag": "pdf-etag"},
        ),
    }

    session = MagicMock(spec=requests.Session)
    session.headers = {}
    session.get.side_effect = lambda url, **kwargs: responses.get(url, MockResponse(status_code=404))

    crawler = WebCrawler(
        root_url="https://example.com",
        max_depth=2,
        max_pages=10,
        rate_limit_delay=0.0,
        session=session,
    )

    results = crawler.crawl()

    # 1. Output format check: List of (url, content_type, raw_bytes)
    assert isinstance(results, list)
    assert len(results) == 3

    for item in results:
        assert isinstance(item, tuple)
        assert len(item) == 3
        url, content_type, raw_bytes = item
        assert isinstance(url, str)
        assert isinstance(content_type, str)
        assert isinstance(raw_bytes, bytes)

    # 2. Check URLs and content types
    url_to_ct = {url: ct for url, ct, _ in results}
    assert "https://example.com" in url_to_ct
    assert url_to_ct["https://example.com"] == "text/html"
    assert "https://example.com/about" in url_to_ct
    assert url_to_ct["https://example.com/about"] == "text/html"
    assert "https://example.com/files/whitepaper.pdf" in url_to_ct
    assert url_to_ct["https://example.com/files/whitepaper.pdf"] == "application/pdf"

    # 3. Check PDF raw bytes
    pdf_item = next(item for item in results if item[0] == "https://example.com/files/whitepaper.pdf")
    assert pdf_item[2] == pdf_bytes


def test_stay_within_same_domain():
    """Verify crawler strictly stays within root netloc and ignores external links."""
    html_root = """
    <html>
      <body>
        <a href="/internal">Internal Page</a>
        <a href="https://external.com/page">External Site</a>
        <a href="https://subdomain.example.com/page">Subdomain</a>
      </body>
    </html>
    """
    responses = {
        "https://example.com/robots.txt": MockResponse(status_code=404),
        "https://example.com": MockResponse(text=html_root),
        "https://example.com/internal": MockResponse(text="<html><body>Internal</body></html>"),
    }

    session = MagicMock(spec=requests.Session)
    session.headers = {}
    session.get.side_effect = lambda url, **kwargs: responses.get(url, MockResponse(status_code=404))

    crawler = WebCrawler(
        root_url="https://example.com",
        max_depth=2,
        rate_limit_delay=0.0,
        allow_subdomains=False,
        session=session,
    )

    results = crawler.crawl()
    crawled_urls = [r[0] for r in results]

    assert "https://example.com" in crawled_urls
    assert "https://example.com/internal" in crawled_urls
    assert "https://external.com/page" not in crawled_urls
    assert "https://subdomain.example.com/page" not in crawled_urls


def test_respect_robots_txt():
    """Verify crawler fetches robots.txt and does not request disallowed paths."""
    robots_content = """
    User-agent: *
    Disallow: /admin
    Disallow: /secret.pdf
    Allow: /admin/public
    """
    html_root = """
    <html>
      <body>
        <a href="/admin/dashboard">Admin Dashboard (Disallowed)</a>
        <a href="/secret.pdf">Secret PDF (Disallowed)</a>
        <a href="/public-page">Public Page (Allowed)</a>
      </body>
    </html>
    """

    responses = {
        "https://example.com/robots.txt": MockResponse(text=robots_content),
        "https://example.com": MockResponse(text=html_root),
        "https://example.com/public-page": MockResponse(text="<html><body>Public</body></html>"),
        "https://example.com/admin/dashboard": MockResponse(text="<html><body>Admin</body></html>"),
        "https://example.com/secret.pdf": MockResponse(content=b"%PDF secret"),
    }

    session = MagicMock(spec=requests.Session)
    session.headers = {}
    session.get.side_effect = lambda url, **kwargs: responses.get(url, MockResponse(status_code=404))

    crawler = WebCrawler(
        root_url="https://example.com",
        max_depth=2,
        rate_limit_delay=0.0,
        session=session,
    )

    results = crawler.crawl()
    crawled_urls = [r[0] for r in results]

    assert "https://example.com" in crawled_urls
    assert "https://example.com/public-page" in crawled_urls
    assert "https://example.com/admin/dashboard" not in crawled_urls
    assert "https://example.com/secret.pdf" not in crawled_urls


def test_rate_limiting():
    """Verify rate limit delay is respected between successive HTTP requests."""
    delay = 0.08  # 80ms
    html_root = '<html><body><a href="/page1">P1</a><a href="/page2">P2</a></body></html>'

    responses = {
        "https://example.com/robots.txt": MockResponse(status_code=404),
        "https://example.com": MockResponse(text=html_root),
        "https://example.com/page1": MockResponse(text="<html><body>1</body></html>"),
        "https://example.com/page2": MockResponse(text="<html><body>2</body></html>"),
    }

    session = MagicMock(spec=requests.Session)
    session.headers = {}
    session.get.side_effect = lambda url, **kwargs: responses.get(url, MockResponse(status_code=404))

    crawler = WebCrawler(
        root_url="https://example.com",
        max_depth=2,
        max_pages=3,
        rate_limit_delay=delay,
        session=session,
    )

    start_time = time.time()
    results = crawler.crawl()
    elapsed = time.time() - start_time

    assert len(results) == 3
    # 3 requests -> at least 2 rate limit intervals (2 * 0.08 = 0.16s)
    assert elapsed >= (2 * delay) * 0.85


def test_url_normalization_and_deduplication():
    """Verify URL defragmentation, query sorting, trailing slash normalization, and deduplication."""
    # Base normalization tests
    base = "https://example.com/docs/intro"

    # Strips fragments
    assert WebCrawler.normalize_url(base, "/page#section1") == "https://example.com/page"
    assert WebCrawler.normalize_url(base, "/page#section2") == "https://example.com/page"

    # Resolves relative paths
    assert WebCrawler.normalize_url(base, "../tutorial") == "https://example.com/tutorial"

    # Query sorting for identical endpoints
    url_a = WebCrawler.normalize_url(base, "/search?b=2&a=1")
    url_b = WebCrawler.normalize_url(base, "/search?a=1&b=2")
    assert url_a == url_b == "https://example.com/search?a=1&b=2"

    # Deduplication in crawling
    html_root = """
    <html>
      <body>
        <a href="/docs">Docs</a>
        <a href="/docs/">Docs with trailing slash</a>
        <a href="/docs#top">Docs with fragment</a>
        <a href="/docs?ref=1&utm=2">Docs with params</a>
        <a href="/docs?utm=2&ref=1">Docs with swapped params</a>
      </body>
    </html>
    """

    responses = {
        "https://example.com/robots.txt": MockResponse(status_code=404),
        "https://example.com": MockResponse(text=html_root),
        "https://example.com/docs": MockResponse(text="<html><body>Docs Page</body></html>"),
        "https://example.com/docs?ref=1&utm=2": MockResponse(text="<html><body>Docs Params</body></html>"),
    }

    session = MagicMock(spec=requests.Session)
    session.headers = {}
    session.get.side_effect = lambda url, **kwargs: responses.get(url, MockResponse(status_code=404))

    crawler = WebCrawler(
        root_url="https://example.com",
        max_depth=2,
        rate_limit_delay=0.0,
        session=session,
    )

    results = crawler.crawl()
    crawled_urls = [r[0] for r in results]

    # /docs, /docs/, /docs#top should only be crawled ONCE
    assert crawled_urls.count("https://example.com/docs") == 1
    # /docs?ref=1&utm=2 and /docs?utm=2&ref=1 should only be crawled ONCE
    assert crawled_urls.count("https://example.com/docs?ref=1&utm=2") == 1


def test_configurable_max_depth_and_max_pages():
    """Verify crawler strictly respects max_depth and max_pages constraints."""
    # Tree: root -> d1 -> d2 -> d3
    responses = {
        "https://example.com/robots.txt": MockResponse(status_code=404),
        "https://example.com": MockResponse(text='<a href="/d1">D1</a>'),
        "https://example.com/d1": MockResponse(text='<a href="/d2">D2</a>'),
        "https://example.com/d2": MockResponse(text='<a href="/d3">D3</a>'),
        "https://example.com/d3": MockResponse(text='<h1>D3 End</h1>'),
    }

    session = MagicMock(spec=requests.Session)
    session.headers = {}
    session.get.side_effect = lambda url, **kwargs: responses.get(url, MockResponse(status_code=404))

    # Test max_depth = 1 (should only crawl root and d1)
    crawler_depth = WebCrawler(
        root_url="https://example.com",
        max_depth=1,
        max_pages=10,
        rate_limit_delay=0.0,
        session=session,
    )
    results_depth = crawler_depth.crawl()
    urls_depth = [r[0] for r in results_depth]
    assert "https://example.com" in urls_depth
    assert "https://example.com/d1" in urls_depth
    assert "https://example.com/d2" not in urls_depth
    assert "https://example.com/d3" not in urls_depth

    # Test max_pages = 2 (should stop after 2 pages even if more discovered)
    crawler_pages = WebCrawler(
        root_url="https://example.com",
        max_depth=5,
        max_pages=2,
        rate_limit_delay=0.0,
        session=session,
    )
    results_pages = crawler_pages.crawl()
    assert len(results_pages) == 2


def test_retries_and_timeout_handling():
    """Verify retry handling on transient errors and timeout exception tolerance."""
    attempt_counter = {"/retry-page": 0}

    def mocked_get(url, timeout=10.0, **kwargs):
        if url.endswith("robots.txt"):
            return MockResponse(status_code=404)
        if url == "https://example.com" or url == "https://example.com/":
            return MockResponse(text='<a href="/retry-page">Retry</a><a href="/timeout-page">Timeout</a>')
        if url == "https://example.com/retry-page":
            attempt_counter["/retry-page"] += 1
            if attempt_counter["/retry-page"] == 1:
                return MockResponse(status_code=500)  # Fails first
            return MockResponse(text="Success after retry")  # Succeeds second
        if url == "https://example.com/timeout-page":
            raise requests.Timeout("Simulated connection timeout")
        return MockResponse(status_code=404)

    session = MagicMock(spec=requests.Session)
    session.headers = {}
    session.get.side_effect = mocked_get

    crawler = WebCrawler(
        root_url="https://example.com",
        max_depth=2,
        retries=2,
        timeout=2.0,
        rate_limit_delay=0.0,
        session=session,
    )

    # Crawler should complete gracefully without throwing on timeout or initial 500
    results = crawler.crawl()
    urls = [r[0] for r in results]

    assert "https://example.com" in urls
    # Timeout page failed gracefully and was skipped
    assert "https://example.com/timeout-page" not in urls


def test_store_etag_and_last_modified_headers():
    """Verify ETag and Last-Modified headers are captured and stored."""
    headers_root = {
        "Content-Type": "text/html",
        "ETag": '"abc123xyz"',
        "Last-Modified": "Fri, 24 Oct 2025 12:00:00 GMT",
    }
    headers_pdf = {
        "Content-Type": "application/pdf",
        "ETag": '"pdf999etag"',
        "Last-Modified": "Sun, 01 Nov 2025 08:30:00 GMT",
    }

    responses = {
        "https://example.com/robots.txt": MockResponse(status_code=404),
        "https://example.com": MockResponse(
            text='<a href="/doc.pdf">PDF</a>',
            headers=headers_root,
        ),
        "https://example.com/doc.pdf": MockResponse(
            content=b"%PDF sample",
            headers=headers_pdf,
        ),
    }

    session = MagicMock(spec=requests.Session)
    session.headers = {}
    session.get.side_effect = lambda url, **kwargs: responses.get(url, MockResponse(status_code=404))

    crawler = WebCrawler(
        root_url="https://example.com",
        max_depth=2,
        rate_limit_delay=0.0,
        session=session,
    )

    items: List[CrawlItem] = crawler.crawl_items()
    assert len(items) == 2

    root_item = next(i for i in items if i.url == "https://example.com")
    assert root_item.etag == '"abc123xyz"'
    assert root_item.last_modified == "Fri, 24 Oct 2025 12:00:00 GMT"

    pdf_item = next(i for i in items if i.url == "https://example.com/doc.pdf")
    assert pdf_item.etag == '"pdf999etag"'
    assert pdf_item.last_modified == "Sun, 01 Nov 2025 08:30:00 GMT"

    # Verify headers_store dictionary on the crawler instance
    assert "https://example.com" in crawler.headers_store
    assert crawler.headers_store["https://example.com"]["etag"] == '"abc123xyz"'
    assert crawler.headers_store["https://example.com"]["last_modified"] == "Fri, 24 Oct 2025 12:00:00 GMT"

    assert "https://example.com/doc.pdf" in crawler.headers_store
    assert crawler.headers_store["https://example.com/doc.pdf"]["etag"] == '"pdf999etag"'
