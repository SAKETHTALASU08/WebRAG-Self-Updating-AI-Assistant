import logging
import posixpath
import re
import time
from collections import deque
from typing import Any, Callable, Dict, Generator, List, NamedTuple, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import settings
from app.crawler.extractor import ContentExtractor, ExtractedDocument

logger = logging.getLogger(__name__)


class CrawlItem(NamedTuple):
    """Container for crawled resource with headers metadata."""
    url: str
    content_type: str
    raw_bytes: bytes
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    status_code: int = 200
    is_modified: bool = True


class WebCrawler:
    """
    Robust domain-scoped web crawler.
    
    Features:
    - Scoped strictly to the same domain (netloc)
    - Respects robots.txt rules
    - Rate limiting with configurable delay between requests
    - URL normalization and deduplication
    - Configurable max_depth and max_pages
    - Retries with backoff and configurable timeouts
    - Captures and stores ETag / Last-Modified headers
    - Discovers HTML pages and linked PDFs
    - Outputs list of (url, content_type, raw_bytes)
    """

    def __init__(
        self,
        root_url: str,
        max_depth: int = 3,
        max_pages: int = 50,
        rate_limit_delay: float = 0.2,
        delay_seconds: Optional[float] = None,
        timeout: float = 10.0,
        retries: int = 3,
        backoff_factor: float = 0.3,
        user_agent: Optional[str] = None,
        allow_subdomains: bool = False,
        session: Optional[requests.Session] = None,
        known_headers: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    ):
        self.raw_root_url = root_url.strip()
        self.root_url = self.normalize_url(self.raw_root_url, self.raw_root_url) or self.raw_root_url
        parsed_root = urlparse(self.root_url)
        self.root_netloc = parsed_root.netloc.lower()
        self.root_scheme = parsed_root.scheme.lower() or "https"
        
        self.max_depth = max_depth
        self.max_pages = max_pages
        # Support either rate_limit_delay or delay_seconds
        self.rate_limit_delay = delay_seconds if delay_seconds is not None else rate_limit_delay
        self.timeout = timeout
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.user_agent = user_agent or settings.USER_AGENT
        self.allow_subdomains = allow_subdomains

        # Stored headers for conditional requests: url -> {"etag": ..., "last_modified": ...}
        self.known_headers: Dict[str, Dict[str, Optional[str]]] = dict(known_headers or {})
        self.pages_skipped_304: int = 0

        # Tracking sets
        self.visited_urls: Set[str] = set()
        self.enqueued_urls: Set[str] = set()
        
        # Headers store: url -> {"etag": ..., "last_modified": ...}
        self.headers_store: Dict[str, Dict[str, Optional[str]]] = {}

        # Last request timestamp for rate limiting
        self._last_request_time: float = 0.0

        # Session configuration with retries
        self.session = session or self._create_session()

        # Robots.txt parser
        self.robot_parser: Optional[RobotFileParser] = None
        self._init_robots_txt()

    def _create_session(self) -> requests.Session:
        """Create requests session with robust retry strategy."""
        session = requests.Session()
        session.headers.update({"User-Agent": self.user_agent})
        
        retry_strategy = Retry(
            total=self.retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _init_robots_txt(self) -> None:
        """Fetch and parse robots.txt from the root host."""
        robots_url = f"{self.root_scheme}://{self.root_netloc}/robots.txt"
        try:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            resp = self.session.get(robots_url, timeout=self.timeout)
            if resp.status_code == 200 and resp.text:
                rp.parse(resp.text.splitlines())
                self.robot_parser = rp
                logger.info(f"Successfully loaded robots.txt from {robots_url}")
            else:
                self.robot_parser = None
        except Exception as e:
            logger.debug(f"robots.txt not accessible at {robots_url}: {e}")
            self.robot_parser = None

    def is_allowed_by_robots(self, url: str) -> bool:
        """Check if URL fetch is permitted under robots.txt rules."""
        if not self.robot_parser:
            return True
        try:
            return self.robot_parser.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the target domain (or subdomain if allowed)."""
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc.lower()
            if not netloc:
                return False
            if self.allow_subdomains:
                return netloc == self.root_netloc or netloc.endswith(f".{self.root_netloc}")
            return netloc == self.root_netloc
        except Exception:
            return False

    @classmethod
    def normalize_url(cls, base_url: str, link: str) -> Optional[str]:
        """
        Normalize and resolve a link against a base URL:
        - Resolves relative URLs to absolute
        - Strips query fragments (#...)
        - Normalizes scheme and host to lowercase
        - Strips default ports (:80, :443)
        - Normalizes path (resolves '/../', strips redundant trailing slashes)
        - Sorts query parameters for consistent deduplication
        - Rejects non-HTTP(S) schemes
        """
        if not link:
            return None
        link = link.strip()
        if link.startswith(("mailto:", "tel:", "javascript:", "data:", "#")):
            return None

        try:
            resolved = urljoin(base_url, link)
            parsed = urlparse(resolved)
            
            scheme = parsed.scheme.lower()
            if scheme not in ("http", "https"):
                return None

            netloc = parsed.netloc.lower()
            if not netloc:
                return None

            # Strip default ports
            if netloc.endswith(":80") and scheme == "http":
                netloc = netloc[:-3]
            elif netloc.endswith(":443") and scheme == "https":
                netloc = netloc[:-4]

            # Path normalization
            path = parsed.path or "/"
            # Normalize redundant slashes or relative components
            normalized_path = posixpath.normpath(path)
            if path.endswith("/") and not normalized_path.endswith("/"):
                normalized_path += "/"
            if not normalized_path.startswith("/"):
                normalized_path = "/" + normalized_path

            # Strip trailing slash, normalize root '/' to ''
            if normalized_path == "/":
                normalized_path = ""
            elif normalized_path.endswith("/"):
                normalized_path = normalized_path.rstrip("/")

            # Canonicalize query parameters (sorted order)
            query = ""
            if parsed.query:
                params = parse_qsl(parsed.query, keep_blank_values=True)
                params.sort()
                query = urlencode(params)

            # Omit fragment completely
            normalized = urlunparse((scheme, netloc, normalized_path, "", query, ""))
            return normalized
        except Exception as e:
            logger.debug(f"Failed to normalize link '{link}' against '{base_url}': {e}")
            return None

    def should_crawl(self, url: str) -> bool:
        """Filter URLs to ensure they meet domain, robots, and media criteria."""
        if not url:
            return False
        if url in self.visited_urls or url in self.enqueued_urls:
            return False
        if not self.is_same_domain(url):
            return False
        if not self.is_allowed_by_robots(url):
            return False

        # Filter out common binary/media extensions except PDFs
        ignored_extensions = (
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp", ".tiff",
            ".mp4", ".mp3", ".wav", ".avi", ".mov", ".mkv", ".webm", ".flv",
            ".zip", ".tar", ".gz", ".rar", ".7z", ".exe", ".dmg", ".pkg", ".iso",
            ".css", ".js", ".json", ".xml", ".rss", ".woff", ".woff2", ".ttf", ".eot"
        )
        path = urlparse(url).path.lower()
        if any(path.endswith(ext) for ext in ignored_extensions):
            return False

        return True

    def _apply_rate_limit(self) -> None:
        """Enforce polite rate limiting between successive requests."""
        if self.rate_limit_delay <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay and self._last_request_time > 0:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _extract_links_from_html(self, base_url: str, html_text: str) -> Set[str]:
        """Extract all valid internal hyperlinks and PDF references from HTML."""
        links: Set[str] = set()
        soup = BeautifulSoup(html_text, "html.parser")

        # Extract anchor tags
        for a in soup.find_all("a", href=True):
            norm = self.normalize_url(base_url, a["href"])
            if norm and self.is_same_domain(norm):
                links.add(norm)

        # Extract iframe/embed src (often used for linked PDFs)
        for tag in soup.find_all(["iframe", "embed"], src=True):
            norm = self.normalize_url(base_url, tag["src"])
            if norm and self.is_same_domain(norm):
                links.add(norm)

        return links

    def crawl_items(
        self,
        on_page_crawled: Optional[Callable[[str, int, int], None]] = None,
        known_headers: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    ) -> List[CrawlItem]:
        """
        Execute BFS crawling and return list of CrawlItem records containing
        (url, content_type, raw_bytes, etag, last_modified, status_code, is_modified).
        Uses conditional headers (If-None-Match, If-Modified-Since) to skip unchanged resources.
        """
        if known_headers:
            self.known_headers.update(known_headers)

        results: List[CrawlItem] = []
        queue: deque = deque([(self.root_url, 0)])
        
        self.visited_urls.clear()
        self.enqueued_urls.clear()
        self.headers_store.clear()
        self.pages_skipped_304 = 0
        self.enqueued_urls.add(self.root_url)

        pages_crawled_count = 0

        while queue and pages_crawled_count < self.max_pages:
            current_url, depth = queue.popleft()
            self.visited_urls.add(current_url)
            pages_crawled_count += 1

            if on_page_crawled:
                on_page_crawled(current_url, pages_crawled_count, self.max_pages)

            logger.info(f"Crawling [{pages_crawled_count}/{self.max_pages}, depth={depth}]: {current_url}")

            try:
                # Apply rate limiting
                self._apply_rate_limit()

                # Build conditional request headers if ETag or Last-Modified are stored
                req_headers: Dict[str, str] = {}
                stored_meta = self.known_headers.get(current_url)
                if stored_meta:
                    if stored_meta.get("etag"):
                        req_headers["If-None-Match"] = str(stored_meta["etag"])
                    if stored_meta.get("last_modified"):
                        req_headers["If-Modified-Since"] = str(stored_meta["last_modified"])

                # Fetch resource
                resp = self.session.get(
                    current_url,
                    headers=req_headers if req_headers else None,
                    timeout=self.timeout,
                    allow_redirects=True,
                )

                # 304 Not Modified: page unchanged, skip downloading & parsing
                if resp.status_code == 304:
                    self.pages_skipped_304 += 1
                    etag = resp.headers.get("ETag") or resp.headers.get("etag") or (stored_meta.get("etag") if stored_meta else None)
                    last_mod = resp.headers.get("Last-Modified") or resp.headers.get("last-modified") or (stored_meta.get("last_modified") if stored_meta else None)
                    self.headers_store[current_url] = {
                        "etag": etag,
                        "last_modified": last_mod,
                    }
                    item = CrawlItem(
                        url=current_url,
                        content_type="not_modified",
                        raw_bytes=b"",
                        etag=etag,
                        last_modified=last_mod,
                        status_code=304,
                        is_modified=False,
                    )
                    results.append(item)
                    logger.info(f"Conditional request: {current_url} unchanged (HTTP 304 Not Modified)")

                    # If page is 304, ensure known child URLs from previous crawl are enqueued if within max_depth
                    if depth < self.max_depth and self.known_headers:
                        for known_url in self.known_headers:
                            if known_url not in self.visited_urls and known_url not in self.enqueued_urls:
                                if self.should_crawl(known_url):
                                    self.enqueued_urls.add(known_url)
                                    queue.append((known_url, depth + 1))
                    continue

                if resp.status_code != 200:
                    logger.warning(f"Failed to fetch {current_url}: status {resp.status_code}")
                    continue

                # Capture ETag and Last-Modified headers
                etag = resp.headers.get("ETag") or resp.headers.get("etag")
                last_modified = resp.headers.get("Last-Modified") or resp.headers.get("last-modified")
                self.headers_store[current_url] = {
                    "etag": etag,
                    "last_modified": last_modified,
                }

                # Determine content type
                header_ct = resp.headers.get("Content-Type", "").lower()
                is_pdf = "application/pdf" in header_ct or current_url.lower().endswith(".pdf")

                if is_pdf:
                    content_type = "application/pdf"
                elif "text/html" in header_ct or not header_ct:
                    content_type = "text/html"
                else:
                    logger.debug(f"Skipping non-HTML/PDF content-type '{header_ct}' for {current_url}")
                    continue

                raw_bytes = resp.content
                item = CrawlItem(
                    url=current_url,
                    content_type=content_type,
                    raw_bytes=raw_bytes,
                    etag=etag,
                    last_modified=last_modified,
                    status_code=200,
                    is_modified=True,
                )
                results.append(item)

                # Discover more links if HTML and within max_depth
                if depth < self.max_depth and content_type == "text/html":
                    try:
                        discovered = self._extract_links_from_html(current_url, resp.text)
                        for link in discovered:
                            if self.should_crawl(link):
                                self.enqueued_urls.add(link)
                                queue.append((link, depth + 1))
                    except Exception as e:
                        logger.debug(f"Failed to extract links from {current_url}: {e}")

            except requests.RequestException as e:
                logger.warning(f"Request error crawling {current_url}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error crawling {current_url}: {e}", exc_info=True)

        return results

    def crawl(
        self,
        on_page_crawled: Optional[Callable[[str, int, int], None]] = None,
        known_headers: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    ) -> List[Tuple[str, str, bytes]]:
        """
        Execute crawling and output a list of (url, content_type, raw_bytes).
        """
        items = self.crawl_items(on_page_crawled=on_page_crawled, known_headers=known_headers)
        return [(item.url, item.content_type, item.raw_bytes) for item in items]

    def crawl_documents(
        self,
        on_page_crawled: Optional[Callable[[str, int, int], None]] = None,
        known_headers: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    ) -> Generator[ExtractedDocument, None, None]:
        """
        Generator yielding ExtractedDocument objects for direct integration with WebRAG pipeline.
        Skips 304 Not Modified pages automatically.
        """
        items = self.crawl_items(on_page_crawled=on_page_crawled, known_headers=known_headers)
        for item in items:
            if not item.is_modified or item.status_code == 304:
                continue
            try:
                if item.content_type == "application/pdf":
                    doc = ContentExtractor.extract_pdf(item.url, item.raw_bytes)
                else:
                    doc = ContentExtractor.extract_html(item.url, item.raw_bytes)
                if doc and doc.raw_text.strip():
                    yield doc
            except Exception as e:
                logger.error(f"Error extracting document for {item.url}: {e}")
                logger.error(f"Error extracting document for {item.url}: {e}")
