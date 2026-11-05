import logging
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class ExtractedSection:
    """Represents a discrete section within an extracted document."""

    def __init__(self, text: str, heading: Optional[str] = None, page_number: Optional[int] = None):
        self.text = text.strip()
        self.heading = heading
        self.page_number = page_number

    def __repr__(self) -> str:
        return f"<ExtractedSection heading={self.heading!r} page={self.page_number} len={len(self.text)}>"


class StructuredDocument(dict):
    """
    Structured document representation returned by extractors:
    {url, title, section_headings, text, page_no}
    
    Subclasses dict so it can be used directly as a dictionary or as an object.
    """

    def __init__(
        self,
        url: str,
        title: str,
        section_headings: Optional[List[str]] = None,
        text: Optional[str] = None,
        page_no: Optional[int] = None,
        content_type: str = "html",
        raw_text: Optional[str] = None,
        sections: Optional[List[ExtractedSection]] = None,
        links: Optional[Set[str]] = None,
        **kwargs,
    ):
        body_text = text if text is not None else (raw_text or "")
        headings = section_headings if section_headings is not None else []
        data = {
            "url": url,
            "title": title,
            "section_headings": headings,
            "text": body_text,
            "page_no": page_no,
        }
        super().__init__(data)
        self.__dict__.update(data)
        
        # Pipeline compatibility attributes
        self.raw_text = body_text
        self.content_type = content_type
        self.sections = sections or []
        self.links = links or set()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self["url"],
            "title": self["title"],
            "section_headings": self["section_headings"],
            "text": self["text"],
            "page_no": self["page_no"],
        }

    def __repr__(self) -> str:
        return f"<StructuredDocument url={self['url']!r} title={self['title']!r} page_no={self['page_no']} text_len={len(self['text'])}>"


class ExtractedDocument(StructuredDocument):
    """Backward compatibility subclass for ExtractedDocument positional & keyword arguments."""

    def __init__(
        self,
        url: str,
        title: str,
        content_type: str = "html",
        raw_text: str = "",
        sections: Optional[List[ExtractedSection]] = None,
        links: Optional[Set[str]] = None,
        **kwargs,
    ):
        super().__init__(
            url=url,
            title=title,
            section_headings=kwargs.get("section_headings", []),
            text=kwargs.get("text") or raw_text,
            page_no=kwargs.get("page_no"),
            content_type=content_type,
            raw_text=raw_text,
            sections=sections,
            links=links,
            **kwargs,
        )


class HTMLExtractor:
    """
    Extracts structured content from HTML using BeautifulSoup:
    - Strips nav, footer, scripts, styles, forms, buttons, and boilerplate.
    - Preserves headings, bullet/ordered lists, and formatted tables as readable text.
    - Safely decodes text handling encoding errors gracefully.
    - Returns structured document: {url, title, section_headings, text, page_no}.
    """

    UNWANTED_TAGS = [
        "script", "style", "nav", "footer", "header", "noscript",
        "aside", "iframe", "svg", "canvas", "form", "button", "input",
        "select", "textarea", "meta", "link", "dialog", "template",
    ]

    @classmethod
    def _decode_content(cls, html_content: str | bytes) -> str:
        """Decode raw HTML bytes handling encoding errors gracefully."""
        if isinstance(html_content, str):
            return html_content
        if not isinstance(html_content, (bytes, bytearray)):
            return str(html_content)

        # Attempt UTF-8 first
        try:
            return html_content.decode("utf-8")
        except UnicodeDecodeError:
            pass

        # Fallback to Latin-1 / Windows-1252
        try:
            return html_content.decode("latin-1")
        except Exception:
            pass

        # Final safe fallback replacing invalid bytes
        return html_content.decode("utf-8", errors="replace")

    @classmethod
    def _format_tables(cls, soup: BeautifulSoup) -> None:
        """Convert <table> elements into readable markdown-style text tables."""
        for table in soup.find_all("table"):
            rows_text: List[str] = []
            has_th = False
            col_count = 0

            for tr in table.find_all("tr"):
                th_cells = tr.find_all("th")
                td_cells = tr.find_all("td")
                cells = th_cells if th_cells else td_cells
                if not cells:
                    continue

                if th_cells:
                    has_th = True

                row_vals = [re.sub(r"\s+", " ", cell.get_text(strip=True)).replace("|", "/") for cell in cells]
                col_count = max(col_count, len(row_vals))
                rows_text.append("| " + " | ".join(row_vals) + " |")

                # Insert markdown separator after header row
                if th_cells and len(rows_text) == 1:
                    sep = "| " + " | ".join(["---"] * len(row_vals)) + " |"
                    rows_text.append(sep)

            if rows_text:
                # If no <th> was present but table has multiple rows, format cleanly
                table_md = "\n".join(rows_text)
                table.replace_with(soup.new_string(f"\n\n{table_md}\n\n"))
            else:
                table.decompose()

    @classmethod
    def _format_lists(cls, soup: BeautifulSoup) -> None:
        """Convert <ul> and <ol> elements into readable bulleted or numbered lines."""
        # Unordered lists
        for ul in soup.find_all("ul"):
            items: List[str] = []
            for li in ul.find_all("li", recursive=False):
                txt = re.sub(r"\s+", " ", li.get_text(strip=True))
                if txt:
                    items.append(f"- {txt}")
            if items:
                ul_md = "\n".join(items)
                ul.replace_with(soup.new_string(f"\n\n{ul_md}\n\n"))
            else:
                ul.decompose()

        # Ordered lists
        for ol in soup.find_all("ol"):
            items = []
            for idx, li in enumerate(ol.find_all("li", recursive=False), start=1):
                txt = re.sub(r"\s+", " ", li.get_text(strip=True))
                if txt:
                    items.append(f"{idx}. {txt}")
            if items:
                ol_md = "\n".join(items)
                ol.replace_with(soup.new_string(f"\n\n{ol_md}\n\n"))
            else:
                ol.decompose()

    @classmethod
    def extract(cls, url: str, html_content: str | bytes) -> StructuredDocument:
        """Extract clean text and structure from HTML."""
        decoded_html = cls._decode_content(html_content).strip()
        if not decoded_html:
            # Handle empty page gracefully
            return StructuredDocument(
                url=url,
                title=cls._fallback_title(url),
                section_headings=[],
                text="",
                page_no=None,
                content_type="text/html",
                sections=[],
                links=set(),
            )

        soup = BeautifulSoup(decoded_html, "html.parser")

        # 1. Discover outgoing links before tag decomposition
        discovered_links: Set[str] = set()
        for a in soup.find_all("a", href=True):
            norm = ContentExtractor.normalize_url(url, a["href"])
            if norm:
                discovered_links.add(norm)

        # 2. Extract Document Title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.h1:
            title = soup.h1.get_text(strip=True)
        else:
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()
            else:
                title = cls._fallback_title(url)

        # 3. Discover section headings before replacing elements
        section_headings: List[str] = []
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            h_text = h.get_text(strip=True)
            if h_text and h_text not in section_headings:
                section_headings.append(h_text)

        # 4. Strip boilerplate and clutter elements
        for tag in soup.find_all(cls.UNWANTED_TAGS):
            tag.decompose()

        # 5. Format tables, lists, and headings into clean readable text
        cls._format_tables(soup)
        cls._format_lists(soup)

        for level in range(1, 7):
            for h in soup.find_all(f"h{level}"):
                h_text = h.get_text(strip=True)
                if h_text:
                    h.replace_with(soup.new_string(f"\n\n{'#' * level} {h_text}\n\n"))
                else:
                    h.decompose()

        # 6. Extract clean formatted text from body/main
        target_root = soup.find("main") or soup.find("article") or soup.find("body") or soup
        raw_text = target_root.get_text(separator="\n")
        
        # Clean lines and normalize excessive blank lines
        clean_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
        full_text = re.sub(r"\n{3,}", "\n\n", "\n".join(clean_lines)).strip()

        # 7. Construct ExtractedSections by splitting on markdown headings
        sections: List[ExtractedSection] = []
        current_heading = title
        current_lines: List[str] = []

        for line in full_text.splitlines():
            line_strip = line.strip()
            heading_match = re.match(r"^#{1,6}\s+(.+)$", line_strip)
            if heading_match:
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    if sec_text:
                        sections.append(ExtractedSection(text=sec_text, heading=current_heading))
                    current_lines = []
                current_heading = heading_match.group(1).strip()
            elif line_strip:
                current_lines.append(line_strip)

        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            if sec_text:
                sections.append(ExtractedSection(text=sec_text, heading=current_heading))

        if not sections and full_text:
            sections.append(ExtractedSection(text=full_text, heading=title))

        return StructuredDocument(
            url=url,
            title=title,
            section_headings=section_headings,
            text=full_text,
            page_no=None,
            content_type="html",
            sections=sections,
            links=discovered_links,
        )

    @staticmethod
    def _fallback_title(url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.strip("/").split("/")[-1]
        return path or parsed.netloc or "Web Page"


class PDFExtractor:
    """
    Extracts structured content from PDF documents using PyMuPDF (fitz):
    - Extracts per-page text and assigns 1-indexed page numbers (page_no).
    - Handles empty pages gracefully (skipping or retaining them without crashing).
    - Handles corrupted/empty PDF byte streams gracefully.
    - Returns structured documents: {url, title, section_headings, text, page_no}.
    """

    @classmethod
    def extract_pages(
        cls,
        url: str,
        pdf_bytes: bytes,
        skip_empty_pages: bool = False,
    ) -> List[StructuredDocument]:
        """
        Extract each PDF page as an individual StructuredDocument with page_no.
        Handles empty or corrupt bytes without raising exceptions.
        """
        if not pdf_bytes:
            return []

        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            logger.warning(f"Could not open PDF from {url}: {e}")
            return []

        meta = doc.metadata or {}
        title = meta.get("title") or url.split("/")[-1] or "PDF Document"

        page_documents: List[StructuredDocument] = []

        try:
            for page_idx in range(len(doc)):
                page_no = page_idx + 1
                page = doc[page_idx]
                raw_page_text = page.get_text("text").strip()

                # Clean whitespace
                clean_text = re.sub(r"[ \t]+", " ", raw_page_text)
                clean_text = re.sub(r"\n\s*\n", "\n\n", clean_text).strip()

                if not clean_text:
                    if skip_empty_pages:
                        continue
                    # Retain empty page document with text=""
                    page_documents.append(
                        StructuredDocument(
                            url=url,
                            title=title,
                            section_headings=[],
                            text="",
                            page_no=page_no,
                            content_type="pdf",
                            sections=[],
                            links=set(),
                        )
                    )
                    continue

                # Detect headings on this page
                lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
                headings: List[str] = []
                # First non-trivial short line is often the page heading/title
                if lines and len(lines[0]) < 80:
                    headings.append(lines[0])

                section = ExtractedSection(text=clean_text, heading=f"Page {page_no}", page_number=page_no)

                page_documents.append(
                    StructuredDocument(
                        url=url,
                        title=title,
                        section_headings=headings,
                        text=clean_text,
                        page_no=page_no,
                        content_type="pdf",
                        sections=[section],
                        links=set(),
                    )
                )
        finally:
            if doc:
                doc.close()

        return page_documents

    @classmethod
    def extract(
        cls,
        url: str,
        pdf_bytes: bytes,
        skip_empty_pages: bool = True,
    ) -> StructuredDocument:
        """
        Extract complete PDF as a unified StructuredDocument containing all page sections.
        """
        pages = cls.extract_pages(url, pdf_bytes, skip_empty_pages=skip_empty_pages)
        if not pages:
            fallback_title = url.split("/")[-1] or "PDF Document"
            return StructuredDocument(
                url=url,
                title=fallback_title,
                section_headings=[],
                text="",
                page_no=None,
                content_type="pdf",
                sections=[],
                links=set(),
            )

        title = pages[0].title
        all_headings: List[str] = []
        all_sections: List[ExtractedSection] = []
        raw_parts: List[str] = []

        for p in pages:
            if p["section_headings"]:
                all_headings.extend(p["section_headings"])
            if p.sections:
                all_sections.extend(p.sections)
            if p["text"]:
                raw_parts.append(f"--- Page {p['page_no']} ---\n{p['text']}")

        combined_text = "\n\n".join(raw_parts).strip()

        return StructuredDocument(
            url=url,
            title=title,
            section_headings=all_headings,
            text=combined_text,
            page_no=None,
            content_type="pdf",
            sections=all_sections,
            links=set(),
        )


class ContentExtractor:
    """Unified entry point for HTML and PDF extraction."""

    @staticmethod
    def normalize_url(base_url: str, link: str) -> Optional[str]:
        """Convert relative link to absolute, strip fragments and trailing whitespace."""
        if not link:
            return None
        link = link.strip()
        if link.startswith(("mailto:", "tel:", "javascript:", "#", "data:")):
            return None

        try:
            absolute = urljoin(base_url, link)
            clean, _ = urldefrag(absolute)
            parsed = urlparse(clean)
            if parsed.scheme not in ("http", "https"):
                return None

            path = parsed.path
            if path == "":
                path = "/"

            normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
            if parsed.query:
                normalized += f"?{parsed.query}"
            return normalized
        except Exception as e:
            logger.debug(f"Failed to normalize url {link}: {e}")
            return None

    @classmethod
    def extract_html(cls, url: str, html_content: str | bytes) -> StructuredDocument:
        """Parse HTML, removing boilerplate and preserving headings, lists, tables."""
        return HTMLExtractor.extract(url, html_content)

    @classmethod
    def extract_pdf(cls, url: str, pdf_bytes: bytes, skip_empty_pages: bool = True) -> StructuredDocument:
        """Parse PDF document with PyMuPDF, preserving page numbers and text sections."""
        return PDFExtractor.extract(url, pdf_bytes, skip_empty_pages=skip_empty_pages)

    @classmethod
    def extract_pdf_pages(cls, url: str, pdf_bytes: bytes, skip_empty_pages: bool = False) -> List[StructuredDocument]:
        """Parse PDF document returning a list of per-page StructuredDocument objects."""
        return PDFExtractor.extract_pages(url, pdf_bytes, skip_empty_pages=skip_empty_pages)
