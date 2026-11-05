import fitz  # PyMuPDF
import pytest
from app.crawler.extractor import ContentExtractor, HTMLExtractor, PDFExtractor, StructuredDocument


def test_html_extractor_strips_boilerplate_and_keeps_structure():
    """Verify HTML extractor strips nav, footer, scripts and keeps headings, lists, tables."""
    html_sample = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <title>WebRAG Architecture Guide</title>
        <script>function track() { window.location = '/track'; }</script>
        <style>body { font-family: sans-serif; } .ad { display: block; }</style>
      </head>
      <body>
        <header>
          <div class="logo">Company Logo</div>
        </header>
        <nav>
          <a href="/home">Home</a>
          <a href="/docs">Docs</a>
        </nav>
        <main>
          <h1>System Architecture</h1>
          <p>WebRAG uses an isolated SQLite and FAISS storage model per site.</p>

          <h2>Core Capabilities</h2>
          <ul>
            <li>Incremental SHA-256 chunk updates</li>
            <li>Zero external API dependencies</li>
            <li>Per-page PDF citation resolution</li>
          </ul>

          <h3>Execution Steps</h3>
          <ol>
            <li>Crawl root URL</li>
            <li>Extract text and headings</li>
            <li>Index vector embeddings</li>
          </ol>

          <h2>Component Matrix</h2>
          <table>
            <thead>
              <tr>
                <th>Component</th>
                <th>Technology</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Crawler</td>
                <td>Requests + BeautifulSoup</td>
                <td>Discovery</td>
              </tr>
              <tr>
                <td>Parser</td>
                <td>PyMuPDF</td>
                <td>PDF Extraction</td>
              </tr>
            </tbody>
          </table>
        </main>
        <aside>
          <div class="ad-banner">Sponsored Ad</div>
        </aside>
        <footer>
          <p>Copyright 2026 Acme Corp. All rights reserved.</p>
        </footer>
      </body>
    </html>
    """

    url = "https://example.com/docs/architecture"
    doc: StructuredDocument = HTMLExtractor.extract(url, html_sample)

    # 1. Verify structured document keys and values
    assert isinstance(doc, dict)
    assert doc["url"] == url
    assert doc["title"] == "WebRAG Architecture Guide"
    assert doc["page_no"] is None

    # Attribute access
    assert doc.url == url
    assert doc.title == "WebRAG Architecture Guide"
    assert doc.page_no is None

    # 2. Verify section_headings collected
    expected_headings = ["System Architecture", "Core Capabilities", "Execution Steps", "Component Matrix"]
    for heading in expected_headings:
        assert heading in doc["section_headings"]

    # 3. Verify stripped boilerplate
    assert "window.location" not in doc["text"]
    assert "font-family" not in doc["text"]
    assert "Company Logo" not in doc["text"]
    assert "Copyright 2026 Acme Corp" not in doc["text"]
    assert "Sponsored Ad" not in doc["text"]

    # 4. Verify preserved lists (bulleted and numbered)
    assert "- Incremental SHA-256 chunk updates" in doc["text"]
    assert "- Zero external API dependencies" in doc["text"]
    assert "1. Crawl root URL" in doc["text"]
    assert "2. Extract text and headings" in doc["text"]

    # 5. Verify preserved table as readable text
    assert "| Component | Technology | Role |" in doc["text"]
    assert "| Crawler | Requests + BeautifulSoup | Discovery |" in doc["text"]
    assert "| Parser | PyMuPDF | PDF Extraction |" in doc["text"]


def test_html_extractor_encoding_error_handling():
    """Verify HTML extractor handles broken UTF-8, Latin-1, and invalid byte sequences gracefully."""
    url = "https://example.com/encoding-test"

    # Latin-1 / Windows-1252 encoded content containing accented characters
    latin1_bytes = "<h1>Café & Naïve</h1><p>Resumé details with £50 price.</p>".encode("latin-1")
    doc_latin1 = HTMLExtractor.extract(url, latin1_bytes)
    assert "Café" in doc_latin1["text"] or "Caf" in doc_latin1["text"]
    assert doc_latin1["title"] == "Café & Naïve"

    # Corrupt / invalid byte sequence (e.g. invalid continuation bytes in UTF-8)
    corrupted_bytes = b"<h1>Title</h1><p>Valid text with broken \xff\xfe bytes here.</p>"
    doc_corrupt = HTMLExtractor.extract(url, corrupted_bytes)
    assert doc_corrupt["title"] == "Title"
    assert "Valid text" in doc_corrupt["text"]
    assert isinstance(doc_corrupt["section_headings"], list)


def test_html_extractor_empty_pages():
    """Verify HTML extractor handles empty or whitespace-only inputs gracefully."""
    url = "https://example.com/empty"

    # Empty string
    doc_empty_str = HTMLExtractor.extract(url, "")
    assert doc_empty_str["url"] == url
    assert doc_empty_str["text"] == ""
    assert doc_empty_str["section_headings"] == []

    # Whitespace only
    doc_ws = HTMLExtractor.extract(url, "   \n\t   ")
    assert doc_ws["text"] == ""

    # Page containing only stripped boilerplate (no main content)
    doc_boiler_only = HTMLExtractor.extract(url, "<html><body><script>var x = 1;</script><footer>Bye</footer></body></html>")
    assert doc_boiler_only["text"] == ""


def test_pdf_extractor_per_page_text_and_page_numbers():
    """Verify PDF extractor returns structured documents per page with 1-indexed page numbers."""
    # Create an in-memory 3-page PDF
    pdf_doc = fitz.open()

    p1 = pdf_doc.new_page()
    p1.insert_text((50, 72), "Chapter 1: Introduction\nWebRAG extracts PDF documents per page.")

    p2 = pdf_doc.new_page()
    p2.insert_text((50, 72), "Chapter 2: Methods\nPyMuPDF provides fast text extraction.")

    p3 = pdf_doc.new_page()
    p3.insert_text((50, 72), "Chapter 3: Results\nPage numbering enables exact citation grounding.")

    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    url = "https://example.com/manual.pdf"
    pages = PDFExtractor.extract_pages(url, pdf_bytes)

    assert len(pages) == 3

    # Check each page document structure
    for idx, page in enumerate(pages, start=1):
        assert isinstance(page, dict)
        assert page["url"] == url
        assert page["page_no"] == idx
        assert f"Chapter {idx}" in page["text"]
        assert len(page["section_headings"]) >= 1

    assert pages[0]["page_no"] == 1
    assert "Chapter 1: Introduction" in pages[0]["text"]

    assert pages[1]["page_no"] == 2
    assert "Chapter 2: Methods" in pages[1]["text"]

    assert pages[2]["page_no"] == 3
    assert "Chapter 3: Results" in pages[2]["text"]


def test_pdf_extractor_empty_pages_handling():
    """Verify PDF extractor handles empty pages within a document gracefully."""
    pdf_doc = fitz.open()

    # Page 1: Has content
    p1 = pdf_doc.new_page()
    p1.insert_text((50, 72), "First Page Content")

    # Page 2: Completely Blank / Empty Page
    pdf_doc.new_page()

    # Page 3: Has content
    p3 = pdf_doc.new_page()
    p3.insert_text((50, 72), "Third Page Content")

    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    url = "https://example.com/doc-with-empty-page.pdf"

    # 1. With skip_empty_pages=False: blank page retained with text=""
    all_pages = PDFExtractor.extract_pages(url, pdf_bytes, skip_empty_pages=False)
    assert len(all_pages) == 3
    assert all_pages[0]["page_no"] == 1
    assert "First Page Content" in all_pages[0]["text"]

    assert all_pages[1]["page_no"] == 2
    assert all_pages[1]["text"] == ""

    assert all_pages[2]["page_no"] == 3
    assert "Third Page Content" in all_pages[2]["text"]

    # 2. With skip_empty_pages=True: blank page skipped
    filtered_pages = PDFExtractor.extract_pages(url, pdf_bytes, skip_empty_pages=True)
    assert len(filtered_pages) == 2
    page_numbers = [p["page_no"] for p in filtered_pages]
    assert page_numbers == [1, 3]


def test_pdf_extractor_corrupt_or_zero_bytes():
    """Verify PDF extractor handles corrupt bytes or zero-byte streams without crashing."""
    url = "https://example.com/corrupt.pdf"

    # Empty bytes
    assert PDFExtractor.extract_pages(url, b"") == []
    empty_doc = PDFExtractor.extract(url, b"")
    assert empty_doc["text"] == ""
    assert empty_doc["page_no"] is None

    # Non-PDF garbage bytes
    garbage = b"NOT_A_REAL_PDF_STREAM_1234567890"
    pages_garbage = PDFExtractor.extract_pages(url, garbage)
    assert pages_garbage == []
    doc_garbage = PDFExtractor.extract(url, garbage)
    assert doc_garbage["text"] == ""


def test_content_extractor_unified_interface():
    """Verify ContentExtractor provides unified access returning StructuredDocument objects."""
    url_html = "https://example.com/page"
    html = "<html><body><h1>Hello World</h1><p>Test paragraph</p></body></html>"
    doc_html = ContentExtractor.extract_html(url_html, html)

    assert isinstance(doc_html, StructuredDocument)
    assert doc_html["url"] == url_html
    assert doc_html["title"] == "Hello World"
    assert "Test paragraph" in doc_html["text"]
    assert doc_html["page_no"] is None

    # PDF unified
    pdf_doc = fitz.open()
    p1 = pdf_doc.new_page()
    p1.insert_text((50, 72), "Document Title\nSome content here")
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    url_pdf = "https://example.com/file.pdf"
    doc_pdf = ContentExtractor.extract_pdf(url_pdf, pdf_bytes)
    assert isinstance(doc_pdf, StructuredDocument)
    assert "Some content here" in doc_pdf["text"]
    assert doc_pdf.content_type == "pdf"
