import fitz  # PyMuPDF
from app.crawler.crawler import WebCrawler
from app.crawler.extractor import ContentExtractor, ExtractedSection


def test_url_normalization():
    base = "https://example.com/blog/article-1"
    
    # Relative path
    assert ContentExtractor.normalize_url(base, "/about") == "https://example.com/about"
    assert ContentExtractor.normalize_url(base, "page-2") == "https://example.com/blog/page-2"
    
    # URL with fragment (must strip fragment)
    assert ContentExtractor.normalize_url(base, "/pricing#features") == "https://example.com/pricing"
    
    # Non-http protocols should be ignored
    assert ContentExtractor.normalize_url(base, "mailto:hello@example.com") is None
    assert ContentExtractor.normalize_url(base, "javascript:void(0)") is None
    assert ContentExtractor.normalize_url(base, "#top") is None


def test_html_extraction():
    html_sample = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>WebRAG Docs</title>
        <script>console.log("ignore me");</script>
        <style>body { color: red; }</style>
      </head>
      <body>
        <nav><a href="/home">Home</a></nav>
        <main>
          <h1>Overview of WebRAG</h1>
          <p>WebRAG is a self-updating local AI assistant that crawls websites and generates an isolated vector index.</p>
          <h2>Incremental Updates</h2>
          <p>It hashes chunks with SHA-256 to ensure that unchanged content is never re-embedded.</p>
          <a href="/features">Features Page</a>
          <a href="https://external.com/other">External Link</a>
        </main>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    url = "https://example.com/docs"
    doc = ContentExtractor.extract_html(url, html_sample)

    assert doc.url == url
    assert doc.title == "WebRAG Docs"
    assert doc.content_type == "html"
    assert "ignore me" not in doc.raw_text
    assert "Copyright 2026" not in doc.raw_text
    assert "Overview of WebRAG" in doc.raw_text
    assert "SHA-256" in doc.raw_text

    # Verify discovered links
    assert "https://example.com/features" in doc.links
    assert "https://external.com/other" in doc.links


def test_pdf_extraction():
    # Create an in-memory test PDF with PyMuPDF
    pdf_doc = fitz.open()
    
    page1 = pdf_doc.new_page()
    page1.insert_text((50, 72), "First Page Header\nWebRAG uses PyMuPDF for precise local PDF parsing.")
    
    page2 = pdf_doc.new_page()
    page2.insert_text((50, 72), "Second Page Content\nEvery page preserves its page number for accurate citations.")
    
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    url = "https://example.com/whitepaper.pdf"
    extracted = ContentExtractor.extract_pdf(url, pdf_bytes)

    assert extracted.url == url
    assert extracted.content_type == "pdf"
    assert len(extracted.sections) == 2
    assert extracted.sections[0].page_number == 1
    assert "First Page Header" in extracted.sections[0].text
    assert extracted.sections[1].page_number == 2
    assert "accurate citations" in extracted.sections[1].text


def test_crawler_domain_scoping():
    crawler = WebCrawler(root_url="https://docs.mycompany.org/start", max_depth=2, max_pages=10)
    
    # Internal URLs allowed
    assert crawler.should_crawl("https://docs.mycompany.org/guide") is True
    assert crawler.should_crawl("https://docs.mycompany.org/api/v1") is True
    
    # External URLs blocked
    assert crawler.should_crawl("https://google.com/search") is False
    assert crawler.should_crawl("https://otherdomain.org/docs") is False
    
    # Binary assets blocked
    assert crawler.should_crawl("https://docs.mycompany.org/logo.png") is False
    assert crawler.should_crawl("https://docs.mycompany.org/style.css") is False
    assert crawler.should_crawl("https://docs.mycompany.org/app.js") is False
