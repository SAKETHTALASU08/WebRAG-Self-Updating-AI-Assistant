import hashlib
import re
from typing import List, Optional
from app.config import settings
from app.crawler.extractor import ExtractedDocument, ExtractedSection, StructuredDocument


class DocumentChunk:
    """
    Represents a heading-aware text chunk with content-based SHA-256 chunk_id.
    
    Content-based ID ensures that inserting a paragraph does not invalidate
    unrelated chunks.
    """

    def __init__(
        self,
        url: str,
        text: str,
        heading: Optional[str] = None,
        page_no: Optional[int] = None,
        title: Optional[str] = None,
        chunk_index: int = 0,
        chunk_id: Optional[str] = None,
        chunk_hash: Optional[str] = None,
        page_number: Optional[int] = None,
    ):
        self.url = url
        self.title = title or ""
        self.text = text.strip()
        self.heading = heading
        self.page_no = page_no if page_no is not None else page_number
        self.page_number = self.page_no
        self.chunk_index = chunk_index

        # Content-based hash and ID: SHA-256 of normalized chunk text
        computed = self.compute_hash(self.text)
        self.chunk_hash = chunk_hash or computed
        self.hash = self.chunk_hash
        self.chunk_id = chunk_id or self.chunk_hash
        self.id = self.chunk_id

    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA-256 hex digest of normalized chunk text."""
        normalized = " ".join(text.strip().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_chunk_id(text: str) -> str:
        """Compute content-based chunk ID from normalized text."""
        return DocumentChunk.compute_hash(text)

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.chunk_id[:8]} heading={self.heading!r} page={self.page_no}>"


class HeadingAwareChunker:
    """
    Heading-aware document chunker with content-based SHA-256 identification.
    
    - Target: ~500 tokens (words) per chunk
    - Overlap: ~50 tokens (words)
    - Heading-aware: respects semantic sections under headings and preserves context.
    - Content-based ID: SHA-256 of normalized text, preventing cascading invalidations.
    """

    def __init__(
        self,
        target_tokens: int = 500,
        overlap_tokens: int = 50,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.target_tokens = chunk_size if chunk_size is not None else target_tokens
        self.overlap_tokens = chunk_overlap if chunk_overlap is not None else overlap_tokens
        # Aliases for backward compatibility
        self.chunk_size = self.target_tokens
        self.chunk_overlap = self.overlap_tokens

    def chunk_document(self, doc: ExtractedDocument | StructuredDocument) -> List[DocumentChunk]:
        """Convert a document into heading-aware DocumentChunks."""
        chunks: List[DocumentChunk] = []
        global_idx = 0

        # Retrieve or build structured sections
        sections = self._resolve_sections(doc)

        for section in sections:
            sec_chunks = self._chunk_section(doc, section, global_idx)
            chunks.extend(sec_chunks)
            global_idx += len(sec_chunks)

        return chunks

    def _resolve_sections(self, doc: ExtractedDocument | StructuredDocument) -> List[ExtractedSection]:
        """Extract or parse heading sections from document."""
        if hasattr(doc, "sections") and doc.sections:
            return doc.sections

        raw_text = getattr(doc, "raw_text", None) or doc.get("text", "")
        if not raw_text.strip():
            return []

        # Parse markdown-style headings (# Heading)
        sections: List[ExtractedSection] = []
        current_heading = getattr(doc, "title", None) or doc.get("title", "")
        current_lines: List[str] = []
        page_no = getattr(doc, "page_no", None) or doc.get("page_no")

        for line in raw_text.splitlines():
            line_str = line.strip()
            heading_match = re.match(r"^#{1,6}\s+(.+)$", line_str)
            if heading_match:
                if current_lines:
                    text_block = "\n".join(current_lines).strip()
                    if text_block:
                        sections.append(ExtractedSection(text=text_block, heading=current_heading, page_number=page_no))
                    current_lines = []
                current_heading = heading_match.group(1).strip()
            elif line_str:
                current_lines.append(line_str)

        if current_lines:
            text_block = "\n".join(current_lines).strip()
            if text_block:
                sections.append(ExtractedSection(text=text_block, heading=current_heading, page_number=page_no))

        if not sections and raw_text.strip():
            sections.append(ExtractedSection(text=raw_text.strip(), heading=current_heading, page_number=page_no))

        return sections

    def _chunk_section(
        self,
        doc: ExtractedDocument | StructuredDocument,
        section: ExtractedSection,
        start_idx: int,
    ) -> List[DocumentChunk]:
        """Split a heading section into chunks respecting word boundaries and overlap."""
        text = section.text.strip()
        if not text:
            return []

        words = text.split()
        chunks: List[DocumentChunk] = []

        url = getattr(doc, "url", None) or doc.get("url", "")
        title = getattr(doc, "title", None) or doc.get("title", "")
        page_no = section.page_number if section.page_number is not None else getattr(doc, "page_no", None) or doc.get("page_no")

        # If section fits within target token size, make a single chunk
        if len(words) <= self.target_tokens:
            chunk = DocumentChunk(
                url=url,
                title=title,
                chunk_index=start_idx,
                text=text,
                page_no=page_no,
                heading=section.heading,
            )
            return [chunk]

        # Sliding window with overlap
        step = max(1, self.target_tokens - self.overlap_tokens)
        current_idx = start_idx

        for i in range(0, len(words), step):
            window_words = words[i : i + self.target_tokens]
            if not window_words:
                break

            chunk_text = " ".join(window_words)
            chunk = DocumentChunk(
                url=url,
                title=title,
                chunk_index=current_idx,
                text=chunk_text,
                page_no=page_no,
                heading=section.heading,
            )
            chunks.append(chunk)
            current_idx += 1

            if i + self.target_tokens >= len(words):
                break

        return chunks


# Alias for backward compatibility
IncrementalChunker = HeadingAwareChunker
