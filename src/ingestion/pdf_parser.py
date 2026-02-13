"""PDF parsing with Docling + OCR fallback."""

import logging
from pathlib import Path

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse PDFs using Docling with OCR fallback."""

    def __init__(self) -> None:
        self.converter = DocumentConverter()

    def parse(self, pdf_path: str | Path) -> dict:
        """Parse a PDF file and return structured content.

        Returns a dict with keys: text, tables, metadata.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            result = self.converter.convert(str(pdf_path))
            doc = result.document

            return {
                "text": doc.export_to_markdown(),
                "tables": self._extract_tables(doc),
                "metadata": {
                    "pages": len(doc.pages) if hasattr(doc, "pages") else None,
                    "method": "docling",
                },
            }
        except Exception:
            logger.warning(
                "Docling failed for %s. Trying OCR fallback...",
                pdf_path,
                exc_info=True,
            )
            return self._ocr_fallback(pdf_path)

    def _extract_tables(self, doc: object) -> list[dict]:
        """Extract tables from Docling document."""
        tables: list[dict] = []
        try:
            for table in doc.tables:  # type: ignore[attr-defined]
                tables.append(
                    {
                        "content": table.export_to_markdown(),
                        "caption": getattr(table, "caption", None),
                    }
                )
        except Exception:
            pass
        return tables

    def _ocr_fallback(self, pdf_path: Path) -> dict:
        """Fallback OCR mechanism using EasyOCR + PyMuPDF."""
        try:
            import easyocr  # noqa: PLC0415
            import fitz  # PyMuPDF  # noqa: PLC0415

            reader = easyocr.Reader(["en"])
            doc = fitz.open(str(pdf_path))
            full_text: list[str] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")

                results = reader.readtext(img_bytes, detail=0)
                full_text.append("\n".join(results))

            return {
                "text": "\n\n".join(full_text),
                "tables": [],
                "metadata": {
                    "pages": len(doc),
                    "method": "ocr_fallback",
                },
            }
        except Exception:
            logger.exception("OCR fallback also failed for %s", pdf_path)
            return {
                "text": "",
                "tables": [],
                "metadata": {"method": "failed", "error": str(pdf_path)},
            }
