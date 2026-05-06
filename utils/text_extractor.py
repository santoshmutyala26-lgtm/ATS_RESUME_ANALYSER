"""
Text Extractor — PDF parsing utility
Uses pdfminer for robust text extraction with layout analysis
"""

import io
import re


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract clean text from PDF bytes.
    Returns normalized text string.
    """
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams

        output = io.StringIO()
        input_fp = io.BytesIO(pdf_bytes)

        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
        )

        extract_text_to_fp(input_fp, output, laparams=laparams)
        text = output.getvalue()

    except ImportError:
        # Fallback: try pypdf2
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            raise RuntimeError("No PDF parsing library available. Install pdfminer.six or pypdf.")

    return _clean_text(text)


def _clean_text(text: str) -> str:
    """Normalize extracted text: remove artifacts, fix spacing."""
    if not text:
        return ""

    # Remove null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Fix hyphenation artifacts (word- \nbreak → wordbreak)
    text = re.sub(r"-\s*\n\s*", "", text)

    # Normalize multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize multiple newlines (keep max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix common PDF encoding artifacts
    text = text.replace("\uf0b7", "•")  # bullet
    text = text.replace("\u2019", "'")  # right single quote
    text = text.replace("\u2018", "'")  # left single quote
    text = text.replace("\u201c", '"')  # left double quote
    text = text.replace("\u201d", '"')  # right double quote

    return text.strip()
