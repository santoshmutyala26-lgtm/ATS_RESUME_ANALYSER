import io
import re
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        output = io.StringIO()
        input_fp = io.BytesIO(pdf_bytes)
        laparams = LAParams(line_margin=0.5, word_margin=0.1, char_margin=2.0, boxes_flow=0.5)
        extract_text_to_fp(input_fp, output, laparams=laparams)
        text = output.getvalue()
    except ImportError:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            text = '\n'.join((page.extract_text() or '' for page in reader.pages))
        except Exception:
            raise RuntimeError('No PDF parsing library available. Install pdfminer.six or pypdf.')
    return _clean_text(text)
def _clean_text(text: str) -> str:
    if not text:
        return ''
    text = re.sub('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]', '', text)
    text = re.sub('-\\s*\\n\\s*', '', text)
    text = re.sub('[ \\t]+', ' ', text)
    text = re.sub('\\n{3,}', '\n\n', text)
    text = text.replace('\uf0b7', '•')
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    return text.strip()
