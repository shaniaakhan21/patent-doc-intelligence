"""
ocr.py — Stage 1 of the patent document intelligence pipeline.

Routes each PDF to the right extraction path:
  - Native (born-digital) PDFs  -> direct text + block extraction via PyMuPDF
  - Scanned (image-based) PDFs  -> page rendering + EasyOCR (language-routed)

Usage:
    python src/ocr.py
    python src/ocr.py --file data/raw/native_en_WO2021087334.pdf --max-pages 3
"""

import argparse
import json
import re
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

FNAME_RE = re.compile(r"^(native|scanned)_([a-z]{2})_(.+)\.pdf$", re.IGNORECASE)

# EasyOCR takes a list of languages per reader
EASY_LANG = {
    "en": ["en"],
    "zh": ["ch_sim", "en"],   # Chinese + English (patents have Latin numbers)
    "de": ["de", "en"],
    "fr": ["fr", "en"],
    "ja": ["ja", "en"],
}

_OCR_ENGINES = {}


def get_ocr_engine(lang_code: str):
    """Create (once) and return an EasyOCR Reader for the given language."""
    import easyocr

    key = lang_code if lang_code in EASY_LANG else "en"
    if key not in _OCR_ENGINES:
        _OCR_ENGINES[key] = easyocr.Reader(EASY_LANG[key], gpu=False)
    return _OCR_ENGINES[key]


# ---------------------------------------------------------------------------
# Step 1 — PDF type detection
# ---------------------------------------------------------------------------
def detect_pdf_type(doc: fitz.Document, sample_pages: int = 5,
                    min_chars_per_page: int = 20) -> str:
    pages = min(sample_pages, len(doc))
    chars = sum(len(doc[i].get_text().strip()) for i in range(pages))
    return "native" if chars / max(pages, 1) >= min_chars_per_page else "scanned"


# ---------------------------------------------------------------------------
# Step 2 — Native path: direct extraction with block geometry
# ---------------------------------------------------------------------------
def extract_native_page(page: fitz.Page) -> dict:
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        text = " ".join(
            span["text"] for line in b["lines"] for span in line["spans"]
        ).strip()
        if text:
            blocks.append({
                "bbox": [round(v, 1) for v in b["bbox"]],
                "text": text,
                "conf": 1.0,
            })
    return {
        "text": page.get_text().strip(),
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Step 3 — Scanned path: render page -> EasyOCR
# ---------------------------------------------------------------------------
def ocr_scanned_page(page: fitz.Page, lang_code: str, dpi: int = 300) -> dict:
    pix = page.get_pixmap(dpi=dpi)
    img = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))

    engine = get_ocr_engine(lang_code)
    result = engine.readtext(img) 

    blocks = []
    lines_text = []
    for box, text, conf in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        blocks.append({
            "bbox": [round(float(min(xs)), 1), round(float(min(ys)), 1),
                     round(float(max(xs)), 1), round(float(max(ys)), 1)],
            "text": text,
            "conf": round(float(conf), 3),
        })
        lines_text.append(text)

    return {"text": "\n".join(lines_text), "blocks": blocks}


# ---------------------------------------------------------------------------
# Steps 4 & 5 — Per-document routing + unified JSON output
# ---------------------------------------------------------------------------
def parse_filename(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None, "en", path.stem
    return m.group(1).lower(), m.group(2).lower(), m.group(3)


def process_pdf(path: Path, max_pages: int | None = None) -> dict:
    claimed_type, lang, doc_id = parse_filename(path)
    doc = fitz.open(path)

    detected_type = detect_pdf_type(doc)
    if claimed_type and claimed_type != detected_type:
        print(f"  [note] {path.name}: filename says '{claimed_type}' "
              f"but detector says '{detected_type}' — trusting the detector.")

    n_pages = len(doc) if max_pages is None else min(max_pages, len(doc))
    pages = []
    for i in range(n_pages):
        page = doc[i]
        if detected_type == "native":
            page_data = extract_native_page(page)
        else:
            page_data = ocr_scanned_page(page, lang)
        page_data["page_num"] = i + 1
        pages.append(page_data)
        print(f"  page {i + 1}/{n_pages} "
              f"({len(page_data['blocks'])} blocks)", end="\r")
    print()

    return {
        "doc_id": doc_id,
        "source_pdf": str(path),
        "pdf_type": detected_type,
        "language": lang,
        "num_pages_processed": n_pages,
        "num_pages_total": len(doc),
        "pages": pages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, help="process a single PDF")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="cap pages per doc (useful for fast iteration)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [args.file] if args.file else sorted(RAW_DIR.glob("*.pdf"))
    if not targets:
        print(f"No PDFs found in {RAW_DIR}/ — check your paths.")
        return

    for pdf_path in targets:
        print(f"Processing {pdf_path.name} ...")
        result = process_pdf(pdf_path, max_pages=args.max_pages)
        out_path = OUT_DIR / f"{result['doc_id']}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"  -> {out_path}  "
              f"[{result['pdf_type']}, {result['language']}, "
              f"{result['num_pages_processed']} pages]")


if __name__ == "__main__":
    main()