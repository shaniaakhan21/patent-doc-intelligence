"""
rename_dataset.py — auto-rename raw PDFs to {native|scanned}_{lang}_{docid}.pdf
by reading each document's own front page.
"""

from pydoc import doc
import re
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

RAW_DIR = Path("data/raw")

# Publication numbers as they appear on front pages:
#   "WO 2021/087334", "WO 92/01234", "CN 112348007", etc.
PUBNUM_RE = re.compile(r"\b(WO|CN|EP|US|DE)[\s/]*(\d{2,4})[\s/]*(\d{5,9})", re.I)

# Already-correctly-named files: skip them
DONE_RE = re.compile(r"^(native|scanned)_[a-z]{2}_", re.I)


def detect_type(doc, sample=5, min_chars=50):
    n = min(sample, len(doc))
    chars = sum(len(doc[i].get_text().strip()) for i in range(n))
    return "native" if chars / max(n, 1) >= min_chars else "scanned"


def detect_language(text):
    """Crude but effective: CJK chars -> zh, German stopwords -> de, else en."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if cjk > 20:
        return "zh"
    words = set(re.findall(r"[a-zäöüß]+", text.lower()))
    german_hits = words & {"und", "der", "die", "das", "verfahren",
                           "vorrichtung", "mit", "für", "einer", "wird"}
    return "de" if len(german_hits) >= 3 else "en"


def page1_text(doc, pdf_type):
    """Embedded text for native docs; quick OCR of page 1 for scans."""
    if pdf_type == "native":
        return doc[0].get_text()
    import easyocr
    pix = doc[0].get_pixmap(dpi=200)
    img = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    result = reader.readtext(img)
    if not result:
        return ""
    return "\n".join(text for _, text, _ in result)


def find_pubnum(text):
    m = PUBNUM_RE.search(text)
    if not m:
        return None
    office, part1, part2 = m.group(1).upper(), m.group(2), m.group(3)
    return f"{office}{part1}{part2}"


def main():
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        if DONE_RE.match(pdf.name):
            print(f"skip (already named): {pdf.name}")
            continue

        doc = fitz.open(pdf)
        pdf_type = detect_type(doc)
        text = page1_text(doc, pdf_type)
        lang = detect_language(text)
        pubnum = find_pubnum(text)
        doc.close()

        if not pubnum:
            print(f"??  couldn't find a publication number in {pdf.name} "
                  f"— rename this one manually.")
            continue

        new_name = f"{pdf_type}_{lang}_{pubnum}.pdf"
        target = RAW_DIR / new_name
        if target.exists():
            print(f"!!  {new_name} already exists — skipping {pdf.name}")
            continue

        pdf.rename(target)
        print(f"{pdf.name[:40]:<42} -> {new_name}")


if __name__ == "__main__":
    main()