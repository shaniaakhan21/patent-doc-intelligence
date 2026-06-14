"""
extract.py — Stage 2 of the patent document intelligence pipeline.

Reads OCR JSON outputs (from ocr.py) and extracts structured patent fields:
  - title, abstract, claims, description sections, figure references
  
Uses rule-based extraction leveraging the known structure of WIPO PCT
patent documents — no ML models needed. Patent sections follow a rigid
order marked by standard headings.

Usage:
    python src/extract.py                                    # process all JSONs
    python src/extract.py --file data/processed/WO2021087334.json
"""

import argparse
import json
import re
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
EXTRACTED_DIR = Path("data/extracted")

# ---------------------------------------------------------------------------
# Section heading patterns (case-insensitive)
# These are the standard headings in WIPO PCT patent publications.
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "title": re.compile(
        r"(?:title|invention\s+title|\(54\)\s*title)", re.IGNORECASE
    ),
    "abstract": re.compile(
        r"(?:\(57\)\s*abstract|^abstract\b)", re.IGNORECASE
    ),
    "technical_field": re.compile(
        r"(?:technical\s+field|field\s+of\s+(?:the\s+)?invention)", re.IGNORECASE
    ),
    "background": re.compile(
        r"(?:background(?:\s+(?:of|art))?|prior\s+art|related\s+art)", re.IGNORECASE
    ),
    "summary": re.compile(
        r"(?:summary(?:\s+of\s+(?:the\s+)?invention)?|brief\s+summary)", re.IGNORECASE
    ),
    "detailed_description": re.compile(
        r"(?:detailed\s+description|description\s+of\s+(?:the\s+)?(?:preferred\s+)?embodiments?)",
        re.IGNORECASE,
    ),
    "claims": re.compile(
        r"^claims\b", re.IGNORECASE
    ),
    "drawings": re.compile(
        r"(?:brief\s+description\s+of\s+(?:the\s+)?drawings|description\s+of\s+(?:the\s+)?figures)",
        re.IGNORECASE,
    ),
}

# Publication number pattern on front page
PUBNUM_RE = re.compile(r"\b(WO|CN|EP|US|DE)\s*(\d{4})[/\s]*(\d{5,9})", re.IGNORECASE)

# Figure reference pattern in text (e.g., "Figure 1", "Fig. 3", "FIG. 12")
FIGURE_REF_RE = re.compile(r"(?:figure|fig\.?)\s*(\d+)", re.IGNORECASE)


def classify_block(text: str) -> str | None:
    """Check if a text block is a section heading. Returns section name or None."""
    clean = text.strip()
    # Section headings are typically short (under 100 chars)
    if len(clean) > 100:
        return None
    for section_name, pattern in SECTION_PATTERNS.items():
        if pattern.search(clean):
            return section_name
    return None


def extract_front_page(pages: list[dict]) -> dict:
    """
    Extract metadata from the front page (page 1).
    
    Patent front pages have a dense, structured layout with INID codes
    like (54) Title, (57) Abstract, (71) Applicant, (72) Inventor, etc.
    We grab the key ones.
    """
    if not pages:
        return {}
    
    page1_text = pages[0].get("text", "")
    page1_blocks = pages[0].get("blocks", [])
    
    result = {}
    
    # Publication number
    m = PUBNUM_RE.search(page1_text)
    if m:
        result["publication_number"] = f"{m.group(1)} {m.group(2)}/{m.group(3)}"
    
    # Title — look for (54) marker
    title_match = re.search(
        r"\(54\)\s*(?:Title\s*:?\s*)?(.+?)(?:\n\(|\n\n|$)",
        page1_text,
        re.IGNORECASE | re.DOTALL,
    )
    if title_match:
        result["title"] = " ".join(title_match.group(1).split())
    
    # Abstract — look for (57) marker
    abstract_match = re.search(
        r"\(57\)\s*(?:Abstract\s*:?\s*)?(.+?)(?:\n\(|\Z)",
        page1_text,
        re.IGNORECASE | re.DOTALL,
    )
    if abstract_match:
        abstract_text = " ".join(abstract_match.group(1).split())
        # Abstracts can be long; cap at ~2000 chars to avoid grabbing noise
        result["abstract"] = abstract_text[:2000]
    
    # Applicant — (71) marker
    applicant_match = re.search(
        r"\(71\)\s*(?:Applicant\s*:?\s*)?(.+?)(?:\n\(|$)",
        page1_text,
        re.IGNORECASE,
    )
    if applicant_match:
        result["applicant"] = " ".join(applicant_match.group(1).split())
    
    # Inventors — (72) marker
    inventor_match = re.search(
        r"\(72\)\s*(?:Inventors?\s*:?\s*)?(.+?)(?:\n\(|$)",
        page1_text,
        re.IGNORECASE,
    )
    if inventor_match:
        result["inventors"] = " ".join(inventor_match.group(1).split())
    
    # Filing date — (22) marker
    date_match = re.search(
        r"\(22\)\s*(?:International\s+Filing\s+Date\s*:?\s*)?(.+?)(?:\n\(|$)",
        page1_text,
        re.IGNORECASE,
    )
    if date_match:
        result["filing_date"] = " ".join(date_match.group(1).split())
    
    # IPC classification — (51) marker
    ipc_match = re.search(
        r"\(51\)\s*(?:International\s+Patent\s+Classification\s*:?\s*)?(.+?)(?:\n\(|$)",
        page1_text,
        re.IGNORECASE | re.DOTALL,
    )
    if ipc_match:
        result["ipc_classification"] = " ".join(ipc_match.group(1).split())
    
    return result


def segment_body(pages: list[dict]) -> dict[str, str]:
    """
    Segment the patent body (pages 2+) into sections using heading detection.
    
    Strategy: walk through all blocks on all pages sequentially. When a block
    matches a section heading pattern, start accumulating text under that
    section name. Text before any recognized heading goes into 'preamble'.
    """
    sections = {}
    current_section = "preamble"
    current_text = []
    
    for page in pages[1:]:  # skip front page (page 1)
        for block in page.get("blocks", []):
            text = block.get("text", "").strip()
            if not text:
                continue
            
            # Check if this block is a section heading
            section = classify_block(text)
            if section:
                # Save the accumulated text for the previous section
                if current_text:
                    sections[current_section] = "\n".join(current_text).strip()
                current_section = section
                current_text = []
            else:
                current_text.append(text)
    
    # Don't forget the last section
    if current_text:
        sections[current_section] = "\n".join(current_text).strip()
    
    return sections


def extract_claims(claims_text: str) -> list[dict]:
    """
    Parse individual claims from the claims section text.
    
    Claims are numbered (1. ... 2. ... etc.) and have two types:
    - Independent claims: standalone
    - Dependent claims: reference another claim ("The method of claim 1, ...")
    """
    if not claims_text:
        return []
    
    # Split on claim numbers at the start of lines
    claim_splits = re.split(r"\n\s*(\d+)\.\s+", "\n" + claims_text)
    
    claims = []
    # claim_splits alternates: [preamble, num, text, num, text, ...]
    for i in range(1, len(claim_splits) - 1, 2):
        num = int(claim_splits[i])
        text = claim_splits[i + 1].strip()
        
        # Detect dependency
        dep_match = re.search(
            r"(?:of|in|to)\s+claim\s+(\d+)", text, re.IGNORECASE
        )
        
        claims.append({
            "number": num,
            "text": text,
            "type": "dependent" if dep_match else "independent",
            "depends_on": int(dep_match.group(1)) if dep_match else None,
        })
    
    return claims


def extract_figure_refs(full_text: str) -> list[int]:
    """Find all figure numbers referenced in the text."""
    refs = set()
    for m in FIGURE_REF_RE.finditer(full_text):
        refs.add(int(m.group(1)))
    return sorted(refs)


def process_document(ocr_json_path: Path) -> dict:
    """
    Main extraction: read an OCR JSON, produce a structured patent record.
    """
    data = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    pages = data.get("pages", [])
    
    # 1. Front page metadata
    front_page = extract_front_page(pages)
    
    # 2. Body segmentation
    sections = segment_body(pages)
    
    # 3. Parse claims if found
    claims_list = extract_claims(sections.get("claims", ""))
    
    # 4. Collect full text for search indexing
    full_text = "\n\n".join(
        page.get("text", "") for page in pages
    )
    
    # 5. Figure references
    figure_refs = extract_figure_refs(full_text)
    
    # Build the structured record
    record = {
        "doc_id": data.get("doc_id"),
        "source_pdf": data.get("source_pdf"),
        "pdf_type": data.get("pdf_type"),
        "language": data.get("language"),
        "num_pages": data.get("num_pages_total"),
        
        # Front page metadata
        "publication_number": front_page.get("publication_number"),
        "title": front_page.get("title"),
        "abstract": front_page.get("abstract"),
        "applicant": front_page.get("applicant"),
        "inventors": front_page.get("inventors"),
        "filing_date": front_page.get("filing_date"),
        "ipc_classification": front_page.get("ipc_classification"),
        
        # Body sections (each as a text string)
        "sections": {
            k: v for k, v in sections.items()
            if k != "preamble" and v  # skip empty sections and preamble
        },
        
        # Parsed claims
        "claims": claims_list,
        "num_claims": len(claims_list),
        "num_independent_claims": sum(
            1 for c in claims_list if c["type"] == "independent"
        ),
        
        # Figure references found in text
        "figure_references": figure_refs,
        
        # Full concatenated text (for search indexing in Stage 3)
        "full_text": full_text,
    }
    
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, help="process a single OCR JSON")
    args = ap.parse_args()
    
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.file:
        targets = [args.file]
    else:
        targets = sorted(PROCESSED_DIR.glob("*.json"))
    
    if not targets:
        print(f"No JSON files found in {PROCESSED_DIR}/ — run ocr.py first.")
        return
    
    for json_path in targets:
        print(f"Extracting {json_path.name} ...")
        record = process_document(json_path)
        
        out_path = EXTRACTED_DIR / json_path.name
        out_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        # Summary
        title = (record.get("title") or "???")[:60]
        n_sections = len(record.get("sections", {}))
        n_claims = record.get("num_claims", 0)
        figs = record.get("figure_references", [])
        
        print(f"  title: {title}")
        print(f"  sections: {n_sections} | claims: {n_claims} | "
              f"figures referenced: {len(figs)}")
        print(f"  -> {out_path}")
    
    print(f"\nDone. {len(targets)} documents extracted to {EXTRACTED_DIR}/")


if __name__ == "__main__":
    main()