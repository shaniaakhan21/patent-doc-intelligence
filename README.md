# Patent Document Intelligence Pipeline

An end-to-end system for extracting, structuring, and searching patent documents from WIPO's PatentScope. Handles native (born-digital) and scanned PDFs across multiple languages (English, Chinese, German), producing unified structured output ready for downstream search and analysis.

Built as a demonstration of document AI techniques applied to intellectual property — the kind of work performed by WIPO's Advanced Technology Applications Center (ATAC).

## What it does

```
Patent PDF (any language, native or scanned)
    │
    ├─ Native? ──→ Direct text + block extraction (PyMuPDF)
    │
    └─ Scanned? ─→ Page rendering → OCR (EasyOCR, language-routed)
                         │
                         ▼
              Unified JSON per document
           (text + bounding boxes + confidence)
                         │
                         ▼
              Structured extraction (extract.py)
           (title, abstract, claims, figures)
                         │
                         ▼
              Hybrid search index (ElasticSearch)
           (BM25 + dense vector via sentence-transformers)
                         │
                         ▼
              REST API (FastAPI + Docker)
```

## Key design decisions

**Native vs. scanned routing.** The pipeline's first step is a heuristic classifier: extract text from a few sample pages — if meaningful text comes out, it's born-digital and we use PyMuPDF's lossless extraction; if not, it's a scan and we route to OCR. This avoids wasting compute on native PDFs and avoids trusting garbage text layers on poorly-processed scans. The threshold is tunable (`min_chars_per_page`).

**Language-routed OCR.** Document language is encoded in the filename convention (`native_zh_CN112348007.pdf`) and used to load the correct EasyOCR model. Chinese patents get `ch_sim + en` (for Latin-script numbers and codes that appear in all patents); German gets `de + en`. This is a simple but effective form of multilingual document processing.

**Unified output schema.** Whether a page went through the native path or the OCR path, the output is identical: full text, a list of text blocks with bounding-box coordinates `[x0, y0, x1, y1]`, and a confidence score (1.0 for embedded text, model confidence for OCR). Downstream stages (extraction, indexing, search) don't need to know which path produced the data.

**Engine-agnostic architecture.** The OCR engine is isolated behind a simple factory function. The project originally targeted PaddleOCR but was swapped to EasyOCR due to a Windows-specific oneDNN runtime bug — with zero changes to extraction, indexing, or search code. This modularity is intentional.

## Dataset

12 patent documents sourced from WIPO PatentScope and Google Patents:

| Category | Count | Source |
|----------|-------|--------|
| Native English (2018–2025) | 6 | PatentScope (PCT collection) |
| Scanned English (1980–1993) | 3 | PatentScope (PCT collection) |
| Native Chinese | 2 | Google Patents (CN collection) |
| Native German | 1 | PatentScope (PCT collection) |

Documents were selected to cover a range of applicants, layouts, and figure styles. The dataset includes both text-heavy patents and diagram-rich filings.

## Project structure

```
patent-doc-intelligence/
├── src/
│   ├── ocr.py              # Stage 1: PDF type detection + text extraction / OCR
│   ├── extract.py           # Stage 2: Layout analysis + structured field extraction
│   ├── index.py             # Stage 3: ElasticSearch indexing (BM25 + dense vectors)
│   ├── search.py            # Stage 4: Hybrid search queries
│   └── rename_dataset.py    # Utility: auto-rename raw PDFs by reading their front page
├── data/
│   ├── raw/                 # Source PDFs (naming: {native|scanned}_{lang}_{docid}.pdf)
│   └── processed/           # JSON outputs from ocr.py (gitignored)
├── examples/                # Sample outputs for quick inspection
├── requirements.txt
└── README.md
```

## Quick start

```bash
# Clone and setup
git clone https://github.com/shaniaakhan21/patent-doc-intelligence.git
cd patent-doc-intelligence
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run OCR on a single document (fast smoke test)
python src/ocr.py --file data/raw/native_en_WO2021087334.pdf --max-pages 3

# Run OCR on everything
python src/ocr.py

# Auto-rename messy download filenames (reads each PDF's front page)
python src/rename_dataset.py
```

## Pipeline stages

### Stage 1 — OCR (`src/ocr.py`)

Classifies each PDF as native or scanned, routes to the appropriate extraction path, and produces one JSON file per document in `data/processed/`.

```bash
python src/ocr.py                                              # all documents
python src/ocr.py --file data/raw/native_zh_CN112348007.pdf    # single document
python src/ocr.py --max-pages 5                                # cap pages (fast iteration)
```

### Stage 2 — Structured extraction (`src/extract.py`)

*In progress.* Uses bounding-box geometry from Stage 1 to identify patent sections: title, abstract, claims, drawings. Outputs structured JSON records.

### Stage 3 — Search index (`src/index.py`)

*In progress.* Indexes structured records into ElasticSearch with both BM25 text fields and dense vector embeddings (via `sentence-transformers`) for hybrid retrieval.

### Stage 4 — Search API (`src/search.py`)

*In progress.* FastAPI endpoint exposing hybrid search (BM25 + kNN) over the indexed patent collection.

## Requirements

- Python 3.10+
- PyMuPDF (PDF reading and page rendering)
- EasyOCR (multilingual OCR for scanned documents)
- NumPy, Pillow (image handling)

See `requirements.txt` for pinned versions.

## Limitations and next steps

- **OCR accuracy on 1980s scans:** Typewriter-era documents with degraded print quality produce noticeable OCR errors. A post-processing spell-check or language-model correction step would improve accuracy.
- **Layout analysis:** The current bounding-box output captures *where* text sits on the page but doesn't yet classify regions (title vs. abstract vs. claims). LayoutParser or LayoutLMv3 integration is the planned next step.
- **Tesseract comparison:** A systematic PaddleOCR/EasyOCR vs. Tesseract comparison on the scanned subset would quantify engine tradeoffs — planned for the evaluation section.
- **Image similarity:** Trademark logos and industrial design images could be embedded with CLIP for visual similarity search — a natural extension given the multimodal architecture.

## License

MIT