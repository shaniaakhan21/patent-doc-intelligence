# Patent Document Intelligence Pipeline

An end-to-end system for extracting, structuring, and searching patent documents from WIPO's PatentScope. Handles native (born-digital) and scanned PDFs across multiple languages (English, Chinese, German), producing unified structured output and exposing hybrid search via a REST API.

Built as a demonstration of document AI techniques applied to intellectual property — the kind of work performed by WIPO's Advanced Technology Applications Center (ATAC).

## Architecture

```
Patent PDF (any language, native or scanned)
        │
   ┌────▼─────┐
   │  ocr.py  │─── Detect PDF type (native vs scanned)
   └────┬─────┘    Route to PyMuPDF (native) or EasyOCR (scanned)
        │          Language-routed models (en, zh, de)
        ▼
  Unified JSON per document
  (text + bounding boxes + confidence scores)
        │
   ┌────▼──────────┐
   │  extract.py   │─── Parse INID codes: (54) title, (57) abstract
   └────┬──────────┘    Segment body by section headings
        │               Parse and classify claims
        ▼
  Structured patent record
  (title, abstract, claims, sections, figures)
        │
   ┌────▼─────────┐
   │  index.py    │─── BM25 text indexing (ElasticSearch/Lucene)
   └────┬─────────┘    Dense vector embeddings (sentence-transformers)
        │
        ▼
   ┌────────────┐
   │  search.py │─── Hybrid search: BM25 + kNN + Reciprocal Rank Fusion
   └────────────┘    REST API via FastAPI
        │
        ▼
   GET /search?q=neural+network+OCR&mode=hybrid
```

## Key Design Decisions

**Native vs. scanned routing.** The pipeline's first step classifies each PDF by extracting text from sample pages — if meaningful text comes out, it's born-digital and uses PyMuPDF's lossless extraction; otherwise it routes to OCR. Some modern PDFs store text as vector outlines (drawn shapes rather than text objects), producing zero extractable characters despite appearing digital. The detector correctly routes these to OCR alongside true scans.

**Language-routed OCR.** Document language is encoded in the filename convention and used to load the correct EasyOCR model. Chinese patents get `ch_sim + en` (for Latin-script numbers that appear in all patents); German gets `de + en`. The engine is isolated behind a factory function — originally targeting PaddleOCR, swapped to EasyOCR due to a platform-specific runtime bug, with zero changes to downstream code.

**Unified output schema.** Whether a page went through the native path or OCR, the output is identical: full text, text blocks with bounding-box coordinates `[x0, y0, x1, y1]`, and a confidence score (1.0 for embedded text, model confidence for OCR). Downstream stages don't need to know which path produced the data.

**Hybrid retrieval.** Combines BM25 keyword search (exact term matching via ElasticSearch/Lucene) with dense vector semantic search (meaning-based similarity via sentence-transformers). Results are fused using Reciprocal Rank Fusion (RRF). This catches both exact terminology matches and semantic synonyms — e.g., the query "reading words from scanned documents" (zero keyword overlap with "OCR") correctly returns OCR patents as the top results.

## Example: Semantic Search in Action

Query: `"reading words from scanned documents"` (dense mode — no keyword matching)

| Rank | Document | Title | Score |
|------|----------|-------|-------|
| 1 | WO2019009916 | Document image processing apparatus | 0.724 |
| 2 | WO2021087334 | Neural Network-Based Optical Character Recognition | 0.711 |
| 3 | WO2023206271 | Transformer for Optical Character Recognition | 0.707 |
| 4 | WO2019238976 | Image Classification Using Neural Networks | 0.623 |
| 5 | CN112348007 | 一种基于神经网络的光学字符识别方法 | 0.589 |

The top 3 results are OCR patents despite the query containing none of the words "OCR," "optical," "character," or "recognition." The score gap between ranks 3 and 4 (0.707 → 0.623) shows the model distinguishes between directly relevant patents (OCR = reading text) and tangentially related ones (image classification).

## Dataset

10 patent documents sourced from WIPO PatentScope and Google Patents:

| Category | Count | Source |
|----------|-------|--------|
| Modern English (2018–2025) | 5 | PatentScope (PCT collection) |
| Scanned English (1980–1993) | 2 | PatentScope (PCT collection) |
| Chinese | 2 | Google Patents (CN collection) |
| German | 1 | PatentScope (PCT collection) |

Documents selected to cover a range of applicants, layouts, figure styles, and languages. Includes both text-heavy patents and diagram-rich filings, as well as PDFs with text stored as vector outlines (a common real-world edge case).

## Project Structure

```
patent-doc-intelligence/
├── src/
│   ├── ocr.py              # Stage 1: PDF type detection + text extraction / OCR
│   ├── extract.py           # Stage 2: Layout analysis + structured field extraction
│   ├── index.py             # Stage 3: ElasticSearch indexing (BM25 + dense vectors)
│   ├── search.py            # Stage 4: Hybrid search + FastAPI REST API
│   └── rename_dataset.py    # Utility: auto-rename PDFs by reading their front page
├── data/
│   ├── raw/                 # Source PDFs ({native|scanned}_{lang}_{docid}.pdf)
│   ├── processed/           # OCR JSON outputs
│   └── extracted/           # Structured patent records
├── docs/
│   └── screenshots/         # API documentation screenshots
├── docker-compose.yml       # ElasticSearch + API deployment
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/shaniaakhan21/patent-doc-intelligence.git
cd patent-doc-intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Stage 1: OCR — extract text from all patent PDFs
python src/ocr.py                          # process all documents
python src/ocr.py --file data/raw/native_en_WO2021087334.pdf --max-pages 3  # single doc

# Stage 2: Extract — structure patent fields
python src/extract.py

# Stage 3: Index — start ElasticSearch and index documents
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.0

python src/index.py --recreate

# Stage 4: Search — query from CLI or start the API
python src/search.py "neural network image classification"
python src/search.py "reading words from scanned documents" --mode dense
python src/search.py "OCR text recognition" --mode bm25

# Start the REST API
python -m uvicorn src.search:app --port 8000
# Interactive docs at http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/search?q=...&mode=hybrid&top_k=5` | Hybrid patent search (modes: `bm25`, `dense`, `hybrid`) |
| GET | `/health` | ElasticSearch connection status |
| GET | `/stats` | Index document count |

## Pipeline Stages

### Stage 1 — OCR (`ocr.py`)

Classifies each PDF as native or scanned, routes to the appropriate extraction path, and produces one JSON file per document with unified schema.

- **Native path:** PyMuPDF direct text + block extraction with bounding boxes (confidence: 1.0)
- **Scanned path:** Page rendering at 300 DPI → EasyOCR with language-specific models
- **Language routing:** English (`en`), Chinese (`ch_sim`), German (`de`) — loaded on demand, cached per language

### Stage 2 — Structured Extraction (`extract.py`)

Parses OCR output into structured patent records using rule-based extraction:

- **Front page parsing:** INID code matching — (54) title, (57) abstract, (71) applicant, (72) inventors, (22) filing date, (51) IPC classification
- **Body segmentation:** Keyword-based section detection (Technical Field, Background, Summary, Detailed Description, Claims, Drawings)
- **Claims parsing:** Individual claim extraction with dependency classification (independent vs. dependent)

### Stage 3 — Search Indexing (`index.py`)

Indexes structured records into ElasticSearch with dual retrieval:

- **BM25 text fields:** Title (3x boost), abstract (2x), claims (1.5x), full text — powered by Lucene's inverted index
- **Dense vectors:** 384-dimensional embeddings via `all-MiniLM-L6-v2` (sentence-transformers), indexed for cosine kNN search
- **Embedding composition:** Title + abstract + first claim, capped at 2000 characters

### Stage 4 — Hybrid Search (`search.py`)

Three search modes exposed via FastAPI:

- **BM25:** Multi-field keyword search with field boosting
- **Dense:** Semantic kNN search via cosine similarity
- **Hybrid:** Reciprocal Rank Fusion (RRF) combining BM25 and dense rankings with configurable weights

## Requirements

- Python 3.10+
- Docker (for ElasticSearch)
- PyMuPDF, EasyOCR, sentence-transformers, elasticsearch, FastAPI

See `requirements.txt` for pinned versions.

## Limitations and Next Steps

- **OCR accuracy on 1980s scans:** Typewriter-era documents with degraded print produce noticeable errors (e.g., "ALGORITTHM"). A post-processing spell-check or language-model correction step would improve quality.
- **Vector-outline PDFs:** Some modern PDFs render text as vector paths rather than text objects, yielding zero extractable characters. The detector correctly routes these to OCR, but a secondary heuristic (checking for vector path density) could distinguish these from true scans.
- **Chinese/German section extraction:** Section heading patterns are tuned for English WIPO patents. Chinese patents use different headings (权利要求, 摘要) — extending the pattern set would improve CJK extraction.
- **Cross-lingual retrieval:** The current embedding model (`all-MiniLM-L6-v2`) is English-focused. Swapping to `paraphrase-multilingual-MiniLM-L12-v2` would improve cross-lingual semantic search.
- **Tesseract comparison:** A systematic EasyOCR vs. Tesseract evaluation on the scanned subset would quantify engine tradeoffs.
- **Image similarity:** Trademark logos and industrial design images could be embedded with CLIP for visual similarity search — a natural extension for the Global Brand Database use case.

## License

MIT