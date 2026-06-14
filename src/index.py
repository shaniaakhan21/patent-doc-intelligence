"""
index.py — Stage 3 of the patent document intelligence pipeline.

Indexes structured patent records (from extract.py) into ElasticSearch
with two retrieval modes:
  - BM25 (keyword search on text fields)
  - Dense vector (semantic search via sentence-transformers embeddings)

This enables hybrid retrieval: combining exact keyword matches with
meaning-based similarity — the same approach used in modern patent
search systems.

Usage:
    python src/index.py                # index all extracted documents
    python src/index.py --recreate     # delete and recreate the index first

Requirements:
    - ElasticSearch running on localhost:9200 (see README for Docker command)
    - pip install elasticsearch sentence-transformers
"""

import argparse
import json
from pathlib import Path

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

EXTRACTED_DIR = Path("data/extracted")
INDEX_NAME = "patents"
ES_URL = "http://localhost:9200"

# Model for generating dense vectors — small, fast, good for English
# For multilingual, swap to "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384  # output dimension of all-MiniLM-L6-v2


def get_es_client() -> Elasticsearch:
    """Connect to ElasticSearch and verify it's reachable."""
    es = Elasticsearch(ES_URL)
    if not es.ping():
        raise ConnectionError(
            f"Cannot reach ElasticSearch at {ES_URL}. "
            "Start it with: docker run -d --name elasticsearch "
            "-p 9200:9200 -e discovery.type=single-node "
            "-e xpack.security.enabled=false "
            "docker.elastic.co/elasticsearch/elasticsearch:8.13.0"
        )
    info = es.info()
    print(f"Connected to ElasticSearch {info['version']['number']}")
    return es


def create_index(es: Elasticsearch, recreate: bool = False):
    """
    Create the patents index with mappings for both BM25 and dense vector search.
    
    The mapping defines:
    - text fields (title, abstract, full_text) → BM25 keyword search
    - dense_vector field (embedding) → kNN semantic search
    - keyword fields (doc_id, language, etc.) → exact filtering
    """
    if es.indices.exists(index=INDEX_NAME):
        if recreate:
            es.indices.delete(index=INDEX_NAME)
            print(f"Deleted existing index '{INDEX_NAME}'")
        else:
            print(f"Index '{INDEX_NAME}' already exists (use --recreate to rebuild)")
            return

    mapping = {
        "mappings": {
            "properties": {
                # Identity
                "doc_id": {"type": "keyword"},
                "source_pdf": {"type": "keyword"},
                "pdf_type": {"type": "keyword"},
                "language": {"type": "keyword"},
                
                # Metadata (keyword = exact match, not analyzed)
                "publication_number": {"type": "keyword"},
                "applicant": {"type": "text"},
                "inventors": {"type": "text"},
                "filing_date": {"type": "keyword"},
                "ipc_classification": {"type": "text"},
                
                # Searchable text fields (BM25-analyzed)
                "title": {"type": "text", "analyzer": "standard"},
                "abstract": {"type": "text", "analyzer": "standard"},
                "claims_text": {"type": "text", "analyzer": "standard"},
                "full_text": {"type": "text", "analyzer": "standard"},
                
                # Structured data
                "num_claims": {"type": "integer"},
                "num_independent_claims": {"type": "integer"},
                "num_pages": {"type": "integer"},
                "figure_references": {"type": "integer"},
                
                # Dense vector for semantic search
                "embedding": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
    }

    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created index '{INDEX_NAME}' with BM25 + dense vector mappings")


def build_embedding_text(record: dict) -> str:
    """
    Compose the text that gets embedded for semantic search.
    
    We combine title + abstract + first claim — these carry the most
    semantic signal about what the patent covers. Full text would be
    too noisy and too long for a single embedding.
    """
    parts = []
    if record.get("title"):
        parts.append(record["title"])
    if record.get("abstract"):
        parts.append(record["abstract"])
    # Add first independent claim if available
    claims = record.get("claims", [])
    if claims:
        first = claims[0].get("text", "")
        if first:
            parts.append(first[:500])  # cap length
    
    return " ".join(parts)[:2000]  # model max ~256 tokens, this is safe


def index_documents(es: Elasticsearch, model: SentenceTransformer):
    """Read all extracted JSONs, embed them, and index into ElasticSearch."""
    files = sorted(EXTRACTED_DIR.glob("*.json"))
    if not files:
        print(f"No files found in {EXTRACTED_DIR}/ — run extract.py first.")
        return
    
    print(f"Indexing {len(files)} documents...")
    
    for i, json_path in enumerate(files):
        record = json.loads(json_path.read_text(encoding="utf-8"))
        
        # Build the text to embed
        embed_text = build_embedding_text(record)
        embedding = model.encode(embed_text).tolist()
        
        # Combine claims into one text field for BM25 search
        claims_text = "\n".join(
            c.get("text", "") for c in record.get("claims", [])
        )
        
        # The document to index
        doc = {
            "doc_id": record.get("doc_id"),
            "source_pdf": record.get("source_pdf"),
            "pdf_type": record.get("pdf_type"),
            "language": record.get("language"),
            "publication_number": record.get("publication_number"),
            "title": record.get("title"),
            "abstract": record.get("abstract"),
            "applicant": record.get("applicant"),
            "inventors": record.get("inventors"),
            "filing_date": record.get("filing_date"),
            "ipc_classification": record.get("ipc_classification"),
            "claims_text": claims_text,
            "full_text": record.get("full_text", ""),
            "num_claims": record.get("num_claims", 0),
            "num_independent_claims": record.get("num_independent_claims", 0),
            "num_pages": record.get("num_pages", 0),
            "figure_references": record.get("figure_references", []),
            "embedding": embedding,
        }
        
        es.index(index=INDEX_NAME, id=record.get("doc_id"), document=doc)
        title = (record.get("title") or "???")[:50]
        print(f"  [{i+1}/{len(files)}] {record.get('doc_id')}: {title}")
    
    # Refresh so documents are immediately searchable
    es.indices.refresh(index=INDEX_NAME)
    print(f"\nDone. {len(files)} documents indexed in '{INDEX_NAME}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true",
                    help="delete and recreate the index")
    args = ap.parse_args()
    
    es = get_es_client()
    create_index(es, recreate=args.recreate)
    
    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    index_documents(es, model)


if __name__ == "__main__":
    main()