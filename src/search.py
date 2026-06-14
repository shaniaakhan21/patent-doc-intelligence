"""
search.py — Stage 4 of the patent document intelligence pipeline.

FastAPI application exposing hybrid patent search:
  - BM25 keyword search (ElasticSearch's native text matching)
  - Dense vector kNN search (semantic similarity via sentence-transformers)
  - Hybrid mode: weighted combination of both

Usage:
    # Start the API server
    uvicorn src.search:app --reload --port 8000

    # Or run a quick CLI search (no server needed)
    python src/search.py "neural network image classification"
    python src/search.py "OCR document recognition" --mode hybrid

API endpoints:
    GET  /search?q=neural+network&mode=hybrid&top_k=5
    GET  /health
    GET  /stats
"""

import argparse
import json
import sys
from pathlib import Path

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

INDEX_NAME = "patents"
ES_URL = "http://localhost:9200"
MODEL_NAME = "all-MiniLM-L6-v2"

# Lazy-loaded globals (initialized on first request)
_es = None
_model = None


def get_es():
    global _es
    if _es is None:
        _es = Elasticsearch(ES_URL)
    return _es


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Classic keyword search using ElasticSearch's BM25 algorithm.
    
    Searches across title, abstract, claims, and full text with
    boosted weights: title matches are worth 3x, abstract 2x.
    This is the same algorithm that underlies Lucene/Solr.
    """
    es = get_es()
    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^3",        # title matches weighted 3x
                    "abstract^2",     # abstract matches weighted 2x
                    "claims_text^1.5",
                    "full_text",
                ],
                "type": "best_fields",
            }
        },
        "size": top_k,
        "_source": {
            "excludes": ["embedding", "full_text"]  # don't return bulky fields
        },
    }
    
    results = es.search(index=INDEX_NAME, body=body)
    return _format_results(results)


def dense_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search using dense vector similarity (kNN).
    
    Embeds the query with the same model used at index time,
    then finds the closest patent embeddings by cosine similarity.
    Catches meaning even when exact keywords don't match
    (e.g., "picture categorization" finds "image classification").
    """
    es = get_es()
    model = get_model()
    
    query_vector = model.encode(query).tolist()
    
    body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 10,
        },
        "_source": {
            "excludes": ["embedding", "full_text"]
        },
    }
    
    results = es.search(index=INDEX_NAME, body=body)
    return _format_results(results)


def hybrid_search(query: str, top_k: int = 5,
                  bm25_weight: float = 0.5, dense_weight: float = 0.5) -> list[dict]:
    """
    Hybrid search: combine BM25 keyword scores with dense vector similarity.
    
    Strategy: Reciprocal Rank Fusion (RRF). Get top results from both
    BM25 and dense search, then merge by combining their reciprocal ranks.
    A document ranked #1 by BM25 and #3 by dense gets a higher fused score
    than one ranked #5 by both.
    
    This is the state-of-the-art approach used in modern search systems —
    keywords catch exact terms, vectors catch semantic meaning, and fusion
    gives you the best of both.
    """
    k_constant = 60  # RRF smoothing constant
    
    # Get results from both methods (fetch more than needed for better fusion)
    fetch_k = top_k * 3
    bm25_results = bm25_search(query, top_k=fetch_k)
    dense_results = dense_search(query, top_k=fetch_k)
    
    # Build reciprocal rank scores
    scores = {}  # doc_id -> {score, doc}
    
    for rank, result in enumerate(bm25_results):
        doc_id = result["doc_id"]
        rr_score = bm25_weight / (k_constant + rank + 1)
        scores[doc_id] = {
            "score": rr_score,
            "bm25_rank": rank + 1,
            "dense_rank": None,
            "doc": result,
        }
    
    for rank, result in enumerate(dense_results):
        doc_id = result["doc_id"]
        rr_score = dense_weight / (k_constant + rank + 1)
        if doc_id in scores:
            scores[doc_id]["score"] += rr_score
            scores[doc_id]["dense_rank"] = rank + 1
        else:
            scores[doc_id] = {
                "score": rr_score,
                "bm25_rank": None,
                "dense_rank": rank + 1,
                "doc": result,
            }
    
    # Sort by fused score and return top_k
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    results = []
    for item in ranked:
        doc = item["doc"]
        doc["hybrid_score"] = round(item["score"], 4)
        doc["bm25_rank"] = item["bm25_rank"]
        doc["dense_rank"] = item["dense_rank"]
        results.append(doc)
    
    return results


def _format_results(es_response: dict) -> list[dict]:
    """Convert raw ElasticSearch response to clean result list."""
    results = []
    for hit in es_response.get("hits", {}).get("hits", []):
        doc = hit["_source"]
        doc["score"] = hit.get("_score")
        results.append(doc)
    return results


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, Query
    
    app = FastAPI(
        title="Patent Document Search API",
        description="Hybrid BM25 + semantic search over WIPO patent documents",
        version="1.0.0",
    )
    
    @app.get("/search")
    def search_endpoint(
        q: str = Query(..., description="Search query"),
        mode: str = Query("hybrid", description="Search mode: bm25, dense, or hybrid"),
        top_k: int = Query(5, description="Number of results to return", ge=1, le=50),
    ):
        """Search indexed patents using BM25, dense vector, or hybrid mode."""
        if mode == "bm25":
            results = bm25_search(q, top_k=top_k)
        elif mode == "dense":
            results = dense_search(q, top_k=top_k)
        else:
            results = hybrid_search(q, top_k=top_k)
        
        return {
            "query": q,
            "mode": mode,
            "num_results": len(results),
            "results": results,
        }
    
    @app.get("/health")
    def health():
        """Check if ElasticSearch is reachable."""
        es = get_es()
        return {"status": "ok" if es.ping() else "error", "es_url": ES_URL}
    
    @app.get("/stats")
    def stats():
        """Return index statistics."""
        es = get_es()
        count = es.count(index=INDEX_NAME)
        return {
            "index": INDEX_NAME,
            "document_count": count["count"],
        }

except ImportError:
    app = None  # FastAPI not installed; CLI-only mode


# ---------------------------------------------------------------------------
# CLI mode (for quick testing without starting the server)
# ---------------------------------------------------------------------------

def print_results(results: list[dict], mode: str):
    """Pretty-print search results to the terminal."""
    print(f"\n{'='*60}")
    print(f"  {len(results)} results ({mode} search)")
    print(f"{'='*60}\n")
    
    for i, r in enumerate(results):
        title = (r.get("title") or "???")[:70]
        doc_id = r.get("doc_id", "???")
        lang = r.get("language", "?")
        
        print(f"  {i+1}. [{doc_id}] ({lang})")
        print(f"     {title}")
        
        if r.get("abstract"):
            abstract_preview = r["abstract"][:120] + "..."
            print(f"     {abstract_preview}")
        
        # Show ranking info for hybrid mode
        if r.get("hybrid_score") is not None:
            bm25_r = r.get("bm25_rank") or "-"
            dense_r = r.get("dense_rank") or "-"
            print(f"     hybrid: {r['hybrid_score']} "
                  f"(BM25 rank: {bm25_r}, dense rank: {dense_r})")
        elif r.get("score") is not None:
            print(f"     score: {r['score']:.4f}")
        
        print()


def cli_search():
    """Run a search from the command line."""
    ap = argparse.ArgumentParser(description="Search indexed patent documents")
    ap.add_argument("query", help="Search query text")
    ap.add_argument("--mode", default="hybrid",
                    choices=["bm25", "dense", "hybrid"],
                    help="Search mode (default: hybrid)")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of results (default: 5)")
    args = ap.parse_args()
    
    if args.mode == "bm25":
        results = bm25_search(args.query, top_k=args.top_k)
    elif args.mode == "dense":
        results = dense_search(args.query, top_k=args.top_k)
    else:
        results = hybrid_search(args.query, top_k=args.top_k)
    
    print_results(results, args.mode)


if __name__ == "__main__":
    cli_search()