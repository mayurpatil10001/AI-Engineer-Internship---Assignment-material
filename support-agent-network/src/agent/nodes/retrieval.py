"""
Retrieval node: build and query the local vector index over KB docs + resolved cases.

Design decisions (see docs/design.md §3.2 and §T-2):
  - Paragraph-level chunking (~100-300 tokens) rather than whole-file embedding,
    so the verification node can cite specific passages with char offsets.
  - Linear numpy cosine scan (no FAISS) because ~60 chunks makes FAISS overhead
    larger than query time.
  - Superseded resolved cases are indexed but tagged with is_superseded=True.
    The generation node is instructed (via prompt) to ignore superseded content.
  - retrieval_sufficient=False when top-1 cosine score < RETRIEVAL_THRESHOLD (0.35).

The index is built ONCE at module import time (first call) and cached.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.agent.state import AgentState, Evidence

logger = logging.getLogger("orbitdesk.agent")

# ── Config ────────────────────────────────────────────────────────────────────

RETRIEVAL_THRESHOLD: float = float(__import__("os").getenv("RETRIEVAL_THRESHOLD", "0.35"))
TOP_K: int = 5  # number of chunks to return; all 5 are passed to generation

# Paths — relative to repo root; resolved at runtime
_KB_DIR = Path("data/kb")
_CASES_FILE = Path("data/resolved_cases.json")


# ── Document loading ──────────────────────────────────────────────────────────

def _load_kb_docs(kb_dir: Path) -> list[dict]:
    """
    Load all markdown KB documents and return a list of document records.

    Each record: {source_id, full_text, file_path}
    The source_id is extracted from the YAML frontmatter 'document_id' field.
    """
    records = []
    for md_file in sorted(kb_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        # Extract document_id from frontmatter
        m = re.search(r"document_id:\s*(\S+)", text)
        source_id = m.group(1) if m else md_file.stem
        records.append({"source_id": source_id, "full_text": text, "file_path": str(md_file)})
    return records


def _load_resolved_cases(cases_file: Path) -> list[dict]:
    """
    Load resolved cases JSON and return a list of case records.

    Each record: {source_id, full_text, is_superseded}
    The text is a concatenation of title + symptoms + resolution for embedding.
    """
    if not cases_file.exists():
        logger.warning("resolved_cases.json not found", extra={"path": str(cases_file)})
        return []

    data = json.loads(cases_file.read_text(encoding="utf-8"))
    records = []
    for case in data.get("cases", []):
        case_id = case["case_id"]
        is_superseded = case.get("status", "").lower() == "superseded"

        # Build a single text blob for embedding
        parts = [f"Title: {case.get('title', '')}"]
        symptoms = case.get("symptoms", [])
        if symptoms:
            parts.append("Symptoms: " + "; ".join(symptoms))
        resolution = case.get("resolution", [])
        if resolution:
            parts.append("Resolution: " + "; ".join(resolution))
        if is_superseded and case.get("superseded_reason"):
            parts.append(f"NOTE: This case is superseded. {case['superseded_reason']}")

        full_text = "\n".join(parts)
        records.append({
            "source_id": case_id,
            "full_text": full_text,
            "is_superseded": is_superseded,
        })
    return records


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_document(source_id: str, full_text: str, is_superseded: bool = False) -> list[dict]:
    """
    Split a document into paragraph-level chunks.

    Splits on double-newline (markdown paragraph boundary).
    Short chunks (< 40 chars after strip) are merged with the next chunk.
    Each chunk records its character start/end offset in the original text.

    Returns:
        List of dicts: {source_id, passage, char_start, char_end, is_superseded}
    """
    # Remove YAML frontmatter block
    text = re.sub(r"^---\n.*?\n---\n", "", full_text, flags=re.DOTALL)

    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current_offset = full_text.find(text[len(text) - len(text.lstrip()):][:10])

    cursor = 0
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            cursor += len(para) + 2
            continue

        # Find char start in original text
        char_start = full_text.find(para_stripped, cursor)
        if char_start == -1:
            char_start = cursor
        char_end = char_start + len(para_stripped)

        # Skip very short chunks (headings, single lines)
        if len(para_stripped) < 40:
            cursor = char_end
            continue

        chunks.append({
            "source_id": source_id,
            "passage": para_stripped,
            "char_start": char_start,
            "char_end": char_end,
            "is_superseded": is_superseded,
        })
        cursor = char_end

    return chunks


# ── Index (built once, cached) ────────────────────────────────────────────────

class _VectorIndex:
    """In-memory cosine similarity index over chunked KB + case documents."""

    def __init__(self):
        self.chunks: list[dict] = []          # list of chunk dicts
        self.embeddings: Optional[np.ndarray] = None   # shape (N, D)
        self._built = False

    def build(self, kb_dir: Path, cases_file: Path) -> None:
        """Load documents, chunk them, embed them, cache the matrix."""
        from src.agent.models import embed

        logger.info("Building retrieval index", extra={"kb_dir": str(kb_dir)})
        t0 = time.perf_counter()

        all_chunks: list[dict] = []

        # KB documents
        for doc in _load_kb_docs(kb_dir):
            doc_chunks = _chunk_document(doc["source_id"], doc["full_text"], is_superseded=False)
            all_chunks.extend(doc_chunks)
            logger.debug(
                "Chunked KB doc",
                extra={"source_id": doc["source_id"], "chunk_count": len(doc_chunks)},
            )

        # Resolved cases (one chunk per case)
        for case in _load_resolved_cases(cases_file):
            # Resolved cases are short enough to index as a single chunk
            all_chunks.append({
                "source_id": case["source_id"],
                "passage": case["full_text"],
                "char_start": 0,
                "char_end": len(case["full_text"]),
                "is_superseded": case["is_superseded"],
            })

        if not all_chunks:
            logger.error("No chunks found — retrieval will always return empty results")
            self._built = True
            return

        texts = [c["passage"] for c in all_chunks]
        embedding_matrix = embed(texts)

        # Normalise rows for cosine similarity via dot product
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        embedding_matrix = embedding_matrix / norms

        self.chunks = all_chunks
        self.embeddings = embedding_matrix
        self._built = True

        elapsed = time.perf_counter() - t0
        logger.info(
            "Index built",
            extra={"chunk_count": len(all_chunks), "elapsed_s": round(elapsed, 2)},
        )

    def query(self, query_text: str, top_k: int = TOP_K) -> list[Evidence]:
        """
        Query the index and return the top_k most similar chunks.

        Args:
            query_text: The user query string.
            top_k:      Number of results to return.

        Returns:
            List of Evidence dicts, sorted by score descending.
        """
        if not self._built or self.embeddings is None or len(self.chunks) == 0:
            return []

        from src.agent.models import embed

        q_vec = embed([query_text])[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-9:
            return []
        q_vec = q_vec / q_norm

        # Dot product against pre-normalised embeddings = cosine similarity
        scores = (self.embeddings @ q_vec).astype(float)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[Evidence] = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[int(idx)]
            results.append(Evidence(
                source_id=chunk["source_id"],
                passage=chunk["passage"],
                score=float(scores[idx]),
                chunk_index=rank,
                is_superseded=chunk["is_superseded"],
            ))

        return results


# Module-level singleton
_INDEX = _VectorIndex()


def get_index(kb_dir: Optional[Path] = None, cases_file: Optional[Path] = None) -> _VectorIndex:
    """Return the cached index, building it on first call."""
    if not _INDEX._built:
        _INDEX.build(
            kb_dir=kb_dir or _KB_DIR,
            cases_file=cases_file or _CASES_FILE,
        )
    return _INDEX


# ── Node entry point ──────────────────────────────────────────────────────────

def retrieval_node(state: AgentState) -> dict:
    """
    Retrieval node: query the local vector index and populate retrieved_evidence.

    Always appends 'retrieval' to node_trace before any other work.
    Outputs: retrieved_evidence, retrieval_sufficient, warnings, node_trace.
    """
    t0 = time.perf_counter()
    node_name = "retrieval"
    trace = list(state.get("node_trace", []))
    trace.append(node_name)

    query = state["query"]
    warnings = list(state.get("warnings", []))

    logger.info(f"NODE {node_name} entry", extra={"node": node_name})

    index = get_index()
    results = index.query(query, top_k=TOP_K)

    # Check for superseded content in results
    superseded_in_results = [r for r in results if r["is_superseded"]]
    if superseded_in_results:
        ids = [r["source_id"] for r in superseded_in_results]
        warning_msg = (
            f"Superseded case(s) {ids} appeared in retrieval results; "
            "their guidance is excluded from the answer per assignment rules."
        )
        warnings.append(warning_msg)
        logger.warning("Superseded content in retrieval", extra={"superseded_ids": ids})

    # Determine sufficiency
    top_score = results[0]["score"] if results else 0.0
    retrieval_sufficient = top_score >= RETRIEVAL_THRESHOLD

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"NODE {node_name} exit",
        extra={
            "node": node_name,
            "result_count": len(results),
            "top_score": round(top_score, 4),
            "retrieval_sufficient": retrieval_sufficient,
            "elapsed_ms": round(elapsed, 1),
        },
    )

    return {
        "retrieved_evidence": results,
        "retrieval_sufficient": retrieval_sufficient,
        "warnings": warnings,
        "node_trace": trace,
    }
