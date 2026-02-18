import os, json, glob, pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from dateutil import parser as dtparser
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class Doc:
    doc_id: int
    chunk_id: str
    time_start: str
    time_end: str
    source_types: List[str]
    website: Optional[str]
    client_ips: List[str]
    paths: List[str]
    summary: str
    text: str
    meta: Dict[str, Any]


def _safe_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def load_jsonl(paths: List[str]) -> List[Doc]:
    docs: List[Doc] = []
    doc_id = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                docs.append(Doc(
                    doc_id=doc_id,
                    chunk_id=obj.get("chunk_id", f"doc-{doc_id}"),
                    time_start=str(obj.get("time_start", "")),
                    time_end=str(obj.get("time_end", "")),
                    source_types=_safe_list(obj.get("source_types", [])),
                    website=obj.get("website"),
                    client_ips=_safe_list(obj.get("client_ips", obj.get("src_ips", []))),
                    paths=_safe_list(obj.get("paths", [])),
                    summary=str(obj.get("summary", "")),
                    text=str(obj.get("text", "")),
                    meta=obj
                ))
                doc_id += 1
    return docs


def parse_time(ts: str):
    try:
        return dtparser.parse(ts)
    except Exception:
        return None


def build_index(
    jsonl_glob: str = "data/chunks/*.jsonl",
    out_dir: str = "rag_store",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(jsonl_glob))
    if not files:
        raise SystemExit(f"No chunk files matched: {jsonl_glob}")

    docs = load_jsonl(files)

    # We embed a compact “retrieval view” so search works even if text is noisy.
    def retrieval_text(d: Doc) -> str:
        return (
            f"chunk_id={d.chunk_id}\n"
            f"time_start={d.time_start}\n"
            f"time_end={d.time_end}\n"
            f"source_types={','.join(d.source_types)}\n"
            f"website={d.website}\n"
            f"client_ips={','.join(d.client_ips)}\n"
            f"paths={','.join(d.paths[:10])}\n"
            f"summary={d.summary}\n"
            f"text:\n{d.text[:2000]}"
        )

    corpus = [retrieval_text(d) for d in docs]

    model = SentenceTransformer(embed_model)
    embs = model.encode(corpus, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity because we normalized
    index.add(embs)

    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)

    # Precompute parsed times for quick day filtering
    times = []
    for d in docs:
        t = parse_time(d.time_start)
        times.append(t.isoformat() if t else "")
    with open(os.path.join(out_dir, "times.pkl"), "wb") as f:
        pickle.dump(times, f)

    print(f"[OK] Indexed {len(docs)} chunks from {len(files)} files into {out_dir}/")


if __name__ == "__main__":
    build_index()

