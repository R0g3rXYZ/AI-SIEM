import os, json, glob, pickle, re
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
        with open(p, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                docs.append({
                    "doc_id": doc_id,
                    "chunk_id": obj.get("chunk_id", f"doc-{doc_id}"),
                    "time_start": str(obj.get("time_start", "")),
                    "time_end": str(obj.get("time_end", "")),
                    "source_types": _safe_list(obj.get("source_types", [])),
                    "website": obj.get("website"),
                    "client_ips": _safe_list(obj.get("client_ips", obj.get("src_ips", []))),
                    "paths": _safe_list(obj.get("paths", [])),
                    "summary": str(obj.get("summary", "")),
                    "text": str(obj.get("text", "")),
                    "meta": obj,   # keep full original
                })
                doc_id += 1
    return docs


def parse_time(ts: str):
    try:
        return dtparser.parse(ts)
    except Exception:
        return None


def _contains_any(hay: str, needles: List[str]) -> bool:
    return any(n in hay for n in needles)


def infer_attack_type(doc: Dict[str, Any]) -> str:
    """
    Lightweight attack-type classifier used at indexing time so query workflows
    can use precomputed labels and daily counts.
    """
    summary = str(doc.get("summary", "") or "").lower()
    text = str(doc.get("text", "") or "").lower()
    paths = [str(p).lower() for p in (doc.get("paths", []) or [])]
    source_types = [str(s).lower() for s in (doc.get("source_types", []) or [])]
    meta = doc.get("meta", {}) or {}
    error_type_counts = (meta.get("meta", {}) or {}).get("error_type_counts", {})
    if not isinstance(error_type_counts, dict):
        error_type_counts = {}

    blob = "\n".join([summary, text, " ".join(paths), " ".join(source_types), json.dumps(error_type_counts).lower()])

    if _contains_any(blob, ["%0d%0a", "http splitting", "crlf", "\\r\\n", "\r\n", "http/1.1%0d%0a"]):
        return "protocol_smuggling_or_crlf"
    if _contains_any(blob, ["union select", "or 1=1", "sleep(", "benchmark(", "information_schema", "sql syntax"]):
        return "sql_injection"
    if _contains_any(blob, ["../", "%2e%2e%2f", "/..\\", "path traversal", "directory traversal"]):
        return "path_traversal"
    if _contains_any(blob, ["<script", "onerror=", "onload=", "javascript:"]):
        return "xss"
    if _contains_any(blob, ["${jndi", "cmd=", "powershell", "wget ", "curl ", "shell", "rce"]):
        return "rce_attempt"
    if _contains_any(blob, ["missing a host header", "host header", "x-forwarded-host", "header injection"]):
        return "http_header_injection"
    if _contains_any(blob, ["auth bypass", "login bypass", "authorization=", "digest username", "basic "]):
        return "auth_bypass_attempt"
    if _contains_any(blob, ["brute force", "bruteforce", "password spray", "invalid password", "too many login"]):
        return "auth_bruteforce_or_guessing"
    if _contains_any(blob, ["wp-admin", "wordpress", "drupal", "joomla", "/xmlrpc.php"]):
        return "cms_probe"
    if _contains_any(blob, ["nessus", "nmap", "scan-like", "hnap1", "upstream_connect_failed"]):
        return "scanner/probing"

    if len(paths) >= 20:
        return "scanner/probing"
    return "unknown"


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
            f"chunk_id={d['chunk_id']}\n"
            f"time_start={d['time_start']}\n"
            f"time_end={d['time_end']}\n"
            f"source_types={','.join(d['source_types'])}\n"
            f"website={d['website']}\n"
            f"client_ips={','.join(d['client_ips'])}\n"
            f"paths={','.join(d['paths'][:10])}\n"
            f"summary={d['summary']}\n"
            f"text:\n{d['text'][:2000]}"
        )

    corpus = [retrieval_text(d) for d in docs]

    model = SentenceTransformer(embed_model)
    embs = model.encode(corpus, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity because we normalized
    index.add(embs)

    # Precompute parsed times and attack-type commonness stats.
    times = []
    attack_type_cache: Dict[str, str] = {}
    per_day: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, int] = {}
    for d in docs:
        t = parse_time(d['time_start'])
        times.append(t.isoformat() if t else "")

        attack_type = infer_attack_type(d)
        d["attack_type"] = attack_type
        attack_type_cache[d["chunk_id"]] = attack_type
        print(f"[CLASSIFY] {d['chunk_id']} -> {attack_type}")

        if t:
            day = t.strftime("%Y-%m-%d")
            if day not in per_day:
                per_day[day] = {}
            per_day[day][attack_type] = per_day[day].get(attack_type, 0) + 1
        totals[attack_type] = totals.get(attack_type, 0) + 1

    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)

    with open(os.path.join(out_dir, "times.pkl"), "wb") as f:
        pickle.dump(times, f)
    with open(os.path.join(out_dir, "attack_type_cache.json"), "w", encoding="utf-8") as f:
        json.dump(attack_type_cache, f, indent=2)
    with open(os.path.join(out_dir, "attack_type_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"per_day": per_day, "totals": totals}, f, indent=2)

    print(f"[OK] Indexed {len(docs)} chunks from {len(files)} files into {out_dir}/")


if __name__ == "__main__":
    build_index()
