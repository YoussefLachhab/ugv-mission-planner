from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _split_into_chunks(md_text: str, max_chunk_len: int = 600) -> list[str]:
    lines = md_text.splitlines()
    chunks: list[str] = []
    buf: list[str] = []
    for ln in lines:
        if ln.startswith("#") and buf:
            chunk = "\n".join(buf).strip()
            if chunk:
                chunks.append(chunk)
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        chunk = "\n".join(buf).strip()
        if chunk:
            chunks.append(chunk)
    final: list[str] = []
    for ch in chunks:
        if len(ch) <= max_chunk_len:
            final.append(ch)
        else:
            parts = [p.strip() for p in ch.split("\n\n") if p.strip()]
            final.extend(parts or [ch[:max_chunk_len]])
    return final


@dataclass
class Retrieved:
    chunk: str
    score: float


class PolicyRAG:
    def __init__(self, policy_path: Path) -> None:
        self.policy_path = policy_path
        text = policy_path.read_text(encoding="utf-8")
        self.chunks = _split_into_chunks(text)
        self._build_index(self.chunks)

    def _build_index(self, chunks: list[str]) -> None:
        docs_tokens = [_tokenize(c) for c in chunks]
        vocab: dict[str, int] = {}
        for toks in docs_tokens:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab

        tf = np.zeros((len(chunks), len(vocab)), dtype=np.float32)
        df = np.zeros((len(vocab),), dtype=np.float32)
        for i, toks in enumerate(docs_tokens):
            counts: dict[int, int] = {}
            for t in toks:
                j = vocab[t]
                counts[j] = counts.get(j, 0) + 1
            if counts:
                row_sum = float(sum(counts.values()))
                for j, cnt in counts.items():
                    tf[i, j] = cnt / row_sum
                for j in counts:
                    df[j] += 1.0

        n = float(len(chunks))
        idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
        mat = tf * idf[None, :]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.mat = mat / norms
        self.idf = idf

    def retrieve(self, query: str, k: int = 3) -> list[Retrieved]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        q_vec = np.zeros((len(self.vocab),), dtype=np.float32)
        counts: dict[int, int] = {}
        for t in q_tokens:
            j = self.vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            return []
        q_sum = float(sum(counts.values()))
        for j, cnt in counts.items():
            q_vec[j] = (cnt / q_sum) * self.idf[j]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm
        sims = self.mat @ q_vec
        idx = np.argsort(-sims)[:k]
        return [Retrieved(self.chunks[i], float(sims[i])) for i in idx]

    def _constraints_to_dict(self, cons: Any) -> dict[str, Any]:
        if cons is None:
            return {}
        md = getattr(cons, "model_dump", None)
        if callable(md):
            return cons.model_dump()
        jd = getattr(cons, "dict", None)
        if callable(jd):
            return cons.dict()
        if isinstance(cons, dict):
            return cons
        out = {}
        for k in ("max_speed_mps", "max_speed", "avoid", "avoid_zones", "geofence"):
            if hasattr(cons, k):
                out[k] = getattr(cons, k)
        return out

    def explain_plan(self, plan: Any, k_per_topic: int = 2):
        cons_obj = getattr(plan, "constraints", None) if not isinstance(plan, dict) else plan.get("constraints")
        cons = self._constraints_to_dict(cons_obj)
        max_speed = cons.get("max_speed_mps") or cons.get("max_speed")

        topics = []
        if max_speed is not None:
            topics.append(("max_speed", f"maximum speed {max_speed} m/s limit safety near geofence policy"))
        if cons.get("avoid_zones") or cons.get("geofence") or cons.get("avoid"):
            topics.append(("geofence", "avoid zones geofence hard no-go min clearance inflation cells safety"))
        topics.append(("determinism", "determinism reproducibility artifacts gif compliance metrics"))

        out = []
        for topic, q in topics:
            for r in self.retrieve(q, k=k_per_topic):
                out.append({"topic": topic, "snippet": r.chunk.strip(), "score": round(r.score, 4)})
        return out

    def extract_policy_limits(self) -> dict[str, float]:
        text = self.policy_path.read_text(encoding="utf-8")
        # Global cap = smallest explicit m/s value found (conservative)
        speeds = [float(x) for x in re.findall(r"(\d+\.?\d*)\s*m/s", text)]
        max_speed = min(speeds) if speeds else None

        # Near-geofence rule pattern: "within X cells ... Y m/s"
        m_near = re.search(
            r"within\s+(\d+(?:\.\d+)?)\s*cells?.*?(\d+(?:\.\d+)?)\s*m/s",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        near_cells = float(m_near.group(1)) if m_near else None
        near_speed = float(m_near.group(2)) if m_near else None

        # Min clearance (if specified generically)
        m = re.search(r"(min(?:imum)?\s+clearance).*?(\d+\.?\d*)\s*cells?", text, re.IGNORECASE | re.DOTALL)
        min_clearance_cells = float(m.group(2)) if m else None

        out: dict[str, float] = {}
        if max_speed is not None:
            out["max_speed_mps"] = max_speed
        if min_clearance_cells is not None:
            out["min_clearance_cells"] = min_clearance_cells
        if near_cells is not None and near_speed is not None:
            out["near_geofence_cells"] = near_cells
            out["near_geofence_speed_mps"] = near_speed
        return out
