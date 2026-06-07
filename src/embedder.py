"""
embedder.py  —  Phase 1 : Step 06  (Voyage AI version)

Embeds all 100K candidates + the JD using the Voyage AI API.

Install:
    pip install voyageai tqdm numpy

Run:
    $env:VOYAGE_API_KEY = "your_key_here"
    python src/embedder_voyage.py --jd data/job_description.md


Outputs
-------
  artifacts/embeddings.npy      — float32 array  (N × 1024)
  artifacts/candidate_ids.json  — ordered list of candidate_id strings
  artifacts/jd_embedding.npy    — float32 array  (1 × 1024)
  artifacts/embed_progress.npz  — live checkpoint (every batch)
  artifacts/embedder.log        — log file
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# ── Voyage AI SDK ──────────────────────────────────────────────────────────────
try:
    import voyageai
except ImportError:
    sys.exit(
        "\n[ERROR] voyageai not installed.\n"
        "Run:  pip install voyageai\n"
    )

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("[ERROR] tqdm not installed.\nRun:  pip install tqdm")



# Configuration


VOYAGE_MODEL     = "voyage-4-large"  # 1024-dim, 200M free tokens
EMBEDDING_DIM    = 1024
BATCH_SIZE       = 128              # Voyage API max per request
RPM_TARGET       = 1800             # stay 10% under the 2000 RPM free limit
RPD_SAFETY_LIMIT = 999_999

# Retry / backoff
MAX_RETRIES      = 8
BASE_BACKOFF_S   = 5.0
MAX_BACKOFF_S    = 120.0
JITTER_FRACTION  = 0.25

# Serialization
MAX_TOKENS       = 1_500   # soft token budget per candidate (~1500 tokens)
CHARS_PER_TOKEN  = 4       # ~4 chars per token for trimming

# Paths
_HERE           = Path(__file__).resolve().parent
PROJECT_ROOT    = _HERE.parent

DATA_PATH       = PROJECT_ROOT / "data"      / "candidates.jsonl"
ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
IDS_PATH        = ARTIFACTS_DIR / "candidate_ids.json"
JD_EMB_PATH     = ARTIFACTS_DIR / "jd_embedding.npy"
CHECKPOINT_PATH = ARTIFACTS_DIR / "embed_progress.npz"
LOG_PATH        = ARTIFACTS_DIR / "embedder.log"



# Logging

def _setup_logging() -> logging.Logger:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh  = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger = logging.getLogger("embedder_voyage")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

log = _setup_logging()



# Rate limiter — token bucket, thread-safe

class RateLimiter:
    """Token-bucket limiter. acquire() blocks until a request slot is free."""

    def __init__(self, rpm: int = RPM_TARGET) -> None:
        self._rate     = rpm / 60.0
        self._capacity = float(rpm)
        self._tokens   = float(rpm)   # start full so first request is instant
        self._last     = time.monotonic()
        self._lock     = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now     = time.monotonic()
                elapsed = now - self._last
                self._last   = now
                self._tokens = min(
                    self._capacity,
                    self._tokens + elapsed * self._rate,
                )
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)


_rate_limiter = RateLimiter(RPM_TARGET)



# Voyage client

_vo_client: Optional[voyageai.Client] = None


def init_client(api_key: str) -> None:
    global _vo_client
    _vo_client = voyageai.Client(api_key=api_key)
    log.info("Voyage AI client ready.  Model: %s  (dim=%d)", VOYAGE_MODEL, EMBEDDING_DIM)


def _backoff(attempt: int) -> float:
    base   = min(BASE_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)
    jitter = base * JITTER_FRACTION * (2 * random.random() - 1)
    return max(1.0, base + jitter)



# Core embed call — retry with backoff

def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """
    Embed a list of strings using the Voyage AI API.

    Parameters
    ----------
    texts     : list of strings, max BATCH_SIZE (128) per call
    task_type : "RETRIEVAL_DOCUMENT" for candidates, "RETRIEVAL_QUERY" for JD

    Returns
    -------
    np.ndarray of shape (len(texts), EMBEDDING_DIM), dtype float32
    """
    assert _vo_client is not None, "Call init_client() before embed_texts()"

    # Voyage uses "document" / "query" (not the full task_type strings)
    input_type = "query" if task_type == "RETRIEVAL_QUERY" else "document"

    for attempt in range(MAX_RETRIES):
        _rate_limiter.acquire()
        try:
            result = _vo_client.embed(
                texts,
                model      = VOYAGE_MODEL,
                input_type = input_type,
            )
            return np.array(result.embeddings, dtype=np.float32)

        except voyageai.error.RateLimitError as e:
            wait = _backoff(attempt)
            log.warning(
                "  429 Rate limit (attempt %d/%d). Waiting %.1fs…",
                attempt + 1, MAX_RETRIES, wait,
            )
            time.sleep(wait)

        except voyageai.error.InvalidRequestError as e:
            # Bad input — don't retry, surface immediately
            log.error("  Invalid request: %s", e)
            raise

        except voyageai.error.VoyageError as e:
            wait = _backoff(attempt)
            log.error(
                "  Voyage API error (attempt %d/%d): %s. Waiting %.1fs…",
                attempt + 1, MAX_RETRIES, e, wait,
            )
            time.sleep(wait)

        except Exception as e:
            wait = _backoff(attempt)
            log.error(
                "  Unexpected error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, MAX_RETRIES, e, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Batch failed after {MAX_RETRIES} attempts on model {VOYAGE_MODEL}."
    )



# Text serialization

def serialize_candidate(candidate: dict) -> str:
    """
    Build the text blob to embed for one candidate.

    Blueprint §06 spec:
      • headline + summary   — fixed head, always included
      • career history       — most-recent first; oldest trimmed if over budget
      • top-10 skills        — sorted by endorsements + proficiency

    Total budget ≈ 1 500 tokens (6 000 chars).
    """
    p      = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    budget = MAX_TOKENS * CHARS_PER_TOKEN  # 6 000 chars

    # Head: headline + summary
    head = "\n".join(filter(None, [
        p.get("headline", "").strip(),
        p.get("summary", "").strip(),
    ]))

    # Top-10 skills by endorsements desc, then proficiency rank
    _prof = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    top_skills = sorted(
        skills,
        key=lambda s: (s.get("endorsements", 0), _prof.get(s.get("proficiency", ""), 0)),
        reverse=True,
    )[:10]
    skills_str = "Skills: " + ", ".join(
        f"{s['name']} ({s.get('proficiency', '')})" for s in top_skills
    )

    # Career: most-recent first
    sorted_career = sorted(
        career,
        key=lambda r: r.get("start_date", "0000-00-00") or "0000-00-00",
        reverse=True,
    )
    role_strings = [
        f"{r.get('title', '')} at {r.get('company', '')} "
        f"({r.get('duration_months', 0)}mo): {(r.get('description') or '').strip()}"
        for r in sorted_career
    ]

    # Trim oldest roles if over budget
    overhead      = len(head) + len(skills_str) + 10
    career_budget = budget - overhead
    career_block  = "\n".join(role_strings)

    if len(career_block) > career_budget:
        kept, used = [], 0
        for rs in role_strings:
            need = len(rs) + (1 if kept else 0)
            if used + need > career_budget:
                remaining = career_budget - used - 1
                if remaining > 80:
                    sep = rs.find(": ")
                    if sep != -1:
                        prefix = rs[: sep + 2]
                        kept.append(
                            prefix + rs[sep + 2 : sep + 2 + (remaining - len(prefix))]
                        )
                break
            kept.append(rs)
            used += need
        career_block = "\n".join(kept)

    return "\n\n".join(filter(None, [head, career_block, skills_str]))


def serialize_jd(jd_text: str) -> str:
    """Truncate only if text exceeds Voyage's practical context limit."""
    limit = 32_000
    if len(jd_text) > limit:
        log.warning("JD text truncated from %d to %d chars.", len(jd_text), limit)
        return jd_text[:limit]
    return jd_text



# Checkpoint helpers

def _save_checkpoint(embeddings_list: list[np.ndarray], ids_list: list[str]) -> None:
    """Atomically write progress (tmp file → rename, never corrupts on crash)."""
    if not embeddings_list:
        return
    arr = np.vstack(embeddings_list).astype(np.float32)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, embeddings=arr, ids=np.array(ids_list, dtype=object))
    tmp.replace(CHECKPOINT_PATH)


def _load_checkpoint() -> tuple[Optional[np.ndarray], list[str]]:
    """Load previous checkpoint. Returns (array_or_None, ids_list)."""
    if not CHECKPOINT_PATH.exists():
        return None, []
    try:
        data = np.load(CHECKPOINT_PATH, allow_pickle=True)
        emb  = data["embeddings"].astype(np.float32)
        ids  = [str(x) for x in data["ids"].tolist()]
        log.info("  ♻  Checkpoint loaded — %d candidates already embedded.", len(ids))
        return emb, ids
    except Exception as e:
        log.warning("  Could not read checkpoint (%s) — starting fresh.", e)
        return None, []



# Candidate streaming

def _stream_candidates(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("Skipping malformed JSON line: %s", e)


def _count_lines(path: Path) -> int:
    count = 0
    with open(path, "rb") as fh:
        for _ in fh:
            count += 1
    return count



# JD embedding

def embed_jd(jd_text: str) -> np.ndarray:
    """
    Embed the JD with input_type="query" (Voyage equivalent of RETRIEVAL_QUERY).
    Saves artifacts/jd_embedding_voyage.npy and returns shape (1, 1024).
    """
    log.info("Embedding JD (input_type=query, model=%s)…", VOYAGE_MODEL)
    text = serialize_jd(jd_text)
    emb  = embed_texts([text], task_type="RETRIEVAL_QUERY")
    np.save(JD_EMB_PATH, emb)
    log.info("  ✓  JD embedding saved → %s  (shape %s)", JD_EMB_PATH, emb.shape)
    return emb



# Main candidate embedding loop

def embed_all_candidates(data_path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Stream candidates.jsonl, embed in batches, checkpoint after every batch.

    Voyage has no daily RPD cap, so this runs straight through all 100K
    candidates in one session (~10 minutes at 1800 RPM with batch=128).

    If interrupted, re-run — already-embedded IDs are skipped automatically.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if resuming
    prev_emb, done_ids = _load_checkpoint()
    done_set           = set(done_ids)

    all_emb: list[np.ndarray] = (
        [prev_emb[i : i + 1] for i in range(len(prev_emb))]
        if prev_emb is not None and len(prev_emb) > 0
        else []
    )
    all_ids: list[str] = list(done_ids)

    # Count lines for progress bar
    log.info("Counting lines in %s …", data_path)
    total_lines = _count_lines(data_path)
    remaining   = total_lines - len(done_set)
    log.info(
        "  Dataset: %d total | Already embedded: %d | Remaining: %d",
        total_lines, len(done_set), remaining,
    )

    if remaining == 0:
        log.info("  All candidates already embedded — nothing to do.")
        final = (
            np.vstack(all_emb).astype(np.float32) if all_emb
            else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        )
        return final, all_ids

    # ~782 requests for 100K candidates at batch=128
    expected_requests = math.ceil(remaining / BATCH_SIZE)
    log.info(
        "  Model: %s | Batch: %d | ~%d requests needed | ETA: ~%.0f min",
        VOYAGE_MODEL, BATCH_SIZE, expected_requests,
        expected_requests / RPM_TARGET * 60,
    )

    session_requests = 0
    session_embedded = 0
    batch_texts: list[str] = []
    batch_ids:   list[str] = []
    t_start = time.monotonic()

    with tqdm(
        total=remaining,
        desc="Embedding candidates",
        unit="cand",
        dynamic_ncols=True,
        smoothing=0.05,
        colour="cyan",
    ) as pbar:

        def _flush() -> None:
            nonlocal session_requests, session_embedded

            if not batch_texts:
                return

            emb_block = embed_texts(batch_texts, task_type="RETRIEVAL_DOCUMENT")
            session_requests += 1
            session_embedded += len(batch_texts)

            for i, cid in enumerate(batch_ids):
                all_emb.append(emb_block[i : i + 1])
                all_ids.append(cid)

            pbar.update(len(batch_texts))

            # Checkpoint after every batch — safe to Ctrl-C anytime
            _save_checkpoint(all_emb, all_ids)

            batch_texts.clear()
            batch_ids.clear()

            # Progress log every 100 batches
            if session_requests % 100 == 0:
                elapsed  = time.monotonic() - t_start
                rate     = session_embedded / max(elapsed, 1)
                left     = remaining - session_embedded
                eta_min  = (left / max(rate, 1)) / 60
                log.info(
                    "  ◆ %d requests | %d/%d embedded | %.0f cand/s | ETA ~%.1f min",
                    session_requests, len(all_ids), total_lines, rate, eta_min,
                )

        for candidate in _stream_candidates(data_path):
            cid = candidate.get("candidate_id")
            if not cid or cid in done_set:
                continue

            batch_texts.append(serialize_candidate(candidate))
            batch_ids.append(cid)

            if len(batch_texts) >= BATCH_SIZE:
                _flush()

        _flush()  # final partial batch

    elapsed_total = time.monotonic() - t_start
    log.info(
        "✅  Done — %d requests | %d candidates | %.1f min | model: %s",
        session_requests, session_embedded, elapsed_total / 60, VOYAGE_MODEL,
    )

    final = (
        np.vstack(all_emb).astype(np.float32) if all_emb
        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    )
    return final, all_ids



# Spot-check

def spot_check(
    embeddings:    np.ndarray,
    candidate_ids: list[str],
    jd_embedding:  np.ndarray,
    data_path:     Path,
    n: int = 5,
) -> None:
    """
    Rank all embedded candidates by cosine similarity to the JD, print top-n.
    Good candidates (ML/AI engineers with retrieval/ranking experience)
    should dominate the list.
    """
    log.info("=== Spot-check: top-%d by cosine similarity to JD ===", n)

    norm_emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    norm_jd  = jd_embedding.squeeze() / (np.linalg.norm(jd_embedding) + 1e-9)
    sims     = norm_emb @ norm_jd
    top_idx  = np.argsort(sims)[::-1][:n]

    top_cids = {str(candidate_ids[i]) for i in top_idx}
    top_data: dict[str, dict] = {}
    for cand in _stream_candidates(data_path):
        cid = cand.get("candidate_id")
        if cid in top_cids:
            top_data[cid] = cand
        if len(top_data) == len(top_cids):
            break

    for rank, idx in enumerate(top_idx, 1):
        cid  = str(candidate_ids[idx])
        sim  = float(sims[idx])
        cand = top_data.get(cid, {})
        p    = cand.get("profile", {})
        log.info(
            "  #%d  sim=%.4f  %-15s  %s @ %s  (%s, %.1f YOE)",
            rank, sim, cid,
            p.get("current_title", "?"),
            p.get("current_company", "?"),
            p.get("current_industry", "?"),
            p.get("years_of_experience", 0),
        )
    log.info("  (Top results should be ML/AI engineers, not marketers or accountants)")



# JD loader

def _load_jd(jd_arg: Optional[str]) -> str:
    """Load JD from --jd path, search common locations, or fall back to inline."""
    if jd_arg:
        p = Path(jd_arg)
        if not p.exists():
            sys.exit(f"[ERROR] JD file not found: {p}")
        log.info("Loading JD from %s", p)
        return p.read_text(encoding="utf-8")

    for candidate_path in [
        PROJECT_ROOT / "data" / "job_description.md",
        PROJECT_ROOT / "data" / "job_description.txt",
        PROJECT_ROOT / "job_description.md",
        PROJECT_ROOT / "jd.txt",
    ]:
        if candidate_path.exists():
            log.info("Loading JD from %s", candidate_path)
            return candidate_path.read_text(encoding="utf-8")

    log.warning("No JD file found — using inline fallback. Pass --jd <path> for best results.")
    return _INLINE_JD


_INLINE_JD = """\
Senior AI Engineer (Founding Team) — Redrob AI

Hard requirements:
- Production experience with embedding-based retrieval (sentence-transformers, BGE, E5, OpenAI embeddings).
- Vector database operational experience (Pinecone, Milvus, Qdrant, FAISS, OpenSearch, Elasticsearch).
- Strong Python, production-grade code quality.
- Ranking evaluation frameworks: NDCG, MRR, MAP, offline-to-online correlation, A/B testing.

Strong signals:
- 6-8 years total, 4-5 years applied ML/AI at product companies (not IT services).
- Shipped end-to-end ranking, search, or recommendation system at scale.
- Located in or willing to relocate to Pune or Noida.

Disqualifiers:
- Entire career at IT services (TCS, Infosys, Wipro) with no product company.
- LLM experience limited to LangChain/OpenAI API stitching, no pre-LLM ML history.
- Computer Vision or Speech as primary domain, no NLP/IR depth.
"""



# Entry point

import math

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 · Step 06 — embed all candidates + JD using Voyage AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("VOYAGE_API_KEY"),
        help="Voyage AI API key (default: $VOYAGE_API_KEY env var)",
    )
    ap.add_argument(
        "--data",
        default=str(DATA_PATH),
        help=f"Path to candidates.jsonl  [default: {DATA_PATH}]",
    )
    ap.add_argument(
        "--jd",
        default=None,
        help="Path to job description file (.md / .txt)",
    )
    ap.add_argument(
        "--skip-jd",
        action="store_true",
        help="Skip JD embedding (if jd_embedding_voyage.npy already exists)",
    )
    ap.add_argument(
        "--skip-candidates",
        action="store_true",
        help="Skip candidate embedding (e.g. only re-embed JD)",
    )
    ap.add_argument(
        "--spot-check-only",
        action="store_true",
        help="Load existing embeddings and print top-5; no API calls",
    )
    args = ap.parse_args()

    # API key check
    if not args.spot_check_only:
        if not args.api_key:
            sys.exit(
                "[ERROR] No Voyage API key found.\n"
                "Set $VOYAGE_API_KEY or pass --api-key <your_key>.\n"
                "Get a key at: https://dash.voyageai.com"
            )
        init_client(args.api_key)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)

    if not args.spot_check_only and not data_path.exists():
        sys.exit(f"[ERROR] Dataset not found: {data_path}")

    # Spot-check mode
    if args.spot_check_only:
        for p, label in [
            (EMBEDDINGS_PATH, "embeddings_voyage.npy"),
            (IDS_PATH,        "candidate_ids_voyage.json"),
            (JD_EMB_PATH,     "jd_embedding_voyage.npy"),
        ]:
            if not p.exists():
                sys.exit(f"[ERROR] {label} not found. Run full pipeline first.")
        embeddings    = np.load(EMBEDDINGS_PATH)
        candidate_ids = json.loads(IDS_PATH.read_text())
        jd_embedding  = np.load(JD_EMB_PATH)
        spot_check(embeddings, candidate_ids, jd_embedding, data_path)
        return

    # Load JD text
    jd_text = _load_jd(args.jd)

    # Embed JD
    jd_embedding: Optional[np.ndarray] = None

    if not args.skip_jd:
        if JD_EMB_PATH.exists():
            log.info("JD embedding already exists at %s — skipping.", JD_EMB_PATH)
            jd_embedding = np.load(JD_EMB_PATH)
        else:
            jd_embedding = embed_jd(jd_text)
    else:
        if JD_EMB_PATH.exists():
            jd_embedding = np.load(JD_EMB_PATH)
            log.info("Loaded existing JD embedding from %s.", JD_EMB_PATH)
        else:
            log.warning("--skip-jd set but jd_embedding_voyage.npy not found yet.")

    # Embed candidates
    if not args.skip_candidates:
        t0 = time.monotonic()
        embeddings, candidate_ids = embed_all_candidates(data_path)
        elapsed = time.monotonic() - t0

        # Save final outputs
        np.save(EMBEDDINGS_PATH, embeddings)
        IDS_PATH.write_text(json.dumps(candidate_ids, ensure_ascii=False, indent=0))

        log.info(
            "✅  Saved → %s  (shape %s | %.1f MB)",
            EMBEDDINGS_PATH, embeddings.shape, embeddings.nbytes / 1e6,
        )
        log.info("✅  Saved → %s  (%d IDs)", IDS_PATH, len(candidate_ids))
        log.info("    Total wall time: %.1f min", elapsed / 60)

        # Remove checkpoint only when fully complete
        total_lines = _count_lines(data_path)
        if len(candidate_ids) >= total_lines:
            if CHECKPOINT_PATH.exists():
                CHECKPOINT_PATH.unlink()
                log.info("    Checkpoint removed (all %d candidates embedded).", total_lines)
        else:
            log.info(
                "    %d candidates still unembedded — checkpoint kept for resume.",
                total_lines - len(candidate_ids),
            )

        # Spot-check
        if jd_embedding is not None and len(candidate_ids) > 0:
            spot_check(embeddings, candidate_ids, jd_embedding, data_path)

    elif jd_embedding is not None and EMBEDDINGS_PATH.exists() and IDS_PATH.exists():
        embeddings    = np.load(EMBEDDINGS_PATH)
        candidate_ids = json.loads(IDS_PATH.read_text())
        spot_check(embeddings, candidate_ids, jd_embedding, data_path)


if __name__ == "__main__":
    main()