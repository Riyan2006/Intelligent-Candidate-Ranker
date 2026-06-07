"""
rank.py  —  Phase 2 · Step 09
==============================
The actual submission script. Produces the final ranked CSV.

Constraints (from submission_spec.docx):
  - Must run in ≤ 5 minutes wall-clock on CPU
  - ≤ 16 GB RAM
  - NO network access — no API calls of any kind
  - NO GPU
  - ≤ 5 GB intermediate disk

How it works:
  1. Load all pre-computed artifacts from disk (Steps 3–8)
  2. Compute cosine similarity (numpy) — ~2 sec for 100K vectors
  3. 4-signal weighted fusion:
       final_score = (0.35 × embedding_sim)
                   + (0.30 × llm_score_norm)
                   + (0.20 × must_have_feature_score)
                   + (0.15 × behavioral_score)
  4. Apply honeypot multiplier  (×0.01)
  5. Apply disqualifier multiplier (×0.10 for consulting-only etc)
  6. Sort descending → top 100 → assign ranks 1–100
  7. Attach reasoning strings from cache (template fallback for any missing)
  8. Validate: 100 rows, ranks 1-100 each once, scores non-increasing,
     all IDs exist in source file, no duplicate IDs
  9. Write UTF-8 CSV

Usage:
  python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

  # Dry-run (skip CSV write, just print top-20):
  python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv --dry-run

  # Override artifacts directory:
  python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv --artifacts ./artifacts

Artifacts required (all in --artifacts dir):
  embeddings.npy          — (100K, dim) float32, one row per candidate
  candidate_ids.json      — list[str] of 100K candidate IDs in embedding row order
  jd_embedding.npy        — (dim,) float32, JD embedding vector
  career_features.parquet — output of feature_engineer.py
  behavioral_scores.parquet — output of behavioral_scorer.py
  llm_scores.json         — output of llm_scorer.py
  reasoning_cache.json    — output of reasoning_generator.py
  honeypot_ids.txt        — output of honeypot_detector.py
"""

import os
import sys
import csv
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd


# Fusion weights

W_EMBEDDING    = 0.35   # semantic similarity — captures meaning beyond keywords
W_LLM          = 0.30   # nuanced reasoning — career, red flags, shipping evidence
W_FEATURES     = 0.20   # structural JD-aware features — skill match, company type
W_BEHAVIORAL   = 0.15   # platform signals — availability, responsiveness, github

assert abs(W_EMBEDDING + W_LLM + W_FEATURES + W_BEHAVIORAL - 1.0) < 1e-9

# Penalty multipliers (applied post-fusion, multiplicative)
HONEYPOT_MULT      = 0.01   # mathematically impossible profile
CONSULTING_MULT    = 0.10   # entire career at TCS/Infosys/Wipro/Accenture/etc
RED_FLAG_MULT      = 0.20   # LLM assigned red_flag_check < 3
OUT_OF_DOMAIN_MULT = 0.40   # CV/speech only with no NLP signal
UNAVAILABLE_MULT   = 0.50   # recruiter response rate ≤ 5% AND inactive

# Tie-breaking: when scores are equal, sort by candidate_id ascending (deterministic)


# Logging — minimal, goes to stdout only (no files during ranking)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("rank")



# Template fallback reasoning (zero API calls, zero hallucination)
# Used for any candidate in top-100 that isn't in reasoning_cache

def _template_reasoning(row: dict, rank: int) -> str:
    """
    Construct a specific, fact-grounded reasoning string from pre-computed
    feature columns. No LLM, no hallucination.
    """
    cid   = row.get("candidate_id", "")
    title = row.get("_title", "candidate")
    co    = row.get("_company", "their current employer")
    yoe   = row.get("_yoe", 0)
    feat  = float(row.get("must_have_feature_score", 0))
    beh   = float(row.get("behavioral_score", 0))

    # Strength signal
    if feat >= 0.6:
        strength = f"strong JD skill alignment ({feat:.2f} feature score)"
    elif feat >= 0.4:
        strength = f"moderate JD skill alignment ({feat:.2f} feature score)"
    else:
        strength = f"limited direct JD skill overlap ({feat:.2f} feature score)"

    # Concern signal
    concerns = []
    if row.get("flag_consulting_only"):
        concerns.append("consulting-only career history")
    if row.get("flag_out_of_domain"):
        concerns.append("out-of-domain primary skills (CV/speech)")
    if row.get("flag_unavailable"):
        concerns.append("low platform responsiveness")
    if row.get("flag_title_chaser"):
        concerns.append("title-chasing career pattern")
    if beh < 0.3:
        concerns.append("low behavioral engagement signals")

    concern_str = concerns[0] if concerns else "requires further evaluation against JD depth requirements"

    return (
        f"{title} at {co} with {yoe:.1f} years of experience; "
        f"{strength}, ranked #{rank} based on combined embedding, LLM, and feature signals. "
        f"Primary concern: {concern_str}."
    )



# Artifact loaders


def load_embeddings(artifacts: Path) -> tuple[np.ndarray, list[str]]:
    """Load (100K, dim) embedding matrix and matching candidate ID list."""
    log.info("Loading embeddings…")
    embs     = np.load(artifacts / "embeddings.npy").astype(np.float32)
    cand_ids = json.loads((artifacts / "candidate_ids.json").read_text())
    assert len(cand_ids) == len(embs), (
        f"ID/embedding count mismatch: {len(cand_ids)} IDs vs {len(embs)} rows"
    )
    log.info("  Embeddings: %s  (%.1f MB)", embs.shape, embs.nbytes / 1e6)
    return embs, cand_ids


def load_jd_embedding(artifacts: Path) -> np.ndarray:
    jd_emb = np.load(artifacts / "jd_embedding.npy").astype(np.float32).flatten()
    log.info("  JD embedding dim: %d", jd_emb.shape[0])
    return jd_emb


def load_career_features(artifacts: Path) -> pd.DataFrame:
    log.info("Loading career features…")
    df = pd.read_parquet(artifacts / "career_features.parquet")
    df["candidate_id"] = df["candidate_id"].astype(str)
    log.info("  Career features: %d rows, %d cols", len(df), len(df.columns))
    return df.set_index("candidate_id")


def load_behavioral_scores(artifacts: Path) -> pd.DataFrame:
    log.info("Loading behavioral scores…")
    df = pd.read_parquet(artifacts / "behavioral_scores.parquet")
    df["candidate_id"] = df["candidate_id"].astype(str)

    # Normalise: ensure 'behavioral_score' column exists
    if "behavioral_score" not in df.columns:
        # Try to find the main score column
        score_cols = [c for c in df.columns if "score" in c.lower() and c != "candidate_id"]
        if score_cols:
            df = df.rename(columns={score_cols[0]: "behavioral_score"})
            log.warning("  Renamed '%s' → 'behavioral_score'", score_cols[0])
        else:
            log.warning("  No score column found in behavioral_scores.parquet — defaulting to 0.5")
            df["behavioral_score"] = 0.5

    # Clip to [0, 1]
    df["behavioral_score"] = df["behavioral_score"].clip(0.0, 1.0)
    log.info("  Behavioral scores: %d rows", len(df))
    return df.set_index("candidate_id")


def load_llm_scores(artifacts: Path) -> dict[str, dict]:
    """Returns {candidate_id: {llm_rubric_score, red_flag_penalty, rubric_scores}}."""
    log.info("Loading LLM scores…")
    raw = json.loads((artifacts / "llm_scores.json").read_text())
    result = {}
    for entry in raw:
        cid = entry.get("candidate_id", "")
        result[cid] = {
            "llm_rubric_score": float(entry.get("llm_rubric_score", 0.0)),
            "red_flag_penalty": bool(entry.get("red_flag_penalty", False)),
            "in_top_500"      : bool(entry.get("in_top_500", False)),
        }
    n_scored = sum(1 for v in result.values() if v["in_top_500"])
    log.info("  LLM scores: %d total entries, %d top-500 scored", len(result), n_scored)
    return result


def load_reasoning_cache(artifacts: Path) -> dict[str, str]:
    path = artifacts / "reasoning_cache.json"
    if not path.exists():
        log.warning("  reasoning_cache.json not found — will use template fallback for all")
        return {}
    cache = json.loads(path.read_text())
    log.info("  Reasoning cache: %d entries", len(cache))
    return cache


def load_honeypot_ids(artifacts: Path) -> set[str]:
    path = artifacts / "honeypot_ids.txt"
    if not path.exists():
        log.warning("  honeypot_ids.txt not found — no honeypot suppression")
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split("|")
            if parts:
                cid = parts[0].strip()
                if cid.startswith("CAND_"):
                    ids.add(cid)
    log.info("  Honeypot IDs: %d loaded", len(ids))
    return ids


def stream_candidate_metadata(candidates_path: Path, target_ids: set[str]) -> dict[str, dict]:
    """
    Stream candidates.jsonl once to extract lightweight metadata
    (title, company, yoe) for template reasoning fallback.
    Only fetches the target_ids set.
    """
    log.info("Streaming candidate metadata for top-100…")
    meta = {}
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c   = json.loads(line)
                cid = c.get("candidate_id", "")
                if cid in target_ids:
                    p = c.get("profile", {})
                    meta[cid] = {
                        "_title"  : p.get("current_title", ""),
                        "_company": p.get("current_company", ""),
                        "_yoe"    : p.get("years_of_experience", 0),
                    }
                if len(meta) == len(target_ids):
                    break  # found all, stop early
            except json.JSONDecodeError:
                continue
    log.info("  Loaded metadata for %d / %d candidates", len(meta), len(target_ids))
    return meta



# Cosine similarity — pure numpy, no network


def compute_cosine_similarity(
    embs: np.ndarray,
    jd_emb: np.ndarray,
) -> np.ndarray:
    """L2-normalise and dot-product. Returns (100K,) similarity scores [0, 1]."""
    log.info("Computing cosine similarity for %d candidates…", len(embs))
    t0 = time.monotonic()

    norm_embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    norm_jd   = jd_emb / (np.linalg.norm(jd_emb) + 1e-9)
    sims      = norm_embs @ norm_jd   # (100K,)

    # Clip to [0, 1] — cosine can be slightly negative for unrelated content
    sims = np.clip(sims, 0.0, 1.0)

    elapsed = time.monotonic() - t0
    log.info(
        "  Cosine similarity done in %.2fs — range [%.4f, %.4f]",
        elapsed, float(sims.min()), float(sims.max()),
    )
    return sims.astype(np.float32)



# 4-signal fusion


def fuse_scores(
    cand_ids       : list[str],
    embedding_sims : np.ndarray,
    career_feats   : pd.DataFrame,
    behavioral_df  : pd.DataFrame,
    llm_scores     : dict[str, dict],
    honeypot_ids   : set[str],
) -> pd.DataFrame:
    """
    Combine all four signals into a final_score per candidate.

    Returns a DataFrame with columns:
      candidate_id, embedding_sim, llm_score_norm, must_have_feature_score,
      behavioral_score, raw_fusion_score, final_score,
      applied_honeypot_penalty, applied_red_flag_penalty,
      applied_consulting_penalty, applied_ood_penalty, applied_unavail_penalty
      + all career feature flag columns for template reasoning
    """
    log.info("Running 4-signal fusion for %d candidates…", len(cand_ids))
    t0 = time.monotonic()

    n = len(cand_ids)

    # Build arrays aligned to cand_ids order
    emb_arr   = embedding_sims.astype(np.float64)   # already (n,) aligned

    llm_arr   = np.zeros(n, dtype=np.float64)
    rf_flag   = np.zeros(n, dtype=bool)
    for i, cid in enumerate(cand_ids):
        entry = llm_scores.get(cid, {})
        llm_arr[i] = entry.get("llm_rubric_score", 0.0)
        rf_flag[i] = entry.get("red_flag_penalty", False)

    feat_arr  = np.full(n, 0.3, dtype=np.float64)   # default: average unknown
    consulting_flag = np.zeros(n, dtype=bool)
    ood_flag        = np.zeros(n, dtype=bool)
    unavail_flag    = np.zeros(n, dtype=bool)
    title_chaser_flag = np.zeros(n, dtype=bool)

    # Extra columns for template reasoning
    feat_cols_to_carry = [
        "flag_consulting_only", "flag_out_of_domain", "flag_unavailable",
        "flag_title_chaser", "flag_inactive_coder", "flag_llm_only_wrapper",
        "must_have_feature_score",
    ]
    feat_extra: dict[str, np.ndarray] = {
        col: np.zeros(n, dtype=object) for col in feat_cols_to_carry
    }

    for i, cid in enumerate(cand_ids):
        if cid in career_feats.index:
            row = career_feats.loc[cid]
            feat_arr[i]          = float(row.get("must_have_feature_score", 0.3))
            consulting_flag[i]   = bool(row.get("flag_consulting_only", False))
            ood_flag[i]          = bool(row.get("flag_out_of_domain", False))
            unavail_flag[i]      = bool(row.get("flag_unavailable", False))
            title_chaser_flag[i] = bool(row.get("flag_title_chaser", False))
            for col in feat_cols_to_carry:
                feat_extra[col][i] = row.get(col, False)

    beh_arr = np.full(n, 0.5, dtype=np.float64)    # default: neutral
    for i, cid in enumerate(cand_ids):
        if cid in behavioral_df.index:
            beh_arr[i] = float(behavioral_df.loc[cid, "behavioral_score"])

    # Weighted fusion
    raw = (
        W_EMBEDDING  * emb_arr
        + W_LLM      * llm_arr
        + W_FEATURES * feat_arr
        + W_BEHAVIORAL * beh_arr
    )

    # Multiplicative penalties
    honeypot_mask = np.array([cid in honeypot_ids for cid in cand_ids], dtype=bool)

    mult = np.ones(n, dtype=np.float64)
    mult[honeypot_mask]   *= HONEYPOT_MULT
    mult[consulting_flag] *= CONSULTING_MULT
    mult[rf_flag]         *= RED_FLAG_MULT
    mult[ood_flag]        *= OUT_OF_DOMAIN_MULT
    mult[unavail_flag]    *= UNAVAILABLE_MULT

    final = np.clip(raw * mult, 0.0, 1.0)

    elapsed = time.monotonic() - t0
    log.info("  Fusion done in %.2fs", elapsed)

    # Build output DataFrame
    df = pd.DataFrame({
        "candidate_id"            : cand_ids,
        "embedding_sim"           : emb_arr,
        "llm_score_norm"          : llm_arr,
        "must_have_feature_score" : feat_arr,
        "behavioral_score"        : beh_arr,
        "raw_fusion_score"        : raw,
        "final_score"             : final,
        "applied_honeypot"        : honeypot_mask,
        "applied_red_flag"        : rf_flag,
        "applied_consulting"      : consulting_flag,
        "applied_ood"             : ood_flag,
        "applied_unavail"         : unavail_flag,
    })
    for col in feat_cols_to_carry:
        df[col] = feat_extra[col]

    return df



# Select top 100 + attach reasoning


def select_top_100(
    fusion_df       : pd.DataFrame,
    reasoning_cache : dict[str, str],
    candidates_path : Path,
) -> pd.DataFrame:
    """
    Sort by final_score descending (tie-break: candidate_id ascending),
    take top 100, assign ranks 1–100, attach reasoning strings.
    Returns a 100-row DataFrame ready for CSV output.
    """
    log.info("Selecting top 100…")

    # Sort: primary = final_score desc, secondary = candidate_id asc (deterministic)
    sorted_df = fusion_df.sort_values(
        by=["final_score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    top100 = sorted_df.head(100).copy()
    top100["rank"] = range(1, 101)

    # Stream candidate metadata for template fallback
    missing_from_cache = set(top100["candidate_id"]) - set(reasoning_cache.keys())
    if missing_from_cache:
        log.info("  %d candidates need template reasoning fallback", len(missing_from_cache))
        meta = stream_candidate_metadata(candidates_path, missing_from_cache)
    else:
        meta = {}

    # Attach reasoning
    reasonings = []
    for _, row in top100.iterrows():
        cid  = row["candidate_id"]
        rank = int(row["rank"])

        if cid in reasoning_cache:
            reasonings.append(reasoning_cache[cid])
        else:
            # Template fallback — merge metadata into row dict for template
            row_dict = row.to_dict()
            row_dict.update(meta.get(cid, {
                "_title"  : "Candidate",
                "_company": "current employer",
                "_yoe"    : 0,
            }))
            reasonings.append(_template_reasoning(row_dict, rank))

    top100["reasoning"] = reasonings

    # Score normalisation
    # The submission spec requires scores to be non-increasing with rank.
    # Our final_score is already sorted descending, so this is satisfied.
    # But we verify strictly in the validation step.

    log.info(
        "  Top-100 score range: [%.6f, %.6f]",
        float(top100["final_score"].min()),
        float(top100["final_score"].max()),
    )
    log.info(
        "  Penalties applied: %d honeypot, %d red-flag, %d consulting, "
        "%d OOD, %d unavailable",
        int(top100["applied_honeypot"].sum()),
        int(top100["applied_red_flag"].sum()),
        int(top100["applied_consulting"].sum()),
        int(top100["applied_ood"].sum()),
        int(top100["applied_unavail"].sum()),
    )

    return top100



# Validation — strict per submission_spec.docx


def validate(top100: pd.DataFrame, all_valid_ids: set[str]) -> list[str]:
    """
    Run all validation checks from submission_spec.docx §3.
    Returns list of error strings. Empty list = pass.
    """
    errors = []

    # 1. Exactly 100 rows
    if len(top100) != 100:
        errors.append(f"Row count: expected 100, got {len(top100)}")

    # 2. Ranks 1-100 each exactly once
    ranks = sorted(top100["rank"].tolist())
    if ranks != list(range(1, 101)):
        errors.append(f"Ranks are not exactly 1-100 (found {ranks[:5]}…)")

    # 3. Each candidate_id appears exactly once
    dup_ids = top100[top100.duplicated("candidate_id")]["candidate_id"].tolist()
    if dup_ids:
        errors.append(f"Duplicate candidate_ids: {dup_ids[:5]}")

    # 4. All candidate_ids exist in candidates.jsonl
    unknown = set(top100["candidate_id"]) - all_valid_ids
    if unknown:
        errors.append(f"Unknown candidate_ids (not in candidates.jsonl): {list(unknown)[:5]}")

    # 5. Scores non-increasing with rank
    scores = top100.sort_values("rank")["final_score"].tolist()
    violations = [
        (i + 1, i + 2, scores[i], scores[i + 1])
        for i in range(len(scores) - 1)
        if scores[i] < scores[i + 1] - 1e-9   # tolerance for float rounding
    ]
    if violations:
        errors.append(
            f"Score not non-increasing at ranks: "
            + ", ".join(f"rank{a}={c:.6f} < rank{b}={d:.6f}" for a, b, c, d in violations[:3])
        )

    # 6. Required columns present
    required_cols = {"candidate_id", "rank", "score", "reasoning"}
    # Note: in the CSV we write 'score', not 'final_score'
    csv_cols = {"candidate_id", "rank", "score", "reasoning"}
    # We map final_score → score at write time; check after renaming
    if "final_score" not in top100.columns and "score" not in top100.columns:
        errors.append("Missing 'score' or 'final_score' column")

    # 7. Reasoning column: no empty strings for top-20 (these are manually reviewed)
    top20_reasoning = top100[top100["rank"] <= 20]["reasoning"].tolist()
    empty_reasoning = [i + 1 for i, r in enumerate(top20_reasoning) if not r or not r.strip()]
    if empty_reasoning:
        errors.append(f"Empty reasoning for top-20 ranks: {empty_reasoning[:5]}")

    return errors



# Write CSV


def write_csv(top100: pd.DataFrame, out_path: Path) -> None:
    """Write the final submission CSV in exact required format."""
    top100_sorted = top100.sort_values("rank").copy()
    top100_sorted = top100_sorted.rename(columns={"final_score": "score"})

    # Ensure score has 4 decimal places (spec example shows this)
    top100_sorted["score"] = top100_sorted["score"].round(6)

    out_cols = ["candidate_id", "rank", "score", "reasoning"]

    # Write with quoting — reasoning may contain commas
    top100_sorted[out_cols].to_csv(
        out_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONNUMERIC,
        quotechar='"',
    )
    size_kb = out_path.stat().st_size / 1024
    log.info("  Wrote %s (%.1f KB)", out_path, size_kb)



# Main


def main() -> None:
    ap = argparse.ArgumentParser(
        description="rank.py — Phase 2 Step 09: produce submission CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl (e.g. ./data/candidates.jsonl)",
    )
    ap.add_argument(
        "--out", required=True,
        help="Output CSV path (e.g. ./submission.csv)",
    )
    ap.add_argument(
        "--artifacts", default=None,
        help="Artifacts directory (default: <project_root>/artifacts)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print top-20 to stdout, skip writing CSV",
    )
    args = ap.parse_args()

    t_total = time.monotonic()

    # Resolve paths
    candidates_path = Path(args.candidates).resolve()
    out_path        = Path(args.out).resolve()

    if args.artifacts:
        artifacts = Path(args.artifacts).resolve()
    else:
        # Default: look for artifacts/ relative to rank.py location
        artifacts = Path(__file__).resolve().parent.parent / "artifacts"
        # Fallback: look relative to candidates.jsonl
        if not artifacts.exists():
            artifacts = candidates_path.parent.parent / "artifacts"

    log.info("=" * 60)
    log.info("rank.py — Phase 2 Step 09")
    log.info("  candidates : %s", candidates_path)
    log.info("  artifacts  : %s", artifacts)
    log.info("  output     : %s", out_path)
    log.info("=" * 60)

    # Verify all required artifacts exist before starting
    required_artifacts = [
        "embeddings.npy",
        "candidate_ids.json",
        "jd_embedding.npy",
        "career_features.parquet",
        "behavioral_scores.parquet",
        "llm_scores.json",
    ]
    missing = [f for f in required_artifacts if not (artifacts / f).exists()]
    if missing:
        log.error("Missing required artifacts: %s", missing)
        log.error("Run the offline pipeline steps first.")
        sys.exit(1)

    if not candidates_path.exists():
        log.error("candidates.jsonl not found: %s", candidates_path)
        sys.exit(1)

    # Load all artifacts
    t_load = time.monotonic()

    embs, cand_ids      = load_embeddings(artifacts)
    jd_emb              = load_jd_embedding(artifacts)
    career_feats        = load_career_features(artifacts)
    behavioral_df       = load_behavioral_scores(artifacts)
    llm_scores          = load_llm_scores(artifacts)
    reasoning_cache     = load_reasoning_cache(artifacts)
    honeypot_ids        = load_honeypot_ids(artifacts)

    log.info("  All artifacts loaded in %.1fs", time.monotonic() - t_load)

    # Compute cosine similarity
    sims = compute_cosine_similarity(embs, jd_emb)

    # Free embedding memory — not needed after similarity computed
    del embs
    del jd_emb

    # Fuse scores
    fusion_df = fuse_scores(
        cand_ids,
        sims,
        career_feats,
        behavioral_df,
        llm_scores,
        honeypot_ids,
    )

    # Select top-100 and attach reasoning
    top100 = select_top_100(fusion_df, reasoning_cache, candidates_path)

    # Dry-run: just print, don't write
    if args.dry_run:
        log.info("DRY RUN — top 20:")
        print()
        print(f"{'RANK':>4}  {'CANDIDATE_ID':>14}  {'SCORE':>8}  {'REASONING[:80]'}")
        print("-" * 115)
        for _, row in top100[top100["rank"] <= 20].iterrows():
            r = _clean_reasoning(str(row["reasoning"]))
            print(
                f"{int(row['rank']):>4}  {row['candidate_id']:>14}  "
                f"{float(row['final_score']):>8.6f}  {r[:80]}"
            )
        log.info("Dry run complete — no CSV written")
        return

    # Validate
    log.info("Running validation…")

    # Build set of all valid candidate IDs by streaming jsonl
    log.info("  Building valid-IDs set from candidates.jsonl…")
    all_valid_ids: set[str] = set()
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cid = json.loads(line).get("candidate_id", "")
                if cid:
                    all_valid_ids.add(cid)
            except json.JSONDecodeError:
                continue
    log.info("  Valid IDs in candidates.jsonl: %d", len(all_valid_ids))

    errors = validate(top100, all_valid_ids)
    if errors:
        log.error("VALIDATION FAILED — %d error(s):", len(errors))
        for e in errors:
            log.error("  ✗ %s", e)
        sys.exit(1)
    else:
        log.info("  ✅ Validation passed — all checks OK")

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(top100, out_path)

    # Summary
    elapsed = time.monotonic() - t_total
    log.info("=" * 60)
    log.info("✅  DONE in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    log.info("   Output: %s", out_path)
    log.info("   Top-3 candidates:")
    for _, row in top100[top100["rank"] <= 3].iterrows():
        log.info(
            "     #%d  %s  score=%.6f",
            int(row["rank"]), row["candidate_id"], float(row["final_score"]),
        )
    log.info(
        "   Weights used: emb=%.2f llm=%.2f feat=%.2f beh=%.2f",
        W_EMBEDDING, W_LLM, W_FEATURES, W_BEHAVIORAL,
    )
    if elapsed > 300:
        log.warning(
            "⚠  Total time %.1fs exceeds the 5-minute (300s) submission limit!", elapsed
        )
    log.info("=" * 60)


def _clean_reasoning(s: str) -> str:
    return s.replace("\n", " ").strip()


if __name__ == "__main__":
    main()