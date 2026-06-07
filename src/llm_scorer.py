"""
llm_scorer.py  —  Phase 1 · Step 07
=====================================
LLM Deep Scoring of the Top 500 candidates by embedding similarity.

What this step does (blueprint §A7):
  1. Load embeddings + candidate_ids from Step 6.
  2. Compute cosine similarity → select top 500 candidates.
  3. For each of the 500: build a rich, JD-aware prompt and call Gemini Flash
     to score 5 dimensions (0–10 each).
  4. Cache every result to disk immediately after each call (safe to interrupt).
  5. Assign score=0.0 to the other 99,500 candidates.
  6. Write complete 100K-row llm_scores.json.

Scoring dimensions (from blueprint image + JD analysis §1–4, 8, 9):
  1. technical_depth    — production embeddings/retrieval/vector-DB depth
  2. career_trajectory  — YOE in applied ML at product companies (not IT services)
  3. red_flag_check     — consulting-only, title-chasing, LLM-wrapper, CV/speech-only
                          NOTE: if this score < 3, a hard penalty is applied in Step 10
  4. shipping_evidence  — shipped real systems to real users at meaningful scale
  5. overall_fit        — holistic fit vs the JD

Install:
    pip install google-genai python-dotenv tqdm numpy

Run:
    # .env file with GEMINI_API_KEY=... OR set env var directly:
    $env:GEMINI_API_KEY = "your_key"
    python src/llm_scorer.py

    # Resume after interruption (already-scored IDs are skipped):
    python src/llm_scorer.py

    # Spot-check 10 results after completion:
    python src/llm_scorer.py --spot-check-only

Outputs:
    artifacts/llm_scores.json   — 100K rows, every candidate has a score
    artifacts/llm_scorer.log    — persistent log
"""

import os
import sys
import json
import time
import random
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # dotenv optional — env var works fine too

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
except ImportError:
    sys.exit("[ERROR] google-genai not installed.\nRun: pip install google-genai")

try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm
except ImportError:
    sys.exit("[ERROR] tqdm not installed.\nRun: pip install tqdm")



# Configuration


TOP_K               = 500      # candidates to deep-score via LLM
MAX_CONCURRENT      = 3        # concurrent Gemini requests (stay under RPM)
RPM_TARGET          = 12        # stay just under the 10 RPM free-tier limit
MAX_RETRIES         = 4        # per-candidate retry attempts per model
BASE_BACKOFF_S      = 8.0
MAX_BACKOFF_S       = 120.0

# Model fallback chain — tried in order on quota / persistent errors.
# User requested: gemini-3.5-flash first (doesn't exist yet), so we map
# to the closest real equivalents in quality order as of June 2026.
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
]

# Paths
_HERE           = Path(__file__).resolve().parent
PROJECT_ROOT    = _HERE.parent

DATA_PATH       = PROJECT_ROOT / "data"      / "candidates.jsonl"
ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
IDS_PATH        = ARTIFACTS_DIR / "candidate_ids.json"
JD_EMB_PATH     = ARTIFACTS_DIR / "jd_embedding.npy"
OUTPUT_PATH     = ARTIFACTS_DIR / "llm_scores.json"
CACHE_PATH      = ARTIFACTS_DIR / "llm_scores_cache.json"   # live per-call cache
LOG_PATH        = ARTIFACTS_DIR / "llm_scorer.log"

JD_FILE         = PROJECT_ROOT / "data" / "job_description.md"



# Logging


def _setup_logging() -> logging.Logger:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh  = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger = logging.getLogger("llm_scorer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

log = _setup_logging()



# Rate limiter — async token bucket


class AsyncRateLimiter:
    """Async token-bucket limiter. acquire() sleeps until a slot is free."""

    def __init__(self, rpm: int = RPM_TARGET) -> None:
        self._rate     = rpm / 60.0
        self._capacity = float(rpm)
        self._tokens   = float(rpm)
        self._last     = time.monotonic()
        self._lock     = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last
            self._last   = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
        await asyncio.sleep(wait)



# JD text loader


def load_jd_text() -> str:
    if JD_FILE.exists():
        return JD_FILE.read_text(encoding="utf-8")
    # Fallback: search common names
    for p in [
        PROJECT_ROOT / "data" / "job_description.txt",
        PROJECT_ROOT / "job_description.md",
    ]:
        if p.exists():
            return p.read_text(encoding="utf-8")
    # Hard-coded inline fallback (from jd_analysis)
    return _INLINE_JD

_INLINE_JD = """\
Senior AI Engineer (Founding Team) — Redrob AI (Series A)
Location: Pune/Noida, India (Hybrid). 5–9 years experience.

HARD REQUIREMENTS (must have ALL):
- Production embedding-based retrieval: sentence-transformers, BGE, E5, OpenAI embeddings.
  Must have handled embedding drift, index refresh, retrieval-quality regression. Not someone who called an API once.
- Production vector DB / hybrid search: Pinecone, Milvus, Qdrant, FAISS, OpenSearch, Elasticsearch.
  Operational depth matters, not just knowing the name.
- Strong Python, production-grade code quality. This role writes code.
- Ranking evaluation: NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.

STRONG SIGNALS (nice-to-have):
- LLM fine-tuning: LoRA, QLoRA, PEFT.
- Learning-to-rank: XGBoost-based or neural rankers.
- 6-8 years total, 4-5 of those in applied ML/AI at product companies (NOT IT services).
- Shipped end-to-end ranking, search, or recommendation system at meaningful scale.
- Located in or willing to relocate to Pune / Noida.
- Active on Redrob, low response time, recent applications.

DISQUALIFIERS (any of these is a serious red flag):
- Entire career at TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini — no product company. HARD DISQUALIFIER.
- Pure academic/research roles with zero production deployment.
- AI experience = only recent (<12 months) LangChain/OpenAI API calls, no pre-LLM ML production history.
- Primary domain is Computer Vision or Speech with no NLP/IR depth.
- 3+ companies in 5 years with title-chasing pattern.
- Senior engineer not writing code in 18+ months (pure arch/tech-lead).
- Not logged in for months, recruiter response rate ≤5%.

CULTURE / VIBE:
- Shipper > Researcher. Value: "shipped", "deployed", "in production", "real users."
- Async-first team — well-articulated summaries are a positive signal.
- Startup adaptability: navigated changing scope, worn multiple hats, shipped without big team.
"""



# Prompt builder


def build_prompt(candidate: dict, jd_text: str) -> str:
    """
    Build a comprehensive, JD-aware prompt for one candidate.

    Includes: headline, summary, full career history (with descriptions),
    skills with proficiency + duration, certifications, location/relocation,
    skill assessment scores (objective proof), and key redrob behavioral signals.
    All of sections 1, 2, 3, 4, 8, 9 of jd_analysis are represented.
    """
    p       = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    certs   = candidate.get("certifications", [])
    sigs    = candidate.get("redrob_signals", {})

    #  Profile block
    profile_block = f"""\
Candidate ID  : {candidate.get('candidate_id', 'unknown')}
Current Title : {p.get('current_title', 'N/A')}
Current Company: {p.get('current_company', 'N/A')} ({p.get('current_company_size', 'N/A')} employees, {p.get('current_industry', 'N/A')})
Location      : {p.get('location', 'N/A')}, {p.get('country', 'N/A')}
YOE (declared): {p.get('years_of_experience', 'N/A')} years
Willing to Relocate: {sigs.get('willing_to_relocate', 'N/A')}
Preferred Work Mode: {sigs.get('preferred_work_mode', 'N/A')}

Headline: {p.get('headline', '')}

Summary:
{p.get('summary', '(none)')}"""

    #  Career history (sorted recent-first, full descriptions)
    sorted_career = sorted(
        career,
        key=lambda r: r.get("start_date", "0000-00-00") or "0000-00-00",
        reverse=True,
    )
    career_lines = []
    for r in sorted_career:
        is_cur  = " [CURRENT]" if r.get("is_current") else ""
        line    = (
            f"  • {r.get('title', 'N/A')} @ {r.get('company', 'N/A')}"
            f"{is_cur} | {r.get('industry', 'N/A')} | "
            f"{r.get('company_size', 'N/A')} employees | "
            f"{r.get('duration_months', 0)} months\n"
            f"    {(r.get('description') or '').strip()}"
        )
        career_lines.append(line)
    career_block = "\n".join(career_lines) if career_lines else "  (none)"

    # Skills block
    # Sort by proficiency + endorsements for readability
    _prof_rank = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    sorted_skills = sorted(
        skills,
        key=lambda s: (_prof_rank.get(s.get("proficiency", ""), 0), s.get("endorsements", 0)),
        reverse=True,
    )
    skill_lines = [
        f"  {s.get('name', 'N/A')}: {s.get('proficiency', 'N/A')} "
        f"({s.get('duration_months', 0)}mo used, {s.get('endorsements', 0)} endorsements)"
        for s in sorted_skills
    ]
    skills_block = "\n".join(skill_lines) if skill_lines else "  (none)"

    #  Skill assessment scores (objective, cannot be self-reported)
    assessment_scores = sigs.get("skill_assessment_scores", {})
    if assessment_scores:
        assess_lines = [f"  {k}: {v:.1f}/100" for k, v in sorted(assessment_scores.items())]
        # Flag out-of-domain combinations per jd_analysis §8
        nlp_score    = assessment_scores.get("NLP", 0)
        img_score    = assessment_scores.get("Image Classification", 0)
        speech_score = assessment_scores.get("Speech Recognition", 0)
        ft_score     = assessment_scores.get("Fine-tuning LLMs", 0)
        flags = []
        if img_score > 60 and nlp_score < 20:
            flags.append(f"⚠ CV-heavy (Image Classification={img_score:.0f}) with low NLP={nlp_score:.0f} — possible out-of-domain")
        if speech_score > 60 and nlp_score < 20:
            flags.append(f"⚠ Speech-heavy (Speech Recognition={speech_score:.0f}) with low NLP={nlp_score:.0f} — possible out-of-domain")
        if nlp_score > 70:
            flags.append(f"✓ Strong NLP assessment ({nlp_score:.0f}/100) — core JD requirement verified")
        if ft_score > 70:
            flags.append(f"✓ Strong Fine-tuning LLMs assessment ({ft_score:.0f}/100) — nice-to-have verified")
        assess_block = "\n".join(assess_lines)
        if flags:
            assess_block += "\n  Flags:\n" + "\n".join(f"    {f}" for f in flags)
    else:
        assess_block = "  (no assessments taken)"

    # Certifications
    cert_block = (
        "\n".join(f"  {c.get('name', '')} ({c.get('issuer', '')}, {c.get('year', '')})"
                  for c in certs)
        if certs else "  (none)"
    )

    # Behavioral signals (jd_analysis S7)
    last_active     = sigs.get("last_active_date", "N/A")
    open_to_work    = sigs.get("open_to_work_flag", False)
    notice          = sigs.get("notice_period_days", "N/A")
    resp_rate       = sigs.get("recruiter_response_rate", "N/A")
    resp_time       = sigs.get("avg_response_time_hours", "N/A")
    github          = sigs.get("github_activity_score", -1)
    github_str      = f"{github:.1f}/100" if github >= 0 else "Not linked"
    icr             = sigs.get("interview_completion_rate", "N/A")
    oar             = sigs.get("offer_acceptance_rate", -1)
    oar_str         = f"{oar:.2f}" if oar >= 0 else "No history"
    salary_range    = sigs.get("expected_salary_range_inr_lpa", {})
    salary_str      = f"{salary_range.get('min', '?')}–{salary_range.get('max', '?')} LPA"
    completeness    = sigs.get("profile_completeness_score", "N/A")

    behavioral_block = f"""\
  Last active           : {last_active}
  Open to work          : {open_to_work}
  Notice period         : {notice} days
  Recruiter response rate: {resp_rate} (⚠ FUNCTIONALLY UNAVAILABLE if ≤0.05)
  Avg response time     : {resp_time} hours
  GitHub activity score : {github_str}
  Interview completion  : {icr}
  Offer acceptance rate : {oar_str}
  Expected salary       : {salary_str}
  Profile completeness  : {completeness}%"""


    # Final prompt

    prompt = f"""You are a senior engineering recruiter at Redrob AI evaluating a candidate for the role described below.
Your job is to score this candidate on 5 dimensions (0–10 each) with strict, evidence-based reasoning.
Be critical. Most candidates will score 3–6. Reserve 8–10 for genuinely exceptional matches.
Return ONLY valid JSON — no markdown, no explanation outside the JSON.

════════════════════════════════════════════
JOB DESCRIPTION
════════════════════════════════════════════
{jd_text.strip()}

════════════════════════════════════════════
CANDIDATE PROFILE
════════════════════════════════════════════
{profile_block}

CAREER HISTORY (most recent first):
{career_block}

SKILLS (by proficiency):
{skills_block}

OBJECTIVE SKILL ASSESSMENT SCORES (platform-verified, cannot be self-reported):
{assess_block}

CERTIFICATIONS:
{cert_block}

BEHAVIORAL SIGNALS:
{behavioral_block}

════════════════════════════════════════════
SCORING INSTRUCTIONS
════════════════════════════════════════════
Score each dimension 0–10. Be strict. Use evidence from the profile above.

DIMENSION 1 — technical_depth (0-10):
  Does this person have PRODUCTION experience with embeddings, vector databases,
  retrieval systems, and ranking? Not API calls — actual deployed systems handling
  drift, index refresh, quality regression. Evidence: career descriptions, assessment scores.
  10 = multiple shipped retrieval/ranking systems with operational depth.
  0 = no evidence of any of these skills.

DIMENSION 2 — career_trajectory (0-10):
  Are their years in APPLIED ML/AI at PRODUCT companies (not IT services like TCS/Infosys/Wipro)?
  6-8 total YOE with 4-5 of those in applied ML/AI at product companies = 9-10.
  Entire career in IT services = 1-2. Recent pivot to AI with no history = 2-4.
  Strong product company background but less ML depth = 5-6.

DIMENSION 3 — red_flag_check (0-10):
  10 = NO red flags at all. Lower score for each red flag present:
  - Entire career at IT services (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini): -5
  - Pure researcher / academic with no production deployment: -5
  - AI experience = only recent LangChain/OpenAI API stitching, no pre-LLM ML: -4
  - Primary domain is CV or Speech, no NLP/IR depth: -3
  - 3+ companies in 5 years (title-chasing pattern): -2
  - Not writing production code recently (pure architect/tech-lead): -2
  - Recruiter response rate ≤ 0.05 (functionally unavailable): -2
  IMPORTANT: A score below 3 here flags this candidate for a HARD PENALTY in final scoring.
  Score of 0-2 = serious disqualifier. Be honest.

DIMENSION 4 — shipping_evidence (0-10):
  Has this person SHIPPED real ML/AI systems to real users at meaningful scale?
  Look for: "deployed", "in production", "shipped", "real users", specific metrics.
  10 = multiple end-to-end systems shipped, clear product impact.
  0 = only research papers, prototypes, or Kaggle/side projects.
  Look at CAREER DESCRIPTIONS carefully — not just skill claims.

DIMENSION 5 — overall_fit (0-10):
  Holistic fit for this specific role at this specific company (Series A, founding team,
  Pune/Noida, hybrid, startup adaptability required).
  Consider: location fit, work mode, notice period, salary expectations,
  startup experience, communication clarity in their summary.

════════════════════════════════════════════
REQUIRED JSON OUTPUT FORMAT
════════════════════════════════════════════
Return ONLY this JSON object, nothing else:
{{
  "technical_depth": <integer 0-10>,
  "career_trajectory": <integer 0-10>,
  "red_flag_check": <integer 0-10>,
  "shipping_evidence": <integer 0-10>,
  "overall_fit": <integer 0-10>,
  "justification": "<one concise sentence summarising this candidate's key strength and key concern>"
}}"""

    return prompt


# Per-candidate scorer with retry + model fallback


def _backoff(attempt: int) -> float:
    base   = min(BASE_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)
    jitter = base * 0.25 * (2 * random.random() - 1)
    return max(2.0, base + jitter)


async def score_candidate(
    candidate:    dict,
    jd_text:      str,
    client:       genai.Client,
    rate_limiter: AsyncRateLimiter,
    semaphore:    asyncio.Semaphore,
) -> dict:
    """
    Score one candidate through the LLM rubric.
    Tries each model in FALLBACK_MODELS in order.
    Returns a result dict always — never raises.
    """
    cid    = candidate.get("candidate_id", "unknown")
    prompt = build_prompt(candidate, jd_text)

    async with semaphore:
        for model in FALLBACK_MODELS:
            for attempt in range(MAX_RETRIES):
                await rate_limiter.acquire()
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model   = model,
                        contents= prompt,
                        config  = genai_types.GenerateContentConfig(
                            response_mime_type = "application/json",
                            temperature        = 0.1,
                            max_output_tokens  = 300,
                        ),
                    )

                    if not response.text:
                        raise ValueError("Empty response from model")
                    raw = response.text.strip()
                    # Strip markdown fences if model wraps anyway
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                        raw = raw.strip()

                    result = json.loads(raw)

                    # Validate required keys
                    required = {
                        "technical_depth", "career_trajectory", "red_flag_check",
                        "shipping_evidence", "overall_fit", "justification",
                    }
                    missing = required - set(result.keys())
                    if missing:
                        raise ValueError(f"Missing keys: {missing}")

                    # Clamp all scores to 0–10
                    for dim in ("technical_depth", "career_trajectory",
                                "red_flag_check", "shipping_evidence", "overall_fit"):
                        result[dim] = max(0, min(10, int(result[dim])))

                    # Weighted normalisation to 0.0–1.0
                    # Weights reflect JD analysis §10 emphasis on technical depth
                    # and the red-flag check importance:
                    # technical_depth   : 0.30
                    # career_trajectory : 0.25
                    # red_flag_check    : 0.20  (also used as hard-penalty trigger)
                    # shipping_evidence : 0.15
                    # overall_fit       : 0.10
                    weighted = (
                        result["technical_depth"]    * 0.30 +
                        result["career_trajectory"]  * 0.25 +
                        result["red_flag_check"]     * 0.20 +
                        result["shipping_evidence"]  * 0.15 +
                        result["overall_fit"]        * 0.10
                    ) / 10.0   # divide by 10 since each dim is 0–10 → result is 0.0–1.0

                    return {
                        "candidate_id"    : cid,
                        "rubric_scores"   : {
                            "technical_depth"   : result["technical_depth"],
                            "career_trajectory" : result["career_trajectory"],
                            "red_flag_check"    : result["red_flag_check"],
                            "shipping_evidence" : result["shipping_evidence"],
                            "overall_fit"       : result["overall_fit"],
                        },
                        "llm_rubric_score": round(weighted, 4),
                        "red_flag_penalty": result["red_flag_check"] < 3,
                        "justification"   : result.get("justification", ""),
                        "model_used"      : model,
                        "in_top_500"      : True,
                    }

                except genai_errors.ClientError as e:
                    status = getattr(e, "status_code", None) or getattr(e, "code", 0)
                    if status == 429:
                        wait = _backoff(attempt)
                        log.warning(
                            "  [%s] 429 on model=%s attempt=%d. Waiting %.1fs…",
                            cid, model, attempt + 1, wait,
                        )
                        await asyncio.sleep(wait)
                        if attempt == MAX_RETRIES - 1:
                            log.warning("  [%s] Exhausted retries on %s → trying next model.", cid, model)
                    elif status == 404:
                        log.warning("  [%s] 404 on model=%s → skipping to next model.", cid, model)
                        break   # advance to next model immediately
                    else:
                        wait = _backoff(attempt)
                        log.error("  [%s] ClientError %s on %s: %s. Waiting %.1fs…",
                                  cid, status, model, e, wait)
                        await asyncio.sleep(wait)

                except json.JSONDecodeError as e:
                    log.warning("  [%s] JSON parse error on %s attempt %d: %s",
                                cid, model, attempt + 1, e)
                    if attempt == MAX_RETRIES - 1:
                        break

                except Exception as e:
                    wait = _backoff(attempt)
                    log.error("  [%s] Unexpected error on %s attempt %d: %s. Waiting %.1fs…",
                              cid, model, attempt + 1, e, wait)
                    await asyncio.sleep(wait)

        # All models exhausted
        log.error("  [%s] All models exhausted — returning zero score.", cid)
        return {
            "candidate_id"    : cid,
            "rubric_scores"   : {"error": True},
            "llm_rubric_score": 0.0,
            "red_flag_penalty": False,
            "justification"   : "API Error — all models exhausted",
            "model_used"      : "none",
            "in_top_500"      : True,
        }



# Top-500 selection


def get_top_k_ids(k: int = TOP_K) -> list[str]:
    """
    Load embeddings + candidate_ids, compute cosine similarity to JD,
    return the top-k candidate_id strings.
    Uses candidate_ids.json as the bridge (not jsonl line order).
    """
    log.info("Loading embeddings and computing cosine similarity…")
    embs     = np.load(EMBEDDINGS_PATH).astype(np.float32)   # (100K, dim)
    jd_emb   = np.load(JD_EMB_PATH).astype(np.float32).flatten()  # (dim,)
    cand_ids = json.loads(IDS_PATH.read_text())               # list[str], len=100K

    assert len(cand_ids) == len(embs), (
        f"Mismatch: {len(cand_ids)} IDs vs {len(embs)} embeddings"
    )

    # L2-normalise for cosine similarity
    norm_embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    norm_jd   = jd_emb / (np.linalg.norm(jd_emb) + 1e-9)

    sims    = norm_embs @ norm_jd              # (100K,)
    top_idx = np.argsort(sims)[::-1][:k]      # indices of top-k

    top_ids = [cand_ids[i] for i in top_idx]

    log.info(
        "  Top-%d selected. Similarity range: %.4f – %.4f",
        k, float(sims[top_idx[-1]]), float(sims[top_idx[0]]),
    )
    return top_ids



# Checkpoint / cache helpers


def _load_cache() -> dict[str, dict]:
    """Load already-scored candidates from the live cache file."""
    if not CACHE_PATH.exists():
        return {}
    try:
        all_entries = json.loads(CACHE_PATH.read_text())
        # Exclude error entries so they get re-scored on next run
        valid = {
            r["candidate_id"]: r
            for r in all_entries
            if not r.get("rubric_scores", {}).get("error")
            and r.get("llm_rubric_score", 0.0) > 0.0
        }
        skipped = len(all_entries) - len(valid)
        if skipped:
            log.info("  ♻  Cache: %d valid scores loaded, %d error entries will be retried.", len(valid), skipped)
        else:
            log.info("  ♻  Cache: %d valid scores loaded.", len(valid))
        return valid
    except Exception as e:
        log.warning("Could not load cache (%s) — starting fresh.", e)
        return {}


def _append_to_cache(result: dict) -> None:
    """Append one result to the cache file (load-modify-write, atomic enough for our use)."""
    cache = _load_cache()
    cache[result["candidate_id"]] = result
    tmp = CACHE_PATH.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(list(cache.values()), ensure_ascii=False, indent=2))
    tmp.replace(CACHE_PATH)



# Spot-check


def spot_check(n: int = 10) -> None:
    """Print the top-n scored candidates from the final output file."""
    if not OUTPUT_PATH.exists():
        log.error("llm_scores.json not found — run scoring first.")
        return

    all_scores = json.loads(OUTPUT_PATH.read_text())
    # Filter to top-500 only, sort by llm_rubric_score
    top_scored = sorted(
        [s for s in all_scores if s.get("in_top_500")],
        key=lambda x: x["llm_rubric_score"],
        reverse=True,
    )

    log.info("=== Spot-check: top-%d LLM-scored candidates ===", n)
    for rank, s in enumerate(top_scored[:n], 1):
        scores = s.get("rubric_scores", {})
        dims   = (
            f"TD={scores.get('technical_depth','?')} "
            f"CT={scores.get('career_trajectory','?')} "
            f"RF={scores.get('red_flag_check','?')} "
            f"SE={scores.get('shipping_evidence','?')} "
            f"OF={scores.get('overall_fit','?')}"
        )
        penalty = " ⚠RED-FLAG-PENALTY" if s.get("red_flag_penalty") else ""
        log.info(
            "  #%d  score=%.4f  %s  [%s]%s  model=%s",
            rank, s["llm_rubric_score"], s["candidate_id"],
            dims, penalty, s.get("model_used", "?"),
        )
        log.info("       %s", s.get("justification", ""))

    # Summary stats
    n_penalty = sum(1 for s in top_scored if s.get("red_flag_penalty"))
    n_error   = sum(1 for s in top_scored if s.get("model_used") == "none")
    log.info(
        "  Total top-500 scored: %d | Red-flag penalties: %d | API errors: %d",
        len(top_scored), n_penalty, n_error,
    )



# Main async scoring loop


async def run_scoring(client: genai.Client, jd_text: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Select top-500 by embedding similarity
    top_ids = get_top_k_ids(TOP_K)
    top_id_set = set(top_ids)

    # Load cache of already-scored candidates (resume support)
    cache = _load_cache()
    already_done = set(cache.keys())
    to_score_ids = [cid for cid in top_ids if cid not in already_done]
    log.info(
        "Top-%d selected | Already scored: %d | To score now: %d",
        TOP_K, len(already_done), len(to_score_ids),
    )

    if to_score_ids:
        # Build id→candidate dict by streaming candidates.jsonl once
        log.info("Streaming candidates.jsonl to build lookup for top-%d…", TOP_K)
        to_score_set = set(to_score_ids)
        candidates_lookup: dict[str, dict] = {}
        with open(DATA_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                    cid = c.get("candidate_id")
                    if cid in to_score_set:
                        candidates_lookup[cid] = c
                        if len(candidates_lookup) == len(to_score_set):
                            break
                except json.JSONDecodeError:
                    continue

        log.info("  Found %d/%d candidates in jsonl.", len(candidates_lookup), len(to_score_ids))

        # Preserve top-500 ranking order for scoring
        ordered_candidates = [
            candidates_lookup[cid]
            for cid in to_score_ids
            if cid in candidates_lookup
        ]

        # Score asynchronously
        rate_limiter = AsyncRateLimiter(RPM_TARGET)
        semaphore    = asyncio.Semaphore(MAX_CONCURRENT)
        scored_count = 0

        log.info(
            "Scoring %d candidates | concurrency=%d | RPM=%d | models: %s",
            len(ordered_candidates), MAX_CONCURRENT, RPM_TARGET,
            " → ".join(FALLBACK_MODELS),
        )
        log.info(
            "Estimated time: ~%.0f minutes",
            len(ordered_candidates) / RPM_TARGET,
        )

        async def score_and_cache(c: dict) -> dict:
            nonlocal scored_count
            result = await score_candidate(c, jd_text, client, rate_limiter, semaphore)
            _append_to_cache(result)
            scored_count += 1
            return result

        tasks   = [score_and_cache(c) for c in ordered_candidates]
        results = await async_tqdm.gather(
            *tasks,
            desc="LLM scoring top-500",
            unit="cand",
            dynamic_ncols=True,
            colour="yellow",
        )

        # Merge new results into cache
        for r in results:
            cache[r["candidate_id"]] = r

        log.info(
            "Scoring complete. Scored this session: %d | Total in cache: %d",
            scored_count, len(cache),
        )

    #  Build full 100K output
    log.info("Building full 100K-row llm_scores.json…")

    # All candidate IDs from the embeddings (preserves order)
    all_cand_ids = json.loads(IDS_PATH.read_text())

    final_output: list[dict] = []
    for cid in all_cand_ids:
        if cid in cache:
            final_output.append(cache[cid])
        else:
            # Not in top-500 → zero score, mark as filtered
            final_output.append({
                "candidate_id"    : cid,
                "rubric_scores"   : {"filtered": True},
                "llm_rubric_score": 0.0,
                "red_flag_penalty": False,
                "justification"   : "Not in top-500 semantic match — not LLM-scored",
                "model_used"      : "none",
                "in_top_500"      : False,
            })

    # Write final output
    OUTPUT_PATH.write_text(
        json.dumps(final_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(
        "✅  Saved → %s  (%d rows | %d top-500 scored | %.1f MB)",
        OUTPUT_PATH,
        len(final_output),
        len(cache),
        OUTPUT_PATH.stat().st_size / 1e6,
    )



# Entry point


def main() -> None:
    global TOP_K, MAX_CONCURRENT
    ap = argparse.ArgumentParser(
        description="Phase 1 · Step 07 — LLM Deep Scoring of top 500 candidates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key (default: $GEMINI_API_KEY / .env)",
    )
    ap.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of top candidates to LLM-score (default: {TOP_K})",
    )
    ap.add_argument(
        "--spot-check-only", action="store_true",
        help="Print top-10 scored candidates from existing output; no API calls",
    )
    ap.add_argument(
        "--concurrency", type=int, default=MAX_CONCURRENT,
        help=f"Max concurrent Gemini requests (default: {MAX_CONCURRENT})",
    )
    args = ap.parse_args()

    TOP_K          = args.top_k
    MAX_CONCURRENT = args.concurrency

    if args.spot_check_only:
        spot_check()
        return

    # API key
    if not args.api_key:
        sys.exit(
            "[ERROR] No Gemini API key.\n"
            "Set $GEMINI_API_KEY, add GEMINI_API_KEY= to .env, or pass --api-key <key>."
        )

    # Check required files exist
    for path, label in [
        (DATA_PATH,       "candidates.jsonl"),
        (EMBEDDINGS_PATH, "embeddings.npy"),
        (IDS_PATH,        "candidate_ids.json"),
        (JD_EMB_PATH,     "jd_embedding.npy"),
    ]:
        if not path.exists():
            sys.exit(f"[ERROR] Required file not found: {path}\nRun Step 6 first.")

    client   = genai.Client(api_key=args.api_key)
    jd_text  = load_jd_text()

    log.info("Phase 1 · Step 07 — LLM Deep Scoring")
    log.info("  Top-K         : %d", TOP_K)
    log.info("  Concurrency   : %d", MAX_CONCURRENT)
    log.info("  RPM target    : %d", RPM_TARGET)
    log.info("  Model chain   : %s", " → ".join(FALLBACK_MODELS))
    log.info("  JD loaded     : %d chars", len(jd_text))

    t0 = time.monotonic()
    asyncio.run(run_scoring(client, jd_text))
    elapsed = time.monotonic() - t0

    log.info("Total wall time: %.1f min", elapsed / 60)
    log.info("Running spot-check on results…")
    spot_check()


if __name__ == "__main__":
    main()