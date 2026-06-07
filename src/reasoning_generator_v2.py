"""
reasoning_generator_v2.py  —  Phase 1 · Step 08 (targeted re-run)
==================================================================
Regenerates reasoning strings for exactly the top-100 candidates
from your final submission CSV, replacing them in reasoning_cache.json.



Run:
    $env:GEMINI_API_KEY = "your_key"
    python src/regen_top100_reasoning.py --csv submission.csv

    # Preview results without overwriting cache:
    python src/regen_top100_reasoning.py --csv submission.csv --dry-run

    # After completion, review all 100:
    python src/regen_top100_reasoning.py --review-only --csv submission.csv

Output:
    Updates artifacts/reasoning_cache.json in-place for the 100 candidates.
    Backs up original to artifacts/reasoning_cache_backup.json first.
"""

import os
import re
import sys
import csv
import json
import time
import random
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
except ImportError:
    sys.exit("[ERROR] google-genai not installed.  Run: pip install google-genai")

try:
    from tqdm.asyncio import tqdm as async_tqdm
except ImportError:
    sys.exit("[ERROR] tqdm not installed.  Run: pip install tqdm")



# Configuration


MAX_CONCURRENT  = 3
RPM_TARGET      = 12
MAX_RETRIES     = 5
BASE_BACKOFF_S  = 8.0
MAX_BACKOFF_S   = 120.0

FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
]

MIN_WORDS = 20
MAX_WORDS = 30

_HERE           = Path(__file__).resolve().parent
PROJECT_ROOT    = _HERE.parent
DATA_PATH       = PROJECT_ROOT / "data"      / "candidates.jsonl"
ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts"
CACHE_PATH      = ARTIFACTS_DIR / "reasoning_cache.json"
BACKUP_PATH     = ARTIFACTS_DIR / "reasoning_cache_backup.json"
LIVE_PATH       = ARTIFACTS_DIR / "regen_top100_live.json"
LOG_PATH        = ARTIFACTS_DIR / "regen_top100.log"



# Logging


def _setup_logging() -> logging.Logger:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt    = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
    sh     = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh     = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger = logging.getLogger("regen_top100")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

log = _setup_logging()



# Rate limiter


class AsyncRateLimiter:
    def __init__(self, rpm: int = RPM_TARGET) -> None:
        self._rate     = rpm / 60.0
        self._capacity = float(rpm)
        self._tokens   = float(rpm)
        self._last     = time.monotonic()
        self._lock     = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now          = time.monotonic()
            elapsed      = now - self._last
            self._last   = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
        await asyncio.sleep(wait)



# Fact extraction + allowed-term set


def extract_facts(candidate: dict, rank: int, final_score: float) -> dict:
    p      = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    sigs   = candidate.get("redrob_signals", {})
    edu    = candidate.get("education", [])
    certs  = candidate.get("certifications", [])

    sorted_career = sorted(
        career,
        key=lambda r: r.get("start_date", "0000-00-00") or "0000-00-00",
        reverse=True,
    )
    _prof = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    sorted_skills = sorted(
        skills,
        key=lambda s: (s.get("endorsements", 0), _prof.get(s.get("proficiency", ""), 0)),
        reverse=True,
    )

    # Build allowed-terms for hallucination check
    allowed = set()
    for r in career:
        for f in [r.get("company", ""), r.get("title", ""), r.get("industry", "")]:
            if f:
                allowed.add(f)
                for w in f.split():
                    if len(w) > 2:
                        allowed.add(w)
    for s in skills:
        n = s.get("name", "")
        if n:
            allowed.add(n)
            for w in n.split():
                if len(w) > 2:
                    allowed.add(w)
    for e in edu:
        inst = e.get("institution", "")
        if inst:
            allowed.add(inst)
            for w in inst.split():
                if len(w) > 2:
                    allowed.add(w)
    for c in certs:
        n = c.get("name", "")
        if n:
            allowed.add(n)
    for f in [p.get("current_title", ""), p.get("current_company", ""),
              p.get("current_industry", ""), p.get("location", ""), p.get("country", "")]:
        if f:
            allowed.add(f)
            for w in f.split():
                if len(w) > 2:
                    allowed.add(w)
    assessments = sigs.get("skill_assessment_scores", {})
    for k in assessments:
        allowed.add(k)
        for w in k.split():
            if len(w) > 2:
                allowed.add(w)

    # Always-safe tech and location terms
    allowed |= {
        "AI", "ML", "NLP", "IR", "LLM", "LLMs", "RAG", "API", "APIs",
        "Python", "SQL", "GCP", "AWS", "Azure", "FAISS", "NDCG", "MRR", "MAP",
        "LoRA", "QLoRA", "PEFT", "XGBoost", "GPU", "CPU", "SaaS", "B2B",
        "Redrob", "Series", "Pune", "Noida", "India", "IIT", "IIM",
        "GitHub", "LinkedIn", "Kafka", "Spark", "Airflow", "dbt",
        "Pinecone", "Milvus", "Qdrant", "Elasticsearch", "OpenSearch",
        "Kubernetes", "Docker", "TensorFlow", "PyTorch", "HuggingFace",
        "Transformers", "BERT", "GPT", "Gemini", "OpenAI",
        "Weaviate", "LlamaIndex", "LangChain", "BentoML", "Haystack",
        "Weaviate", "Sentence", "Semantic", "Dense", "Hybrid",
    }

    salary   = sigs.get("expected_salary_range_inr_lpa", {})
    github   = sigs.get("github_activity_score", -1)
    notice   = sigs.get("notice_period_days")
    open_w   = sigs.get("open_to_work_flag", False)
    relocate = sigs.get("willing_to_relocate", False)
    location = p.get("location", "")

    return {
        "candidate_id"    : candidate.get("candidate_id"),
        "rank"            : rank,
        "final_score"     : final_score,
        "current_title"   : p.get("current_title", ""),
        "current_company" : p.get("current_company", ""),
        "current_industry": p.get("current_industry", ""),
        "yoe"             : p.get("years_of_experience", 0),
        "location"        : location,
        "country"         : p.get("country", ""),
        "summary"         : p.get("summary", ""),
        "career"          : [
            {
                "title"      : r.get("title", ""),
                "company"    : r.get("company", ""),
                "industry"   : r.get("industry", ""),
                "duration"   : r.get("duration_months", 0),
                "is_current" : r.get("is_current", False),
                "description": (r.get("description") or "")[:500],
            }
            for r in sorted_career[:4]
        ],
        "top_skills"      : [
            {
                "name"        : s.get("name", ""),
                "proficiency" : s.get("proficiency", ""),
                "duration_mo" : s.get("duration_months", 0),
                "endorsements": s.get("endorsements", 0),
            }
            for s in sorted_skills[:7]
        ],
        "assessments"     : assessments,
        "open_to_work"    : open_w,
        "notice_days"     : notice,
        "github_score"    : github,
        "willing_relocate": relocate,
        "salary_min_lpa"  : salary.get("min"),
        "salary_max_lpa"  : salary.get("max"),
        "_allowed"        : allowed,
    }



# Prompt builder  —  rank-aware, no internal metadata


def build_prompt(facts: dict) -> str:
    rank      = facts["rank"]
    yoe       = facts["yoe"]
    title     = facts["current_title"]
    company   = facts["current_company"]
    industry  = facts["current_industry"]
    location  = facts["location"]
    notice    = facts["notice_days"]
    github    = facts["github_score"]
    relocate  = facts["willing_relocate"]
    open_w    = facts["open_to_work"]
    salary    = facts["salary_min_lpa"]

    # Career block
    career_lines = []
    for r in facts["career"]:
        cur  = " [current]" if r["is_current"] else ""
        line = (
            f"  - {r['title']} @ {r['company']}{cur} "
            f"({r['duration']}mo, {r['industry']})\n"
            f"    {r['description']}"
        )
        career_lines.append(line)
    career_block = "\n".join(career_lines) or "  (none)"

    # Skills block
    skill_lines = [
        f"  - {s['name']}: {s['proficiency']} ({s['duration_mo']}mo, {s['endorsements']} endorsements)"
        for s in facts["top_skills"]
    ]
    skills_block = "\n".join(skill_lines) or "  (none)"

    # Assessment block
    assess_block = (
        "\n".join(f"  - {k}: {v:.1f}/100" for k, v in sorted(facts["assessments"].items()))
        if facts["assessments"] else "  (none)"
    )

    # Logistics
    notice_str  = f"{notice} days" if notice is not None else "unknown"
    github_str  = f"{github:.1f}/100" if github >= 0 else "not linked"
    salary_str  = f"{salary} LPA min" if salary else "not disclosed"
    relocate_str= "yes" if relocate else "no"
    open_str    = "yes (actively looking)" if open_w else "no (passive)"

    # Rank-tier context
    if rank <= 10:
        tier_context = (
            f"This is a TOP-{rank} candidate — one of the best matches in the pool of 100K. "
            f"They should be an immediate shortlist priority. "
            f"Lead with their STRONGEST differentiated signal. "
            f"The concern should be minor or logistical only."
        )
    elif rank <= 30:
        tier_context = (
            f"This is a strong candidate ranked #{rank}. "
            f"They have clear JD alignment but one genuine limitation prevents a higher rank. "
            f"Name both the key strength AND the key limitation clearly."
        )
    elif rank <= 60:
        tier_context = (
            f"This candidate ranked #{rank} — a strong contender from a pool of 100,000. "
            f"They have solid ML/AI credentials. Lead with their key strength, "
            f"then name the ONE specific limitation that keeps them from the top 30. "
            f"Tone: positive but honest."
        )
    else:
        tier_context = (
            f"This candidate ranked #{rank} out of 100 — still in the top 100 from a pool of 100,000. "
            f"They have real, relevant ML/AI credentials worth recognising. "
            f"Lead with their genuine strength first, then name ONE specific concern "
            f"that explains why they rank here rather than in the top 50. "
            f"Tone: respectful and balanced, not dismissive."
        )

    return f"""You are a senior technical recruiter at Redrob AI writing a ranking justification.
The role: Senior AI Engineer (Founding Team) — production embeddings, vector DBs, retrieval/ranking at scale.

{tier_context}

══ CANDIDATE FACTS — USE ONLY THESE, INVENT NOTHING ══
Role       : {title} @ {company} ({industry})
YOE        : {yoe} years
Location   : {location}, {facts['country']}

Summary (their own words):
{facts['summary']}

Career history (recent first):
{career_block}

Top skills:
{skills_block}

Platform assessment scores (objective, verified):
{assess_block}

Availability:
  Open to work      : {open_str}
  Notice period     : {notice_str}
  Willing to relocate: {relocate_str}
  GitHub activity   : {github_str}
  Expected salary   : {salary_str}
══════════════════════════════════════════════════════════

TASK: Write a 1–2 sentence recruiter reasoning for why this candidate is ranked #{rank}.

STRICT RULES:
1. NO internal scoring metadata: do NOT mention "weighted score", "score 0.XXX",
   "ranked #N", "llm score", or any numerical score from the ranking system.
   These are internal — a recruiter would never write them.
2. ZERO hallucination: only mention companies, skills, technologies, or tools
   that appear EXACTLY in the facts above. If unsure — omit it.
3. Must name at least one SPECIFIC company from their career.
4. Must name at least one SPECIFIC skill or technology from their profile.
5. Must reference their YOE as a number ({yoe} years).
6. Must connect to a SPECIFIC JD requirement (production retrieval/embeddings/
   vector DBs/ranking evaluation/Python/LLM fine-tuning).
7. Must honestly name the ONE most important concern or limitation.
8. Length: {MIN_WORDS}–{MAX_WORDS} words. ONE sentence only. No bullets. Be extremely concise.9. Vary your sentence opening — do NOT start with "The candidate" or
   "With X years of experience".
10. No generic phrases like "strong candidate with relevant experience".
    Every sentence must contain a specific, verifiable fact.

Return ONLY the reasoning text. Nothing else."""



# Hallucination verifier


COMMON_WORDS = {
    "The", "This", "Their", "They", "Has", "Have", "With", "From", "Been",
    "Strong", "Deep", "Solid", "Clear", "Direct", "Real", "Key", "Core",
    "Senior", "Lead", "Head", "Principal", "Staff", "Junior", "Mid",
    "Years", "Months", "Year", "Month",
    "Role", "Team", "Work", "Job", "Skills", "Experience", "Background",
    "Production", "Technical", "Engineering", "Product", "Platform",
    "System", "Systems", "Service", "Services", "Data", "Model", "Models",
    "Search", "Ranking", "Retrieval", "Matching", "Recommendation",
    "Machine", "Learning", "Natural", "Language", "Processing", "Computer",
    "Large", "Scale", "Users", "Impact", "Results", "Evidence",
    "Company", "Companies", "Startup", "Founding", "Stage", "Early",
    "India", "Indian", "Canadian", "German", "American",
    "Open", "Willing", "Available", "Active", "Current", "Recent", "Former",
    "Notice", "Period", "Salary", "Relocation", "Remote", "Hybrid", "Onsite",
    "Concern", "Limitation", "Caveat", "However", "Although", "Despite",
    "Master", "Bachelor", "Doctor", "Institute", "University", "College",
    "Candidate", "Engineer", "Scientist", "Researcher", "Developer",
    "Manager", "Director", "Architect", "Analyst", "Consultant", "Specialist",
    "Founded", "Built", "Led", "Shipped", "Deployed", "Designed", "Owned",
    "Worked", "Managed", "Developed", "Created", "Improved", "Reduced",
    "Demonstrated", "Proven", "Hands", "End", "Side",
    "Passive", "Active", "Public", "Private",
    "Both", "Either", "First", "Second", "Last", "Next",
    "Overall", "Primary", "Secondary", "Main", "Key", "Top", "Bottom",
    "None", "Some", "All", "Most", "Many", "Few",
}


def verify(reasoning: str, facts: dict) -> tuple[bool, list[str]]:
    allowed = facts["_allowed"]
    tokens  = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', reasoning)
    bad     = []
    for t in tokens:
        if t in COMMON_WORDS:
            continue
        if t in allowed:
            continue
        if any(t in term or term in t for term in allowed):
            continue
        bad.append(t)
    return len(bad) == 0, bad


def contains_internal_metadata(reasoning: str) -> bool:
    """Detect if the LLM leaked internal scoring metadata into the reasoning."""
    patterns = [
        r'ranked? #\d+',
        r'weighted score',
        r'score \d\.\d',
        r'llm.{0,5}score',
        r'\bscore of \d',
        r'rubric',
        r'0\.\d{3,4}',     # score like 0.965
    ]
    text_lower = reasoning.lower()
    for pat in patterns:
        if re.search(pat, text_lower):
            return True
    return False



# Template fallback


def make_template(facts: dict) -> str:
    rank    = facts["rank"]
    title   = facts["current_title"]
    company = facts["current_company"]
    yoe     = facts["yoe"]
    loc     = facts["location"]
    notice  = facts["notice_days"]
    relocate= facts["willing_relocate"]
    top_s   = facts["top_skills"][0]["name"] if facts["top_skills"] else "ML"
    assess  = facts["assessments"]
    best    = max(assess.items(), key=lambda x: x[1]) if assess else None
    ass_str = f" and a {best[1]:.0f}/100 {best[0]} platform assessment" if best else ""

    if not relocate and loc not in {"Pune", "Noida", "Mumbai", "Delhi", "Hyderabad", "Bengaluru"}:
        concern = f"unwillingness to relocate from {loc} is a practical concern for this founding-team role"
    elif notice and notice > 60:
        concern = f"a {notice}-day notice period exceeds the preferred threshold for an immediate founding-team hire"
    elif not facts["open_to_work"]:
        concern = "they are currently a passive candidate, requiring proactive outreach"
    elif rank > 60:
        concern = "domain depth on production retrieval systems is not fully demonstrated in the profile"
    else:
        concern = "further verification of production deployment depth would strengthen their case"

    return (
        f"{yoe}-year {title} at {company} with {top_s} background{ass_str}; "
        f"ranked #{rank} because {concern}."
    )



# Per-candidate generator


def _backoff(attempt: int) -> float:
    base   = min(BASE_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)
    jitter = base * 0.25 * (2 * random.random() - 1)
    return max(2.0, base + jitter)


async def generate_one(
    facts:        dict,
    client:       genai.Client,
    rate_limiter: AsyncRateLimiter,
    semaphore:    asyncio.Semaphore,
) -> tuple[str, str, bool]:
    """
    Returns (candidate_id, reasoning_string, used_template).
    """
    cid    = facts["candidate_id"]
    prompt = build_prompt(facts)

    async with semaphore:
        for model in FALLBACK_MODELS:
            for attempt in range(MAX_RETRIES):
                await rate_limiter.acquire()
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model    = model,
                        contents = prompt,
                        config   = genai_types.GenerateContentConfig(
                            temperature       = 0.25,
                            max_output_tokens = 160,
                        ),
                    )

                    if not response.text:
                        raise ValueError("Empty response")

                    raw = response.text.strip()

                    REPLACEMENTS = {
                        '\u2018': "'", '\u2019': "'", '\u201a': "'",
                        '\u201c': '"', '\u201d': '"', '\u201e': '"',
                        '\u2013': '-', '\u2014': '-', '\u2026': '...',
                        '\u00e2\u0080\u0099': "'",
                    }
                    for char, replacement in REPLACEMENTS.items():
                        raw = raw.replace(char, replacement)

                    # Strip markdown fences
                    if raw.startswith("```"):
                        raw = re.sub(r'```[a-z]*', '', raw).strip('` \n')
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = raw[1:-1].strip()

                    # Word count
                    words = len(raw.split())
                    if words < MIN_WORDS:
                        log.warning("  [%s] Too short (%dw) — retry.", cid, words)
                        continue
                    if words > MAX_WORDS:
                        sentences = re.split(r'(?<=[.!?])\s+', raw)
                        raw = " ".join(sentences[:2])

                    # Check for internal metadata leak
                    if contains_internal_metadata(raw):
                        log.warning(
                            "  [%s] Internal metadata in output — retry. Text: %s",
                            cid, raw[:80]
                        )
                        continue

                    # Hallucination check
                    ok, bad = verify(raw, facts)
                    if not ok:
                        log.warning("  [%s] Hallucination %s — template.", cid, bad[:3])
                        return cid, make_template(facts), True

                    log.info(
                        "  [%s] rank=#%d %dw model=%s",
                        cid, facts["rank"], len(raw.split()), model,
                    )
                    return cid, raw, False

                except genai_errors.ClientError as e:
                    status = getattr(e, "status_code", None) or getattr(e, "code", 0)
                    if status == 429:
                        wait = _backoff(attempt)
                        log.warning("  [%s] 429 %s attempt %d. Wait %.1fs…", cid, model, attempt+1, wait)
                        await asyncio.sleep(wait)
                        if attempt == MAX_RETRIES - 1:
                            break
                    elif status == 404:
                        log.warning("  [%s] 404 %s → next model.", cid, model)
                        break
                    else:
                        wait = _backoff(attempt)
                        log.error("  [%s] Error %s %s. Wait %.1fs…", cid, status, model, wait)
                        await asyncio.sleep(wait)

                except Exception as e:
                    wait = _backoff(attempt)
                    log.error("  [%s] %s attempt %d. Wait %.1fs…", cid, e, attempt+1, wait)
                    await asyncio.sleep(wait)

    log.error("  [%s] All models failed — template.", cid)
    return cid, make_template(facts), True



# CSV reader


def read_submission_csv(csv_path: Path) -> list[tuple[int, str, float]]:
    """
    Read submission CSV and return list of (rank, candidate_id, score).
    Handles both rank_score and score columns.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rank  = int(row.get("rank", row.get("Rank", 0)))
            cid   = row.get("candidate_id", row.get("Candidate_id", "")).strip()
            score = float(row.get("score", row.get("Score", row.get("rank_score", 0))))
            rows.append((rank, cid, score))
    rows.sort(key=lambda x: x[0])
    return rows



# Review mode


def review(csv_path: Path, n: int = 100) -> None:
    if not CACHE_PATH.exists():
        log.error("reasoning_cache.json not found.")
        return

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    rows  = read_submission_csv(csv_path)[:n]

    log.info("=== TOP-%d REASONING REVIEW ===", n)
    log.info("Check: specific facts? JD connection? honest concern? no metadata leak?")
    log.info("")
    for rank, cid, score in rows:
        reasoning = cache.get(cid, "(MISSING — no reasoning generated!)")
        words     = len(reasoning.split())
        has_meta  = contains_internal_metadata(reasoning)
        flag      = " ⚠METADATA" if has_meta else ""
        log.info("  #%d  %s  (%d words)%s", rank, cid, words, flag)
        log.info("  → %s", reasoning)
        log.info("")

    # Stats
    all_r = [cache.get(cid, "") for _, cid, _ in rows]
    wcs   = [len(r.split()) for r in all_r if r]
    meta  = sum(1 for r in all_r if contains_internal_metadata(r))
    missing = sum(1 for _, cid, _ in rows if cid not in cache)
    log.info(
        "Stats: %d reasonings | avg %.0f words | min %d | max %d | metadata leaks: %d | missing: %d",
        len(wcs), sum(wcs)/max(len(wcs),1), min(wcs) if wcs else 0,
        max(wcs) if wcs else 0, meta, missing,
    )



# Main


async def run(csv_path: Path, client: genai.Client, dry_run: bool) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Read submission CSV to get the exact top-100 ranked candidates
    rows = read_submission_csv(csv_path)
    log.info("Loaded %d candidates from %s", len(rows), csv_path)

    top_ids_ordered = [(rank, cid, score) for rank, cid, score in rows]
    target_set      = {cid for _, cid, _ in top_ids_ordered}

    # Load existing cache (backup first)
    existing_cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        existing_cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        # Backup
        import shutil
        shutil.copy2(CACHE_PATH, BACKUP_PATH)
        log.info("Backed up existing cache → %s", BACKUP_PATH)

    # Load live progress cache (resume support)
    live_cache: dict[str, str] = {}
    if LIVE_PATH.exists():
        try:
            live_cache = json.loads(LIVE_PATH.read_text(encoding="utf-8"))
            # Only keep valid non-metadata entries
            live_cache = {
                k: v for k, v in live_cache.items()
                if v and len(v.split()) >= MIN_WORDS and not contains_internal_metadata(v)
            }
            log.info("  Live cache: %d valid entries loaded.", len(live_cache))
        except Exception:
            live_cache = {}

    done_set  = set(live_cache.keys())
    to_gen    = [(rank, cid, score) for rank, cid, score in top_ids_ordered
                 if cid not in done_set]

    log.info(
        "Target: %d candidates | Already done: %d | To generate: %d",
        len(top_ids_ordered), len(done_set), len(to_gen),
    )

    if not to_gen:
        log.info("All %d candidates already have reasonings. Use --review-only to inspect.", len(done_set))
    elif not dry_run:
        # Stream candidates.jsonl to build lookup
        to_gen_set = {cid for _, cid, _ in to_gen}
        lookup: dict[str, dict] = {}
        log.info("Streaming candidates.jsonl…")
        with open(DATA_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    c   = json.loads(line)
                    cid = c.get("candidate_id")
                    if cid in to_gen_set:
                        lookup[cid] = c
                        if len(lookup) == len(to_gen_set):
                            break
                except json.JSONDecodeError:
                    continue
        log.info("  Found %d/%d candidates.", len(lookup), len(to_gen_set))

        # Build work list
        work = []
        for rank, cid, score in to_gen:
            if cid not in lookup:
                log.warning("  [%s] Not found in candidates.jsonl — skipping.", cid)
                continue
            facts = extract_facts(lookup[cid], rank, score)
            work.append(facts)

        # Async generation
        rate_limiter   = AsyncRateLimiter(RPM_TARGET)
        semaphore      = asyncio.Semaphore(MAX_CONCURRENT)
        template_count = 0
        generated      = 0

        log.info(
            "Generating %d reasonings | concurrency=%d | RPM=%d | models: %s",
            len(work), MAX_CONCURRENT, RPM_TARGET, " → ".join(FALLBACK_MODELS),
        )
        log.info("Estimated time: ~%.0f minutes", len(work) / RPM_TARGET)

        async def _gen(facts: dict) -> None:
            nonlocal template_count, generated
            cid, reasoning, used_tmpl = await generate_one(facts, client, rate_limiter, semaphore)
            live_cache[cid] = reasoning
            if used_tmpl:
                template_count += 1
            generated += 1
            # Write live cache after every result
            tmp = LIVE_PATH.with_suffix(".tmp.json")
            tmp.write_text(json.dumps(live_cache, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(LIVE_PATH)

        await async_tqdm.gather(
            *[_gen(f) for f in work],
            desc="Regenerating top-100 reasonings",
            unit="cand",
            dynamic_ncols=True,
            colour="cyan",
        )

        log.info(
            "Done. Generated: %d | LLM success: %d | Templates: %d",
            generated, generated - template_count, template_count,
        )

    # Merge: update existing cache with new top-100 reasonings
    final_cache = dict(existing_cache)   # keep all pre-existing entries
    final_cache.update(live_cache)       # overwrite top-100 with fresh ones

    if dry_run:
        log.info("[DRY RUN] Would update %d entries in reasoning_cache.json", len(live_cache))
        log.info("[DRY RUN] No files written.")
        return

    # Write final cache
    CACHE_PATH.write_text(
        json.dumps(final_cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        "✅  Updated → %s  (%d total entries | %d new/updated)",
        CACHE_PATH, len(final_cache), len(live_cache),
    )

    # Auto-print top-10 for immediate review
    log.info("")
    log.info("=== Auto-review top-10 ===")
    for rank, cid, score in top_ids_ordered[:10]:
        reasoning = final_cache.get(cid, "(MISSING)")
        log.info("  #%d  %s", rank, cid)
        log.info("  → %s", reasoning)
        log.info("")


def main() -> None:
    global MAX_CONCURRENT, RPM_TARGET

    ap = argparse.ArgumentParser(
        description="Regenerate reasoning strings for exactly the top-100 submission candidates.",
    )
    ap.add_argument("--csv",        required=False, default=None,
                    help="Path to submission CSV (default: looks for submission.csv in project root)")
    ap.add_argument("--api-key",    default=os.environ.get("GEMINI_API_KEY"),
                    help="Gemini API key (default: $GEMINI_API_KEY)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Generate but do NOT overwrite reasoning_cache.json")
    ap.add_argument("--review-only",action="store_true",
                    help="Print all 100 reasonings from existing cache; no API calls")
    ap.add_argument("--concurrency",type=int, default=MAX_CONCURRENT,
                    help=f"Max concurrent Gemini calls (default: {MAX_CONCURRENT})")
    ap.add_argument("--rpm",        type=int, default=RPM_TARGET,
                    help=f"Requests per minute target (default: {RPM_TARGET})")
    args = ap.parse_args()

    MAX_CONCURRENT = args.concurrency
    RPM_TARGET     = args.rpm

    # Find CSV
    if args.csv:
        csv_path = Path(args.csv)
    else:
        for candidate in [
            PROJECT_ROOT / "submission.csv",
            PROJECT_ROOT / "artifacts" / "submission.csv",
            PROJECT_ROOT / "output" / "submission.csv",
        ]:
            if candidate.exists():
                csv_path = candidate
                break
        else:
            sys.exit("[ERROR] Could not find submission CSV. Pass --csv <path>")

    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV not found: {csv_path}")

    if args.review_only:
        review(csv_path)
        return

    if not args.api_key:
        sys.exit(
            "[ERROR] No Gemini API key.\n"
            "Set $GEMINI_API_KEY, add to .env, or pass --api-key <key>."
        )
    if not DATA_PATH.exists():
        sys.exit(f"[ERROR] candidates.jsonl not found at {DATA_PATH}")

    client = genai.Client(api_key=args.api_key)

    log.info("Phase 1 · Step 08 — Reasoning Re-generator (top-100 only)")
    log.info("  CSV          : %s", csv_path)
    log.info("  Concurrency  : %d", MAX_CONCURRENT)
    log.info("  RPM target   : %d", RPM_TARGET)
    log.info("  Model chain  : %s", " → ".join(FALLBACK_MODELS))
    log.info("  Word range   : %d–%d", MIN_WORDS, MAX_WORDS)
    log.info("  Dry run      : %s", args.dry_run)

    t0 = time.monotonic()
    asyncio.run(run(csv_path, client, args.dry_run))
    log.info("Total wall time: %.1f min", (time.monotonic() - t0) / 60)


if __name__ == "__main__":
    main()