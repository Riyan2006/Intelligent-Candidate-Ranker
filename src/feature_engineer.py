"""
feature_engineer.py — Phase 1 : Step 05
===========================================================
Computes structured, JD-aware features for every candidate.

These features complement semantic embeddings by catching signals that
pure vector similarity misses:
  - A "Marketing Manager" with AI keywords in skills != AI engineer
  - A career at Swiggy/Razorpay (product) > same title at TCS (consulting)
  - An endorsement-rich, NLP-assessed candidate buried in keyword noise
  - A title-chaser vs a domain-progressive career

Output
------
  artifacts/career_features.parquet
  One row per candidate_id with all feature columns defined below.
  Loaded directly by rank.py for the fusion step.

Feature Groups
--------------
  A. Must-Have Skill Match       (0.0 – 1.0)
  B. Nice-to-Have Skill Match    (0.0 – 1.0)
  C. Keyword Stuffer Penalty     (0.0 = clean, 1.0 = stuffer)
  D. Company Type Score          (0.0 – 1.0)
  E. Career Trajectory Score     (0.0 – 1.0)
  F. YOE Fit Score               (0.0 – 1.0)
  G. Education Score             (0.0 – 1.0)
  H. Assessment Score            (0.0 – 1.0)
  I. Red Flag Penalties          (0.0 = clean, 1.0 = disqualified)
  J. Production Evidence Score   (0.0 – 1.0)
  K. Title-Role Alignment Score  (0.0 – 1.0)

  COMPOSITE: must_have_feature_score  (used directly in fusion)
"""

import os
import sys
import json
import re
import math
from datetime import date, datetime
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm



# A.  JD SKILL TAXONOMY
# Each entry is a regex-compatible pattern that matches the skill name (lowercase).

# Highest weight: core must-haves that map directly to JD S1
MUST_HAVE_SKILLS: list[tuple[str, float]] = [
    # Embedding / retrieval systems
    (r"sentence[- ]transformers?",          1.0),
    (r"\bbge\b",                            1.0),
    (r"\be5\b.*embed|embed.*\be5\b",        0.9),
    (r"\bembeddings?\b",                    1.0),
    (r"information[- ]retrieval",           1.0),
    (r"semantic[- ]search",                 1.0),
    (r"vector[- ]search",                   1.0),
    (r"\bdense[- ]retrieval\b",             0.9),
    (r"\bhybrid[- ]search\b",               1.0),
    (r"\bcohere\b",                         0.7),
    (r"\bvoyage\b",                         0.8),
    (r"\bada\b.*embed|embed.*\bada\b",      0.8),
    (r"\bcolbert\b",                        0.9),
    (r"\bdpr\b",                            0.9),
    (r"\bbi[- ]encoder",                    0.9),
    (r"\bcross[- ]encoder",                 0.9),
    (r"\bsplade\b",                         0.8),
    # Vector DBs / ANN indexes
    (r"\bfaiss\b",                          1.0),
    (r"\bmilvus\b",                         1.0),
    (r"\bpinecone\b",                       1.0),
    (r"\bqdrant\b",                         1.0),
    (r"\blancedb\b",                        0.8),
    (r"\bvespa\b",                          0.9),
    (r"\bsolr\b",                           0.7),
    (r"\btypesense\b",                      0.7),
    (r"\bann[- ]index",                     0.8),
    (r"\bhnsw\b",                           0.9),
    (r"\bivf\b",                            0.8),
    (r"\bscann\b",                          0.8),
    (r"\bweaviate\b",                       1.0),
    (r"\bchroma\b",                         0.8),
    (r"\bpgvector\b",                       0.8),
    (r"\bhaystack\b",                       0.8),
    (r"\bopensearch\b",                     0.9),
    (r"\belasticsearch\b",                  0.9),
    # Ranking & evaluation
    (r"learning[- ]to[- ]rank",             1.0),
    (r"\bbm25\b",                           0.9),
    (r"\btf[- ]?idf\b",                     0.7),
    (r"\binverted[- ]index",                0.8),
    (r"\brecall@",                          0.8),
    (r"\bprecision@",                       0.8),
    (r"\bhit[- ]rate\b",                    0.7),
    (r"\boffline.*eval|eval.*offline",      0.8),
    (r"\bonline.*eval|eval.*online",        0.8),
    (r"\ba/b[- ]test",                      0.8),
    (r"\bndcg\b",                           0.9),
    (r"\bmrr\b",                            0.8),
    (r"\bmap\b.*retriev|retriev.*\bmap\b",  0.8),
    (r"\breranking?\b",                     0.9),
    # NLP core
    (r"\bnlp\b",                            1.0),
    (r"natural[- ]language[- ]processing",  1.0),
    (r"hugging[- ]face",                    0.9),
    (r"\btransformers?\b",                  0.8),  # could be electrical – context low weight
    (r"\bbert\b",                           0.9),
    # RAG / LLM integration
    (r"\brag\b",                            0.9),
    (r"\bllms?\b",                          0.8),
    (r"retrieval[- ]augmented",             1.0),
    (r"\bllamaindex\b",                     0.7),
    (r"\bllamaindex\b",                     0.7),
    (r"\bchunking\b",                       0.6),
    (r"\breranker\b",                       0.9),
    (r"\bmonot5\b",                         0.8),
    (r"recommendation[- ]system",           0.9),
]

# Nice-to-have: JD S2 boosters
NICE_TO_HAVE_SKILLS: list[tuple[str, float]] = [
    (r"\blora\b",                           1.0),
    (r"\bqlora\b",                          1.0),
    (r"\bpeft\b",                           1.0),
    (r"fine[- ]tun",                        0.9),
    (r"\bxgboost\b",                        0.8),
    (r"\blightgbm\b",                       0.7),
    (r"neural[- ]rank",                     0.9),
    (r"distributed[- ]training",            0.7),
    (r"large[- ]scale[- ]inference",        0.8),
    (r"\bkubeflow\b",                       0.6),
    (r"\bmlops\b",                          0.7),
    (r"\bprompt[- ]engineering\b",          0.6),
    (r"\bopenai\b",                         0.6),
    (r"\blangchain\b",                      0.6),
    (r"\bvllm\b",                           0.7),
    (r"\btriton\b",                         0.7),
    (r"\bonnx\b",                           0.6),
    (r"\bopen[- ]source\b",                 0.5),
    (r"\bkaggle\b",                         0.4),
]

# Out-of-domain skills: raise a flag if DOMINANT with no NLP signal
OUT_OF_DOMAIN_SKILLS: list[str] = [
    r"computer[- ]vision",
    r"\bcnn\b",
    r"\byolo\b",
    r"image[- ]classif",
    r"object[- ]detect",
    r"image[- ]segment",
    r"speech[- ]recogn",
    r"\basr\b",
    r"text[- ]to[- ]speech",
    r"\btts\b",
    r"robotics",
    r"openCV",
    r"\bgans?\b",
    r"stable[- ]diffusion",
    r"diffusion[- ]model",
    r"reinforcement[- ]learning",
]


# B.  COMPANY CLASSIFICATION

# Tier 0 = pure consulting disqualifier  (explicit in JD S4)
# Tier 1 = premium product / startup     (strongest signal)
# Tier 2 = good product company          (solid signal)
# Tier 3 = mid-tier / mixed              (neutral)
# Tier 4 = IT services / body-shopping   (soft penalty)
# Tier 5 = non-tech                      (strong penalty)

COMPANY_TIERS: dict[str, int] = {
    # Tier 0 – pure consulting disqualifiers (JD explicit)
    "tcs": 0, "infosys": 0, "wipro": 0, "accenture": 0,
    "cognizant": 0, "capgemini": 0,
    # Tier 1 – premium AI/product (best signal)
    "swiggy": 1, "zomato": 1, "razorpay": 1, "phonepe": 1,
    "cred": 1, "meesho": 1, "dream11": 1, "paytm": 1,
    "flipkart": 1, "nykaa": 1, "inmobi": 1, "zoho": 1,
    "freshworks": 1, "mad street den": 1, "aganitha": 1,
    "glance": 1, "unacademy": 1, "upgrad": 1, "vedantu": 1,
    "policybazaar": 1, "pharmeasy": 1, "ola": 1,
    # Tier 2 – known good product / internet
    "pied piper": 2, "initech": 2, "hooli": 2,  # fictional but dataset product cos
    "globex inc": 2,
    # Tier 3 – mid-tier / mixed signals
    "mphasis": 3, "mindtree": 3, "hcl": 3, "tech mahindra": 3,
    "dunder mifflin": 3,  # fictional but not consulting
    # Tier 4 – IT services body-shopping
    # (HCL/Mphasis also here; covered by Tier 3 above — Tier 3 is lenient)
    # Tier 5 – clearly non-tech manufacturers / conglomerates
    "wayne enterprises": 5, "stark industries": 5,
    "acme corp": 5, "globex": 5,
}

INDUSTRY_SCORES: dict[str, float] = {
    "ai/ml":          1.0,
    "fintech":        0.9,
    "e-commerce":     0.85,
    "food delivery":  0.85,
    "saas":           0.85,
    "edtech":         0.8,
    "adtech":         0.8,
    "gaming":         0.8,
    "insurance tech": 0.75,
    "healthtech":     0.75,
    "transportation": 0.75,
    "software":       0.7,
    "it services":    0.35,
    "consulting":     0.2,
    "manufacturing":  0.15,
    "paper products": 0.1,
    "conglomerate":   0.1,
}

CONSULTING_FIRMS = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}


# C.  AI-RELEVANT JOB TITLE TAXONOMY

AI_TITLE_PATTERNS: list[str] = [
    r"machine[- ]learning",
    r"\bnlp\b",
    r"\bml\b[- ]engineer",
    r"ai[- ]engineer",
    r"data[- ]scientist",
    r"deep[- ]learning",
    r"research[- ]scientist",
    r"applied[- ](ml|ai|scientist)",
    r"search[- ]engineer",
    r"recommendation",
    r"ranking[- ]engineer",
    r"information[- ]retrieval",
    r"llm[- ]engineer",
    r"generative[- ]ai",
    r"ml[- ]platform",
    r"mlops[- ]engineer",
    r"computer[- ]vision[- ]engineer",
]

# Titles that are clearly non-AI (keyword stuffers tend to have these)
NON_AI_TITLE_PATTERNS: list[str] = [
    r"operations[- ]manager",
    r"marketing[- ]manager",
    r"\bhr[- ]manager\b",
    r"business[- ]analyst",
    r"content[- ]writer",
    r"civil[- ]engineer",
    r"mechanical[- ]engineer",
    r"\baccountant\b",
    r"graphic[- ]designer",
    r"project[- ]manager",
    r"sales[- ]executive",
    r"customer[- ]support",
    r"\b\.net[- ]developer\b",
    r"\bjava[- ]developer\b",
    r"mobile[- ]developer",
]


# D.  PRODUCTION EVIDENCE KEYWORDS (in career descriptions)

PRODUCTION_KEYWORDS: list[str] = [
    r"in[- ]production",
    r"shipped[- ]to",
    r"\bshipped\b",              # "shipped multiple ranking models"
    r"deployed[- ]to",
    r"\bdeployed\b",
    r"real[- ]users",
    r"at[- ]scale",
    r"serving[- ]\d",
    r"serving[- ]millions",
    r"millions[- ]of[- ]users",
    r"production[- ]system",
    r"production[- ]deploy",
    r"end[- ]to[- ]end",
    r"launched[- ]",
    r"live[- ]in[- ]prod",
    r"went[- ]live",
    r"a/b[- ]test",              # A/B testing = production signal
    r"online[- ]experiment",
    r"rolled[- ]out",
    r"\bserving\b",              # "serving 10M requests"
    r"traffic[- ]",              # "10% of traffic"
]

# Target locations (JD S2 nice-to-have)
TARGET_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi",
    "delhi ncr", "ncr", "gurugram", "gurgaon", "bengaluru", "bangalore"
}

# Assessment keys and their JD weights (from jd_analysis.md §8)
ASSESSMENT_JD_WEIGHTS: dict[str, float] = {
    # Core must-haves
    "nlp":                      1.0,
    "information retrieval":    1.0,
    "recommendation systems":   0.9,
    "sentence transformers":    1.0,
    "faiss":                    1.0,
    "milvus":                   1.0,
    "pinecone":                 1.0,
    "qdrant":                   1.0,
    "opensearch":               0.9,
    "elasticsearch":            0.9,
    "vector search":            1.0,
    "embeddings":               1.0,
    # High boosters
    "fine-tuning llms":         0.8,
    "llms":                     0.75,
    "rag":                      0.8,
    "peft":                     0.8,
    "lora":                     0.8,
    "qlora":                    0.8,
    "learning to rank":         0.85,
    "xgboost":                  0.6,
    "feature engineering":      0.6,
    "mlops":                    0.5,
    "mlflow":                   0.4,
    "weights & biases":         0.4,
    "data science":             0.4,
    "prompt engineering":       0.5,
    "langchain":                0.5,
    # Out-of-domain (low or penalty if no NLP score)
    "image classification":     -0.1,
    "object detection":         -0.1,
    "yolo":                     -0.1,
    "cnn":                      -0.1,
    "computer vision":          -0.1,
    "speech recognition":       -0.15,
    "tts":                      -0.15,
    "asr":                      -0.15,
    "gans":                     -0.1,
    "diffusion models":         -0.1,
    "reinforcement learning":   -0.1,
    "opencv":                   -0.1,
    "bentoml":                   0.2,  # deployment tool, mild positive
}


# HELPERS


def _match_patterns(text: str, patterns: list) -> list[tuple[str, float]]:
    """Return list of (pattern, weight) that match lowercased text."""
    t = text.lower()
    if isinstance(patterns[0], str):
        return [(p, 1.0) for p in patterns if re.search(p, t)]
    return [(p, w) for p, w in patterns if re.search(p, t)]


def _skill_score(skills: list[dict], patterns: list[tuple[str, float]]) -> float:
    """
    Score how well a candidate's skills cover the given pattern list.
    Weights proficiency level and duration:
      expert   = 1.0
      advanced = 0.75
      intermediate = 0.5
      beginner = 0.25
    Duration weight: min(1.0, duration_months / 24)  [fully weighted at 2+ years]
    Returns normalised score 0-1.
    """
    PROF_WEIGHT = {"expert": 1.0, "advanced": 0.75, "intermediate": 0.5, "beginner": 0.25}
    matched_weights: list[float] = []
    total_pattern_weight = sum(w for _, w in patterns)

    for pat, pat_weight in patterns:
        best = 0.0
        for s in skills:
            name = s.get("name", "").lower()
            if re.search(pat, name):
                prof = PROF_WEIGHT.get(s.get("proficiency", "beginner"), 0.25)
                dur  = min(1.0, s.get("duration_months", 0) / 24.0)
                # Endorsements add a credibility boost (capped at 0.15)
                end  = min(0.15, s.get("endorsements", 0) / 100.0)
                val  = pat_weight * (prof * 0.6 + dur * 0.25 + end * 0.15)
                best = max(best, val)
        matched_weights.append(best)

    if total_pattern_weight == 0:
        return 0.0
    return min(1.0, sum(matched_weights) / total_pattern_weight)


def _title_is_ai(title: str) -> bool:
    t = title.lower()
    return any(re.search(p, t) for p in AI_TITLE_PATTERNS)


def _title_is_non_ai(title: str) -> bool:
    t = title.lower()
    return any(re.search(p, t) for p in NON_AI_TITLE_PATTERNS)


def _parse_year_month(date_str: str | None) -> float | None:
    if not date_str or len(date_str) < 7:
        return None
    try:
        return int(date_str[:4]) + int(date_str[5:7]) / 12.0
    except (ValueError, IndexError):
        return None



# FEATURE FUNCTIONS


def feat_must_have_skill_score(skills: list[dict], career: list[dict]) -> float:
    """Score against skills[] AND career description text combined."""
    # Current: only checks skill names
    skill_score = _skill_score(skills, MUST_HAVE_SKILLS)

    # New: also check all career descriptions
    all_desc = " ".join(r.get("description", "") for r in career).lower()
    desc_hits = sum(
        1 for pat, _ in MUST_HAVE_SKILLS
        if re.search(pat, all_desc)
    )
    desc_score = min(1.0, desc_hits / 8)  # 8+ description hits = full score

    # Combined: skills are stronger signal (self-reported + endorsed)
    # but descriptions catch things people forgot to list as skills
    return round(min(1.0, skill_score * 0.7 + desc_score * 0.3), 4)


def feat_nice_to_have_skill_score(skills: list[dict]) -> float:
    """Group B — Coverage of nice-to-have / booster skills."""
    return _skill_score(skills, NICE_TO_HAVE_SKILLS)


def feat_keyword_stuffer_penalty(
    skills: list[dict],
    career: list[dict],
    profile: dict,
) -> float:
    """
    Group C — Keyword stuffer penalty.

    A keyword stuffer has many JD-relevant skills but:
      (a) Their job titles are all non-AI (Operations, Marketing, HR, etc.)
      (b) Their career descriptions contain no AI/ML context
      (c) Their skill endorsements for AI skills are suspiciously low

    Returns 0.0 = clean,  1.0 = almost certainly a stuffer.
    """
    skill_names = [s.get("name", "").lower() for s in skills]

    # Count how many core JD skills appear in the skills list
    jd_skill_hits = sum(
        1 for name in skill_names
        if any(re.search(pat, name) for pat, _ in MUST_HAVE_SKILLS)
    )

    if jd_skill_hits < 3:
        return 0.0  # Not enough AI skills to even be a stuffer

    # Check if ALL career titles are non-AI
    all_titles = [r.get("title", "") for r in career]
    all_titles.append(profile.get("current_title", ""))
    non_ai_title_count = sum(1 for t in all_titles if _title_is_non_ai(t))
    ai_title_count     = sum(1 for t in all_titles if _title_is_ai(t))

    if ai_title_count > 0:
        return 0.0  # Has at least one AI title → legitimate

    # All titles are non-AI — check description for AI context
    all_desc = " ".join(r.get("description", "") for r in career).lower()
    desc_ai_terms = [
        r"embed", r"retriev", r"nlp", r"vector", r"ranking", r"recommend",
        r"machine learning", r"deep learning", r"\bllm\b", r"neural",
        r"train", r"model", r"inference", r"faiss", r"transformer",
    ]
    desc_hits = sum(1 for p in desc_ai_terms if re.search(p, all_desc))

    # Check endorsements for the AI skills they claim
    ai_skill_endorsements = [
        s.get("endorsements", 0)
        for s in skills
        for pat, _ in MUST_HAVE_SKILLS
        if re.search(pat, s.get("name", "").lower())
    ]
    avg_endorsement = (
        sum(ai_skill_endorsements) / len(ai_skill_endorsements)
        if ai_skill_endorsements else 0
    )

    # High AI skills + no AI titles + no AI descriptions + low endorsements = stuffer
    title_penalty   = 1.0 if non_ai_title_count == len(all_titles) else 0.0
    desc_penalty    = max(0.0, 1.0 - desc_hits / 5.0)       # 5+ desc hits = clean
    endorse_penalty = max(0.0, 1.0 - avg_endorsement / 10.0) # 10 avg = clean

    stuffer_score = (title_penalty * 0.5 + desc_penalty * 0.3 + endorse_penalty * 0.2)
    return round(min(1.0, stuffer_score), 4)


def feat_company_type_score(career: list[dict]) -> tuple[float, bool, bool]:
    """
    Group D — Company type score, consulting flag, product flag.

    Returns (score 0–1, is_consulting_only, has_product_company)
    """
    if not career:
        return 0.3, False, False

    role_scores: list[float] = []
    companies_lower = [r.get("company", "").lower() for r in career]
    industries_lower = [r.get("industry", "").lower() for r in career]

    consulting_count = sum(
        1 for co in companies_lower
        if any(firm in co for firm in CONSULTING_FIRMS)
    )
    is_consulting_only = (consulting_count == len(career) and len(career) > 0)

    has_product = False
    for role in career:
        co    = role.get("company", "").lower()
        ind   = role.get("industry", "").lower()
        dur   = role.get("duration_months", 0)

        # Look up company tier
        tier = None
        for known_co, t in COMPANY_TIERS.items():
            if known_co in co:
                tier = t
                break

        # Fall back to industry scoring
        if tier is None:
            ind_score = INDUSTRY_SCORES.get(ind, 0.5)
        else:
            tier_map = {0: 0.1, 1: 1.0, 2: 0.85, 3: 0.6, 4: 0.4, 5: 0.15}
            ind_score = tier_map.get(tier, 0.5)
            if tier <= 2:
                has_product = True

        # Duration weight: longer tenure at a company = more signal
        dur_weight = min(1.0, dur / 24.0)  # fully weighted at 2+ year roles
        role_scores.append(ind_score * (0.6 + 0.4 * dur_weight))

    if not role_scores:
        return 0.3, False, False

    # Recent companies weighted more (last 2 roles count double)
    weights = [1.0] * len(role_scores)
    if len(role_scores) >= 2:
        weights[-1] = 2.0
        weights[-2] = 1.5
    weighted_sum = sum(s * w for s, w in zip(role_scores, weights))
    total_weight = sum(weights)
    score = weighted_sum / total_weight

    return round(min(1.0, score), 4), is_consulting_only, has_product


def feat_career_trajectory_score(career: list[dict]) -> tuple[float, bool]:
    """
    Group E — Career trajectory: seniority progression + domain consistency.

    Returns (score 0–1, is_title_chaser)

    Title-chaser: 3+ companies in 5 years where each move was to a
    non-AI title (random domain-hopping, not progression).
    """
    if not career:
        return 0.3, False

    SENIORITY_RANK = {
        "intern": 0, "trainee": 0, "fresher": 0,
        "associate": 1, "junior": 1,
        "engineer": 2, "developer": 2, "analyst": 2,
        "senior": 3, "lead": 3, "staff": 3,
        "principal": 4, "architect": 4, "manager": 4,
        "director": 5, "vp": 6, "head": 5, "chief": 6,
    }

    def _seniority(title: str) -> int:
        t = title.lower()
        best = 2  # default mid-level
        for kw, rank in SENIORITY_RANK.items():
            if kw in t:
                best = max(best, rank)
        return best

    ranks = [_seniority(r.get("title", "")) for r in career]

    # Progression: is seniority going up or at least flat?
    if len(ranks) >= 2:
        diffs = [ranks[i+1] - ranks[i] for i in range(len(ranks)-1)]
        progression_score = min(1.0, max(0.0, (sum(diffs) + len(diffs)) / (2 * len(diffs))))
    else:
        progression_score = 0.5

    # AI domain consistency: what fraction of roles had AI/ML titles?
    ai_role_count  = sum(1 for r in career if _title_is_ai(r.get("title", "")))
    ai_consistency = ai_role_count / max(1, len(career))

    # Title-chaser detection
    # Use parsed dates for accurate window; fall back to duration_months
    recent_window = 5 * 12  # 5 years in months
    recent_roles = []
    cumulative = 0
    for r in reversed(career):
        dur = r.get("duration_months", 0)
        cumulative += dur
        if cumulative <= recent_window:
            recent_roles.append(r)
        else:
            break

    # Title-chaser: ≥3 employers in 5yr window with non-AI titles
    # (CAND_0000031 has 4 short AI roles → NOT a title-chaser despite short tenures)
    non_ai_recent = sum(
        1 for r in recent_roles
        if _title_is_non_ai(r.get("title", "")) and r.get("duration_months", 0) <= 18
    )
    is_title_chaser = (non_ai_recent >= 3 and len(recent_roles) >= 4)

    title_chaser_penalty = 0.3 if is_title_chaser else 0.0

    score = (
        progression_score * 0.35
        + ai_consistency  * 0.50
        + (1.0 - title_chaser_penalty) * 0.15
    )
    return round(min(1.0, score), 4), is_title_chaser


def feat_yoe_fit_score(yoe: float) -> float:
    """
    Group F — How well does YOE fit the JD target range?

    JD: 5–9 years total, sweet spot 6–8 years.
    """
    if yoe < 2:
        return 0.05
    if 6 <= yoe <= 8:
        return 1.0                   # sweet spot
    if 5 <= yoe < 6:
        return 0.85
    if 8 < yoe <= 9:
        return 0.85
    if 4 <= yoe < 5:
        return 0.65
    if 9 < yoe <= 12:
        return 0.70
    if 2 <= yoe < 4:
        return 0.40
    if yoe > 12:
        return 0.55                  # over-experienced, still possible
    return 0.3


def feat_relevant_yoe(career: list[dict], yoe: float) -> float:
    """
    Group F (extra) — What fraction of total YOE was spent in AI/ML roles?
    """
    ai_months = sum(
        r.get("duration_months", 0)
        for r in career
        if _title_is_ai(r.get("title", ""))
        or INDUSTRY_SCORES.get(r.get("industry", "").lower(), 0) >= 0.75
    )
    total_months = max(1.0, yoe * 12)
    return round(min(1.0, ai_months / total_months), 4)


def feat_education_score(education: list[dict]) -> float:
    """
    Group G — Education tier and field relevance.

    tier_1 → strong boost, tier_2 → neutral, tier_3 → slight penalty,
    tier_4/unknown → stronger penalty.
    CS/IT/EE/Stats field → boost; unrelated field → penalty.
    """
    if not education:
        return 0.35

    TIER_SCORES = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.5, "tier_4": 0.35, "unknown": 0.4}
    RELEVANT_FIELDS = {
        r"computer",r"software",r"information",r"data",
        r"electrical",r"electronics",r"statistics",r"mathematics",
        r"machine[- ]learning",r"artificial[- ]intelligence",
    }

    # Take the most recent / highest degree
    best_edu = sorted(education, key=lambda e: e.get("end_year", 0), reverse=True)[0]
    tier_score  = TIER_SCORES.get(best_edu.get("tier", "unknown"), 0.4)
    field       = best_edu.get("field_of_study", "").lower()
    field_match = any(re.search(p, field) for p in RELEVANT_FIELDS)
    field_score = 0.9 if field_match else 0.5

    # PhD/M.Tech with relevant field → bonus
    degree = best_edu.get("degree", "").lower()
    degree_bonus = 0.1 if any(x in degree for x in ["ph.d", "phd", "m.tech", "m.s.", "m.e."]) else 0.0

    return round(min(1.0, tier_score * 0.4 + field_score * 0.5 + degree_bonus), 4)


def feat_assessment_score(redrob_signals: dict) -> tuple[float, float]:
    """
    Group H — Objective skill assessment scores.

    Returns (weighted_assessment_score, nlp_raw_score).
    Uses jd_analysis.md §8 weight table.

    Out-of-domain penalty only applied if candidate has NO NLP/IR score
    (CV-only or speech-only candidates who scored badly on the core domain).
    """
    scores: dict[str, float] = redrob_signals.get("skill_assessment_scores", {})
    if not scores:
        return 0.0, 0.0

    # Normalise keys
    scores_lower = {k.lower(): v for k, v in scores.items()}

    nlp_raw = max(
        scores_lower.get("nlp", 0),
        scores_lower.get("information retrieval", 0),
        scores_lower.get("sentence transformers", 0),
        scores_lower.get("embeddings", 0),
    )
    has_core_score = nlp_raw > 0

    weighted_sum  = 0.0
    total_abs_w   = 0.0
    for key, raw_score in scores_lower.items():
        w = ASSESSMENT_JD_WEIGHTS.get(key, 0.0)
        if w < 0 and has_core_score:
            w = 0  # suppress out-of-domain penalty if they also have NLP score
        weighted_sum  += (raw_score / 100.0) * abs(w) * (1 if w >= 0 else -1)
        total_abs_w   += abs(w)

    if total_abs_w == 0:
        return 0.0, nlp_raw / 100.0

    score = weighted_sum / total_abs_w
    return round(min(1.0, max(0.0, score)), 4), round(nlp_raw / 100.0, 4)


def feat_red_flag_score(
    career: list[dict],
    skills: list[dict],
    profile: dict,
    redrob_signals: dict,
) -> tuple[float, dict]:
    """
    Group I — Red flag composite penalty.
    Returns (penalty 0–1, flag_breakdown dict).

    0.0 = clean, 1.0 = severely disqualified.
    """
    flags: dict[str, bool] = {}

    # 1. Pure consulting career (JD §4 explicit disqualifier)
    _, is_consulting, has_product = feat_company_type_score(career)
    flags["consulting_only"]   = is_consulting
    flags["has_product_co"]    = has_product

    # 2. Title-chaser
    _, is_title_chaser = feat_career_trajectory_score(career)
    flags["title_chaser"] = is_title_chaser

    # 3. Out-of-domain dominant (CV/speech with zero NLP signal)
    skill_names_lower = [s.get("name", "").lower() for s in skills]
    ood_count  = sum(1 for p in OUT_OF_DOMAIN_SKILLS
                     for n in skill_names_lower if re.search(p, n))
    core_count = sum(1 for pat, _ in MUST_HAVE_SKILLS
                     for n in skill_names_lower if re.search(pat, n))
    # If out-of-domain skills dominate AND core JD skills are scarce
    flags["out_of_domain_dominant"] = (ood_count >= 3 and core_count < 2)

    # 4. Functionally unavailable (inactive + low response rate)
    response_rate = redrob_signals.get("recruiter_response_rate", 1.0)
    last_active   = redrob_signals.get("last_active_date", "2026-01-01")
    try:
        days_inactive = (date(2026, 6, 3) - date.fromisoformat(last_active)).days
    except (ValueError, TypeError):
        days_inactive = 0
    flags["functionally_unavailable"] = (response_rate <= 0.05 or days_inactive > 180)

    # 5. Inactive coder: github score = -1 (unlinked) with seniority ≥ 5yr and non-AI title
    github = redrob_signals.get("github_activity_score", 0)
    current_title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    flags["inactive_coder"] = (
        github == -1
        and yoe >= 5
        and _title_is_non_ai(current_title)
        and not _title_is_ai(current_title)
    )

    # 6. LLM-only wrapper (< 12 months of NLP/IR skills across all skills)
    total_nlp_months = sum(
        s.get("duration_months", 0)
        for s in skills
        for pat, _ in MUST_HAVE_SKILLS
        if re.search(pat, s.get("name", "").lower())
    )
    recent_llm_only_skills = [
        s for s in skills
        if re.search(r"\blangchain\b|\bopenai\b|\bgpt\b", s.get("name", "").lower())
        and s.get("duration_months", 0) <= 12
    ]
    flags["llm_only_wrapper"] = (total_nlp_months < 12 and len(recent_llm_only_skills) >= 2)

    # Compute composite penalty
    penalty = 0.0
    penalty += 0.40 if flags["consulting_only"]          else 0.0
    penalty += 0.25 if flags["out_of_domain_dominant"]   else 0.0
    penalty += 0.20 if flags["title_chaser"]              else 0.0
    penalty += 0.20 if flags["functionally_unavailable"]  else 0.0
    penalty += 0.15 if flags["inactive_coder"]            else 0.0
    penalty += 0.15 if flags["llm_only_wrapper"]          else 0.0

    return round(min(1.0, penalty), 4), flags


def feat_production_evidence_score(career: list[dict]) -> float:
    """
    Group J — Does career history show evidence of shipping to production?

    Checks descriptions for production-deployment language.
    JD values "shippers over researchers."
    """
    if not career:
        return 0.0

    all_desc = " ".join(r.get("description", "") for r in career).lower()
    hits = sum(1 for p in PRODUCTION_KEYWORDS if re.search(p, all_desc))

    # Also check for shipped systems described in AI context
    ai_shipped = sum(
        1 for r in career
        if _title_is_ai(r.get("title", ""))
        and any(re.search(p, r.get("description", "").lower())
                for p in PRODUCTION_KEYWORDS)
    )
    # Normalize: 3+ hits = full score; ai_shipped roles add extra
    base  = min(1.0, hits / 3.0)
    bonus = min(0.3, ai_shipped * 0.15)
    return round(min(1.0, base + bonus), 4)


def feat_title_role_alignment(profile: dict, career: list[dict]) -> float:
    """
    Group K — Does the current title match the AI/ML domain?

    High for ML/NLP/Search engineers at product companies.
    Low for Operations/Marketing/HR at any company.
    """
    current_title = profile.get("current_title", "")
    if _title_is_ai(current_title):
        return 1.0
    if _title_is_non_ai(current_title):
        return 0.1

    # Ambiguous titles (Software Engineer, Data Engineer, etc.) — check career context
    ai_roles_count = sum(1 for r in career if _title_is_ai(r.get("title", "")))
    ratio = ai_roles_count / max(1, len(career))
    return round(0.3 + ratio * 0.5, 4)


def feat_location_score(profile: dict, redrob_signals: dict) -> float:
    """
    Location + relocation bonus (JD §2 nice-to-have).
    """
    location = profile.get("location", "").lower()
    country  = profile.get("country", "").lower()
    willing  = redrob_signals.get("willing_to_relocate", False)

    in_target = any(city in location for city in TARGET_LOCATIONS)
    in_india  = (country == "india")

    if in_target:
        return 1.0
    if in_india and willing:
        return 0.7
    if in_india:
        return 0.5
    if willing:
        return 0.4
    return 0.2



# COMPOSITE SCORE


def compute_must_have_feature_score(row: dict) -> float:
    """
    Produce the final composite feature score used in rank.py fusion.

    This score represents the structural / JD-aware signal.
    It is combined with embedding similarity, LLM score, and behavioral score.

    Internal weights tuned to JD emphasis (§1–§4 of jd_analysis.md).
    """
    must_have  = row["must_have_skill_score"]
    nice       = row["nice_to_have_skill_score"]
    company    = row["company_type_score"]
    trajectory = row["career_trajectory_score"]
    yoe_fit    = row["yoe_fit_score"]
    relevant_y = row["relevant_yoe_ratio"]
    education  = row["education_score"]
    assessment = row["assessment_score"]
    production = row["production_evidence_score"]
    title_aln  = row["title_role_alignment"]
    location   = row["location_score"]
    stuffer    = row["keyword_stuffer_penalty"]
    red_flag   = row["red_flag_penalty"]

    raw = (
        must_have   * 0.22   # Core JD skills — highest weight
        + assessment  * 0.15 # Objective proof (beats self-reported skills)
        + company     * 0.12 # Product company = strong signal
        + trajectory  * 0.10 # Growing AI career
        + production  * 0.10 # Ships, doesn't just research
        + title_aln   * 0.08 # Title-role match
        + yoe_fit     * 0.07 # YOE in target range
        + relevant_y  * 0.06 # AI-relevant YOE fraction
        + nice        * 0.04 # Nice-to-have skill coverage
        + education   * 0.03 # Education tier
        + location    * 0.03 # Geographic fit
    )

    # Apply multiplicative penalties
    stuffer_mult   = 1.0 - (stuffer   * 0.70)  # max 70% reduction for stuffers
    red_flag_mult  = 1.0 - (red_flag  * 0.60)  # max 60% reduction for red flags

    return round(max(0.0, min(1.0, raw * stuffer_mult * red_flag_mult)), 6)



# MAIN PROCESSING


def engineer_features(candidate: dict) -> dict:
    """Run all feature functions on a single candidate. Returns feature dict."""
    profile  = candidate.get("profile", {})
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])
    edu      = candidate.get("education", [])
    signals  = candidate.get("redrob_signals", {})
    yoe      = float(profile.get("years_of_experience", 0))
    cid      = candidate.get("candidate_id", "UNKNOWN")

    #  compute
    must_have_score          = feat_must_have_skill_score(skills, career)
    nice_score               = feat_nice_to_have_skill_score(skills)
    stuffer_penalty          = feat_keyword_stuffer_penalty(skills, career, profile)
    company_score, consulting_only, has_product = feat_company_type_score(career)
    traj_score, title_chaser = feat_career_trajectory_score(career)
    yoe_fit                  = feat_yoe_fit_score(yoe)
    relevant_yoe             = feat_relevant_yoe(career, yoe)
    edu_score                = feat_education_score(edu)
    assessment_score, nlp_raw= feat_assessment_score(signals)
    red_flag, flags          = feat_red_flag_score(career, skills, profile, signals)
    production_score         = feat_production_evidence_score(career)
    title_alignment          = feat_title_role_alignment(profile, career)
    location_score           = feat_location_score(profile, signals)

    row = dict(
        candidate_id              = cid,
        must_have_skill_score     = must_have_score,
        nice_to_have_skill_score  = nice_score,
        keyword_stuffer_penalty   = stuffer_penalty,
        company_type_score        = company_score,
        is_consulting_only        = consulting_only,
        has_product_company       = has_product,
        career_trajectory_score   = traj_score,
        is_title_chaser           = title_chaser,
        yoe_fit_score             = yoe_fit,
        relevant_yoe_ratio        = relevant_yoe,
        education_score           = edu_score,
        assessment_score          = assessment_score,
        nlp_assessment_raw        = nlp_raw,
        red_flag_penalty          = red_flag,
        flag_consulting_only      = flags.get("consulting_only", False),
        flag_title_chaser         = flags.get("title_chaser", False),
        flag_out_of_domain        = flags.get("out_of_domain_dominant", False),
        flag_unavailable          = flags.get("functionally_unavailable", False),
        flag_inactive_coder       = flags.get("inactive_coder", False),
        flag_llm_only_wrapper     = flags.get("llm_only_wrapper", False),
        production_evidence_score = production_score,
        title_role_alignment      = title_alignment,
        location_score            = location_score,
    )

    row["must_have_feature_score"] = compute_must_have_feature_score(row)
    return row


def run(dataset_path: str, output_path: str) -> None:
    print("⚙️  Feature Engineering Pipeline")
    print(f"   Input  : {dataset_path}")
    print(f"   Output : {output_path}\n")

    rows: list[dict] = []
    errors = 0

    with open(dataset_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    for line in tqdm(lines, desc="Engineering features", unit="cand"):
        line = line.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
            rows.append(engineer_features(candidate))
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"   ⚠ Error on {line[:40]}: {e}")

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Summary
    print(f"\n✅  Done — {len(df):,} candidates processed, {errors} errors")
    print(f"\n📊  Feature summary (mean values):")
    numeric_cols = [c for c in df.columns if df[c].dtype in [float, "float64", "float32"]]
    for col in numeric_cols:
        print(f"   {col:40s}  mean={df[col].mean():.3f}  max={df[col].max():.3f}")
    print(f"\n🚩  Flag counts:")
    flag_cols = [c for c in df.columns if c.startswith("flag_") or c in
                 ["is_consulting_only","is_title_chaser","has_product_company"]]
    for col in flag_cols:
        count = df[col].sum()
        pct   = count / len(df) * 100
        print(f"   {col:40s}  {int(count):,} ({pct:.1f}%)")
    print(f"\n💾  Saved to: {output_path}")


if __name__ == "__main__":
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    dataset_path = os.path.join(project_root, "data", "candidates.jsonl")
    output_path  = os.path.join(project_root, "artifacts", "career_features.parquet")

    run(dataset_path, output_path)