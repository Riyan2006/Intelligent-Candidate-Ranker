# Intelligent Candidate Ranker
**Intelligent Candidate Discovery & Ranking Challenge**
Participant: `riyansarkar2006_2427`

---

## What this does

A two-phase pipeline that ranks 100,000 candidates against a Senior AI Engineer (Founding Team) job description. The offline phase pre-computes embeddings, LLM scores, career features, and reasoning strings. The fast ranker fuses those pre-computed signals in under 10 seconds on CPU with no network access.

---

## Reproduce the submission (≤ 5 minutes, CPU only, no network)

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./riyansarkar2006_2427.csv
```

All pre-computed artifacts are included in the `artifacts/` folder. The ranking step loads them from disk and does pure numpy math — no API calls, no GPU, no downloads.

**Verified environment:** Windows 11, Python 3.11, 16GB RAM, CPU only. Typical runtime: under 10 seconds for 100K candidates.

---

## Setup

```bash
git clone https://github.com/Riyan2006/Intelligent-Candidate-Ranker.git
cd Intelligent-Candidate-Ranker
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac
pip install -r requirements.txt
```

---

## Project structure

```
├── data/
│   ├── candidates.jsonl          # 100K candidate pool (not in repo — too large)
│   ├── job_description.md        # JD used for ranking
│   └── sample_candidates.json   # First 50 candidates for inspection
│
├── artifacts/                   # Pre-computed artifacts (shipped with repo)
│   ├── embeddings.npy            # 100K × 1024 float32 Voyage embeddings
│   ├── candidate_ids.json        # Ordered list of candidate IDs (parallel to embeddings.npy)
│   ├── jd_embedding.npy          # JD embedding vector (1 × 1024)
│   ├── career_features.parquet   # Structured JD-aware features for all 100K candidates
│   ├── behavioral_scores.parquet # Behavioral signal scores for all 100K candidates
│   ├── llm_scores.json           # LLM rubric scores (top 500 deep-scored, rest = 0)
│   ├── reasoning_cache.json      # Pre-generated reasoning strings for top 100 candidates
│   └── honeypot_ids.txt          # IDs of detected mathematically impossible profiles
│
├── src/
│   ├── loader.py                 # JSONL streaming utility
│   ├── honeypot_detector.py      # Mathematical impossibility detection
│   ├── behavioral_scorer.py      # 23 redrob_signals fields → behavioral score
│   ├── feature_engineer.py       # JD-aware career & skill features
│   ├── embedder.py               # Voyage AI embeddings for all 100K candidates
│   ├── llm_scorer.py             # Gemini Flash deep scoring of top 500
│   └── reasoning_generator_v2.py # Pre-computed reasoning strings for top 100
│
├── rank.py                       # Fast ranker - THE submission script
├── riyansarkar2006_2427.csv      # Final submission CSV
├── validate_submission.py        # Format validator
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

---

## Architecture

### Phase 1 — Offline pre-computation (runs once, takes hours, requires API keys)

All offline steps are already run. Their outputs are in `artifacts/`. You do not need to re-run these to reproduce the submission CSV.

| Step | Script | What it does | Runtime |
|------|--------|-------------|---------|
| 03 | `src/honeypot_detector.py` | Detects ~67–80 mathematically impossible profiles using 3 signals: YOE vs career span, expert skills with 0 duration, single role exceeding total YOE | ~2 min |
| 04 | `src/behavioral_scorer.py` | Scores all 23 `redrob_signals` fields across 4 pillars: availability & recency, responsiveness & reliability, market demand, technical & authenticity | ~3 min |
| 05 | `src/feature_engineer.py` | Computes 11 JD-aware feature groups per candidate: must-have skill match (with description scanning), company type, career trajectory, production evidence, keyword stuffer penalty, red flag detection | ~5 min |
| 06 | `src/embedder.py` | Embeds all 100K candidates using Voyage AI `voyage-4-large` (1024-dim). Requires `VOYAGE_API_KEY`. **~2.5 hours** due to free-tier rate limits (2000 RPM, batches of 128). Checkpoints after every batch — safe to interrupt and resume. | **~2.5 hrs** |
| 07 | `src/llm_scorer.py` | Scores top-500 candidates by embedding similarity using Gemini Flash on 5 rubric dimensions: technical depth, career trajectory, red flag check, shipping evidence, overall fit. Requires `GEMINI_API_KEY`. **~1 hour** at free-tier rate limits (12 RPM). Checkpoints after every call. | **~1 hr** |
| 08 | `src/reasoning_generator_v2.py` | Pre-generates 1-sentence recruiter reasoning for top-100 candidates using Gemini Flash. Includes hallucination verification and rank-aware tone calibration. Requires `GEMINI_API_KEY`. | ~15 min |

**To regenerate all artifacts from scratch:**

```bash
# Set API keys
export VOYAGE_API_KEY="your_voyage_key"
export GEMINI_API_KEY="your_gemini_key"

# Run offline pipeline in order
python src/honeypot_detector.py
python src/behavioral_scorer.py
python src/feature_engineer.py
python src/embedder.py --jd data/job_description.md          # ~2.5 hrs
python src/llm_scorer.py                                      # ~1 hr
python src/reasoning_generator_v2.py --csv riyansarkar2006_2427.csv  # ~15 min
```

> ⚠️ Note: `data/candidates.jsonl` (465MB) is not included in the repo due to GitHub file size limits. Download it from the hackathon dataset and place it at `data/candidates.jsonl` before running any offline scripts. `artifacts/embeddings.npy` (~400MB) is tracked via Git LFS.

---

### Phase 2 — Fast ranker (`rank.py`)

Loads all pre-computed artifacts and produces the ranked CSV in one pass. Zero API calls. Zero GPU.

**Four-signal fusion formula:**
```
final_score = (0.35 × embedding_similarity)
            + (0.30 × llm_rubric_score)
            + (0.20 × must_have_feature_score)
            + (0.15 × behavioral_score)
```

**Post-fusion multipliers:**
- Honeypot candidates: `× 0.01`
- Consulting-only career (TCS/Infosys/Wipro etc.): `× 0.10`
- Red flag penalty from LLM scorer: score zeroed for candidates with `red_flag_check < 3`

**Reasoning:** loaded verbatim from `artifacts/reasoning_cache.json`. Candidates not in cache get a deterministic template string built from their profile fields — no API call at rank time.

---

## Signals used

### Embedding similarity (weight: 0.35)
Voyage AI `voyage-4-large` embeddings (1024-dim) for all 100K candidates and the JD. `task_type="document"` for candidates, `task_type="query"` for the JD. Cosine similarity computed in numpy (~2 seconds for 100K vectors).

### LLM rubric score (weight: 0.30)
Gemini Flash scores the top-500 candidates by embedding similarity on 5 dimensions (0–10 each), weighted as: technical depth (0.30), career trajectory (0.25), red flag check (0.20), shipping evidence (0.15), overall fit (0.10).

### Career feature score (weight: 0.20)
11 structured feature groups computed per candidate:
- Must-have skill match against JD taxonomy (scans both `skills[]` names AND career descriptions)
- Company type scoring (product vs consulting vs IT services)
- Career trajectory and seniority progression
- Production evidence detection (shipping language in descriptions)
- Keyword stuffer penalty (AI skills + non-AI titles + no AI descriptions)
- Red flag detection (consulting-only, out-of-domain dominant, LLM-only wrapper, inactive coder)
- YOE fit, relevant YOE fraction, education tier, location score

### Behavioral score (weight: 0.15)
Derived from all 23 `redrob_signals` fields across 4 pillars:
- **Availability & recency:** `last_active_date`, `open_to_work_flag`, `notice_period_days`
- **Responsiveness:** `recruiter_response_rate`, `avg_response_time_hours`, `interview_completion_rate`, `offer_acceptance_rate`
- **Market demand:** `profile_views_received_30d`, `saved_by_recruiters_30d`, `applications_submitted_30d`, `search_appearance_30d`
- **Technical authenticity:** `github_activity_score`, `profile_completeness_score`, `verified_email/phone/linkedin`

### Honeypot detection
Three non-overlapping mathematical impossibility signals:
- **Signal A:** `years_of_experience > career_span + 8.0 years`
- **Signal B:** 3+ skills simultaneously `proficiency="expert"` AND `duration_months=0`
- **Signal C:** Any single role `duration_months > (yoe × 12) + 6`

---

## AI tools used

- **Claude (Anthropic)** — architecture design, script writing, debugging
- **Gemini Flash** — LLM rubric scoring (Step 07) and reasoning generation (Step 08)
- **Voyage AI** — text embeddings (Step 06)

---

## Compute environment

Windows 11, Python 3.11, Intel CPU, 16GB RAM. No GPU used at any stage of the ranking pipeline. Offline pre-computation used Voyage AI and Gemini APIs (free tier).
