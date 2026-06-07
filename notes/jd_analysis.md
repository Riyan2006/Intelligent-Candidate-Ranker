# Job Description Analysis & Feature Mapping
**Role:** Senior AI Engineer (Founding Team) — Redrob AI  
**Target Profile:** 5–9 years total (sweet spot: 6–8 years total, 4–5 years applied ML/AI at product companies)

---

## 1. Hard Requirements (Must-Haves)

- **Embeddings & Retrieval:** Deployed, production experience with embedding-based retrieval systems — sentence-transformers, BGE, E5, OpenAI embeddings, or equivalent. Must have handled embedding drift, index refresh, retrieval-quality regression in production. *Not* someone who called an API once.
- **Vector Infrastructure:** Production operational experience with vector databases or hybrid search — Pinecone, Milvus, Qdrant, FAISS, OpenSearch, Elasticsearch, or similar. The specific technology doesn't matter; the operational depth does.
- **Python:** Strong, production-grade code quality. This role writes code — that's stated explicitly.
- **Evaluation Rigor:** Hands-on experience designing ranking evaluation frameworks: NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If they've never thought about *how to measure* a ranking system, the role will be painful.

---

## 2. Score Boosters (Nice-to-Haves)

- LLM fine-tuning experience: LoRA, QLoRA, PEFT
- Learning-to-rank models: XGBoost-based or neural rankers
- Domain knowledge in HR-tech / recruiting tech / marketplaces
- Background in distributed systems or large-scale inference optimization
- Open-source contributions in the AI/ML space
- **Geographic preference:** Pune, Noida, Hyderabad, Mumbai, Delhi NCR (Tier-1 Indian cities)
- **Notice period:** <30 days = high boost (buyout available); 30–60 days = neutral; >60 days = significant penalty
- Willing to travel quarterly for offsites

---

## 3. High Score Boosters (Strong Signals)

- 6–8 years total, with 4–5 of those years in applied ML/AI roles at *product companies* (not IT services)
- Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale
- Strong opinions — and actual evidence — on retrieval (hybrid vs dense), evaluation (offline vs online), LLM integration (when to fine-tune vs prompt)
- Located in or willing to relocate to Noida or Pune
- Active on Redrob platform, or clear signal of being in the job market (LinkedIn active, recent applications, low response time)

---

## 4. Disqualifiers & Penalties (Red Flags)

- **Consulting-Only Career:** Entire career at TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini — no product company experience at all. This is an explicit disqualifier in the JD.
- **Pure Researchers:** Academic labs or research-only roles with no production deployment history whatsoever.
- **LLM-Only Wrappers:** AI experience limited to recent (<12 months) LangChain/OpenAI API stitching, with no pre-LLM ML production history.
- **Out of Domain:** Primary expertise in Computer Vision, Speech, or Robotics with no demonstrated NLP/IR depth. (High image classification or speech recognition scores with no NLP score = penalty, not boost.)
- **Title-Chasers:** Job-hopping pattern — 3+ companies in 5 years, each move for a title bump rather than scope.
- **Inactive/Unavailable:** Not logged in for months, recruiter response rate ≤5%, very high avg response time. These candidates are functionally unavailable regardless of their skills.
- **Inactive Coder:** Senior engineer who hasn't written production code in 18+ months due to moving into pure architecture/tech-lead roles.
- **No External Validation:** 5+ years exclusively on closed-source proprietary systems with no external signal — no papers, talks, or open-source. Can't evaluate them.

---

## 5. Honeypot Patterns (Mathematically Impossible Profiles)

The dataset contains ~67 deliberately planted impossible profiles. These must be detected and heavily penalized before any scoring. Three clean, non-overlapping signals cover all of them:

### Signal A — YOE vs Actual Career Span
`profile.years_of_experience` claims significantly more experience than the candidate's actual career history permits.  
- Compute: `career_span = 2026 - earliest_role_start_date` (from `career_history`)
- Flag if: `yoe > career_span + 8.0 years` (8yr buffer covers all legitimate edge cases: freelance, internships, part-time during education)
- Example: Claims 13.7yr YOE but earliest career role started 0.8 years ago. Impossible.
- **Catches 25 honeypots.**

### Signal B — Expert Skill with Zero Months of Use
`skills[].proficiency = "expert"` combined with `skills[].duration_months = 0` is a direct internal contradiction. You cannot be expert at something you have never used.  
- Flag if: 3+ skills simultaneously have `proficiency = "expert"` AND `duration_months = 0`
- Threshold of 3 prevents penalizing single data-entry errors.
- Example: 5 skills listed as expert, all with 0 months used. Deliberately constructed.
- **Catches 21 honeypots.**

### Signal C — Single Role Duration Exceeds Total YOE
A single `career_history` entry has `duration_months` greater than the candidate's total stated experience (in months).  
- Flag if: any `role.duration_months > (yoe * 12) + 6` (6-month buffer for rounding)
- Example: Claims 9.9yr total experience, but one role lasted 166 months (13.8yr). Arithmetically impossible.
- **Catches 21 honeypots.**

**Total: 67 unique honeypots (zero overlap between signals).**

> ⚠️ What NOT to flag as honeypots:
> - Candidates with postgraduate degrees (M.Tech, PhD) whose `earliest_edu_end_year` is recent — they may have had careers before returning to study. Education years alone are unreliable.
> - Candidates with more YOE than their degree implies — India has mature entry, part-time study, and diploma pathways.
> - Any YOE-vs-education comparison. These produce thousands of false positives because education data in this dataset is not structured to reliably determine career start.

---

## 6. Hackathon Dataset Traps (Explicitly Stated in JD)

The JD includes a direct note to hackathon participants explaining what not to do:

- **Keyword stuffing trap:** Candidates with 15+ AI keywords in the skills section but career history showing only non-technical roles (Marketing Manager, Accountant, etc.). A pure keyword match would rank these highly — wrong.
- **"Tier 5" signal:** Great plain-language profiles with real ML/AI work described, but no buzzwords (no "RAG", "Pinecone", "LLM"). These should rank high. An embedding system captures this naturally; keyword matching won't.
- **Inactive but perfect-on-paper:** A candidate who matches everything but hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, unavailable. Behavioral signals must down-weight them.
- **Title vs substance:** Candidates whose `current_title` says "Marketing Manager" or "Content Writer" but skills list includes "NLP", "FAISS", "Fine-tuning LLMs" — the title was modified; skills were not.

---

## 7. Redrob Behavioral Signals — All 23 Fields Mapped

All signals live in `redrob_signals`. They translate into four operational pillars.

### Pillar A — Availability & Recency (Highest Priority)
| Field | Signal | Scoring Strategy |
|---|---|---|
| `last_active_date` | Days since last login | Active this week → 1.0; inactive 6+ months → 0.1 (scale linearly) |
| `open_to_work_flag` | Explicitly seeking | True = base availability multiplier; False = heavy down-weight |
| `notice_period_days` | Logistics readiness | <30 days → +0.10 boost; 30–60 → neutral; >60 → −0.10 penalty |

### Pillar B — Responsiveness & Reliability
| Field | Signal | Scoring Strategy |
|---|---|---|
| `recruiter_response_rate` | Will they reply? | ≤0.05 → heavy penalty (functionally unavailable); normalize 0–1 |
| `avg_response_time_hours` | How fast? | Lower = higher score; normalize inverse |
| `interview_completion_rate` | Shows up | High = precision protector for our NDCG metrics |
| `offer_acceptance_rate` | Intent to join | −1 = no history → treat as neutral 0.5; high values = true intent |

### Pillar C — Market Demand & Engagement
| Field | Signal | Scoring Strategy |
|---|---|---|
| `profile_views_received_30d` | Market interest | Normalize; high = market validation |
| `search_appearance_30d` | Recruiter finds them | Normalize |
| `saved_by_recruiters_30d` | Outbound validation | High = multiple recruiters independently found value; strong proxy |
| `applications_submitted_30d` | Active search | High = genuinely in job market right now |
| `connection_count` | Network depth | Normalize; mild signal |

### Pillar D — Technical & Authenticity
| Field | Signal | Scoring Strategy |
|---|---|---|
| `github_activity_score` | Writes code | "This role writes code." −1 (unlinked) → treat as 0; 0–100 normalize |
| `profile_completeness_score` | Effort and intent | Higher = more serious candidate |
| `endorsements_received` | Cross-referenced | Use against skill `duration_months` for honeypot detection |
| `expected_salary_range_inr_lpa` | Budget fit | Flag if `min` > reasonable Series A ceiling (e.g., >80 LPA min) |
| `preferred_work_mode` | Logistics | Cross-check against Pune/Noida hybrid setup |
| `willing_to_relocate` | Logistics | Must be True if location is outside target tier-1 cities |
| `verified_email` + `verified_phone` + `linkedin_connected` | Anti-fraud | All three = authenticity multiplier; treat as combined boolean |

---

## 8. Skill Assessment Scores — Objective Proof

The `redrob_signals.skill_assessment_scores` dict contains objective test scores (0–100) on skills the candidate has actually been assessed on. These are the strongest signals in the dataset — they can't be self-reported.

| Assessment Domain | JD Alignment | Weight & Strategy |
|---|---|---|
| **NLP** | Core Must-Have | **Highest weight.** Direct proof of embeddings, retrieval, and matching fundamentals — the exact core of the JD. |
| **Fine-tuning LLMs** | High Score Booster | **High weight.** Directly verifies LoRA/QLoRA/PEFT nice-to-have. |
| **Image Classification** | Out of Domain | **Penalty if high with NLP=0.** Strong CV background with no NLP = JD disqualifier. |
| **Speech Recognition** | Out of Domain | **Penalty if high with NLP=0.** Strong speech background with no NLP = JD disqualifier. |

Scoring: compute `avg_assessment_score = (NLP × 0.5) + (FineTuning × 0.3) − (ImageClass × 0.1 if NLP=0) − (Speech × 0.1 if NLP=0)`

---

## 9. Cultural & Vibe Signals (LLM Evaluation Context)

These inform the qualitative reasoning the LLM scorer should apply:

- **Shipper > Researcher:** Evidence of shipping working systems fast, even if imperfect, is valued over academic thoroughness. Look for language like "shipped", "deployed", "in production", "real users."
- **Written communication:** The team is async-first and writes extensively. Well-articulated summaries in the profile are a positive signal. Bullet-point-only profiles with no synthesis are mild negatives.
- **Startup adaptability:** Look for candidates who've navigated changing scope, worn multiple hats, or shipped without a large supporting team. Exclusively big-company or consulting backgrounds may struggle.

---

## 10. Scoring Weight Summary (Initial Estimates — Tune in Step 10)

| Signal Group | Initial Weight | Rationale |
|---|---|---|
| Semantic embedding similarity | 0.35 | Captures JD understanding without keyword dependency |
| LLM rubric score (5 dimensions) | 0.30 | Nuanced reasoning — career trajectory, red flags, shipper vs researcher |
| Must-have skill match + assessment scores | 0.20 | Objective, direct JD requirement coverage |
| Behavioral score (availability, responsiveness, GitHub) | 0.15 | "Perfect on paper but unreachable" = not a real candidate |
| Honeypot penalty | ×0.01 multiplier | Applied post-fusion to all 67 flagged IDs |