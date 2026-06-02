# Job Description Analysis & Feature Mapping
**Role:** Senior AI Engineer (Founding Team) — Redrob AI
**Target Profile:** 5-9 years total experience (Flexible: 6-8 years total, 4-5 years applied ML/AI is the sweet spot)

## 1. Hard Requirements (Must-Haves)
*   **Embeddings & Retrieval:** Deployed production experience with systems like sentence-transformers, BGE, E5, or OpenAI embeddings, etc. Must handle embedding drift, index refreshes, retrieval-quality.
*   **Vector Infrastructure:** Production operational experience with Vector DBs or hybrid search (Pinecone, Milvus, Qdrant, FAISS, OpenSearch, Elasticsearch or something similar).
*   **Language Proficiency:** Strong production-grade Python and code quality.
*   **Evaluation Rigor:** Experience in designing ranking evaluation frameworks (NDCG, MRR, MAP, offline-to-online correlation, A/B testing).


## 2. Score Boosters (Nice-to-Haves)
*   LLM fine-tuning experience (LoRA, QLoRA, PEFT).
*   Learning-to-Rank models (XGBoost-based or neural rankers).
*   Domain knowledge in HR-Tech / Recruiting Tech / Marketplaces.
*   Background in distributed systems or large-scale inference optimization.
*   Open-source contributions in AI/ML space. 
*   Geographic preference: Pune, Noida, Hyderabad, Mumbai, Delhi NCR (Tier-1 Indian cities).
*   Notice period: <30 days (High boost/buyout available), 30–60 days (Neutral), 60+ days(High penalty).
*   Open to offsite travel: Open to quarterly travel for offsites.

## 3. High Score Boosters
*   6-8 years total experience, 4-5 years in applied AI/ML role at product companies.
*   Must have shipped at least one end-to-end ranking, search, or recommendation system to real users at scale.
*   High proficiency in retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration (when to fine-tune vs prompt).
*   Located in or willing to relocate to Noida or Pune.
*   Active on Redrob platform (or has clear signal of being in the job market like linkedin). 

## 4. Disqualifiers & Penalties (Red Flags)
*   **Consulting-Only Trap:** Entire career spent exclusively at IT services/consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini).
*   **Pure Researchers:** Academic labs or research-only roles with no history of production deployment.
*   **LLM-Only Wrappers:** AI experience limited strictly to recent (<12 months) LangChain/OpenAI API stitching without pre-LLM historical ML context.
*   **Out of Domain:** Primary focus in Computer Vision, Speech, or Robotics without explicit NLP/IR depth.
*   **Title-Chasers:** Job hopping behavior (e.g., changing companies every <=1.5 years for title bumps).
*   **Inactive/Unavailable:** Candidates who haven't logged in for months or show low responsiveness (<=5% response rate).
*   **Inactive Coder:** Senior Engineer with no code writing experience in last 18 months because they moved to other non-coding roles.
*   **No External Validation:** Work based entirely on closed-source proprietary system for 5+ years without external validation(papers, talks, open-source).

## 5. Honeypot Patterns to Flag
*   Duration of experience at a company exceeding the company's real operational lifetime.
*   "Expert" proficiency claimed in 3+ skills where `duration_months` is 0.
*   Total years of experience mathematically implying the candidate started their career before age 18.
*   Skills displaying 0 endorsements despite a claimed "Expert" proficiency level.
*   AI keywords mentioned in skills not matching with past job roles (example: Marketing manager with AI keywords)
*   Tier 5(or lower) candidates: great plain-language profiles, no AI buzzwords — should rank high
*   Keyword Stuffers: High volume of AI keywords in the skills section but lacking substantive ML/AI contextual evidence in their career history descriptions.

## 6. Cultural & Vibe Signals (For LLM Evaluation)
*   **Shipper over Researcher:** A "scrappy product-engineering attitude" is preferred. Look for evidence of shipping working systems quickly, even if the underlying ML is suboptimal at first.
*   **Communication:** The team is "async-first and write a lot." Well-written, articulate profile summaries are a positive signal.
*   **Adaptability:** The role requires someone comfortable in an environment that "changes every six months" and where the team "moves fast and breaks things" (internally).

## 7. Redrob Behavioral Signals Mapping (All 23 Signals)
We translate the `redrob_signals` raw values into four core operational pillars: Availability, Responsiveness, Engagement/Demand, and Authenticity/Reliability.

### A. Availability & Recency (Crucial JD Constraint)
*   **3. last_active_date:** Convert to `days_since_active`. Map active this week to a 1.0 multiplier; inactive 6+ months drops to a 0.1 multiplier.
*   **4. open_to_work_flag:** Binary toggle. Formulates the base Availability score when multiplied by the activity recency.
*   **12. notice_period_days:** Logistics. <30 days = High Score Booster (buyout candidate). 30–60 days = Neutral. >60 days = High Penalty.

### B. Responsiveness & Reliability
*   **7. recruiter_response_rate:** Fraction of replies. If <= 0.05 (5%), apply a heavy down-weight penalty (candidate is functionally unavailable).
*   **8. avg_response_time_hours:** Speed of engagement. Lower numbers boost the responsiveness score.
*   **19. interview_completion_rate:** Attendance reliability proxy. High rates protect our precision metrics.
*   **20. offer_acceptance_rate:** Closing probability. Handle `-1` (no prior offers) as a neutral 0.5; high values show true intent to jump.

### C. Market Demand & Platform Engagement
*   **5. profile_views_received_30d** & **17. search_appearance_30d:** Measures macro market interest.
*   **18. saved_by_recruiters_30d:** Outbound verification proxy. If multiple recruiters bookmark them, it highly correlates with a strong technical profile.
*   **6. applications_submitted_30d:** High application velocity signals active intent to find a new job.
*   **10. connection_count:** General platform networking depth.

### D. Technical & Authenticity Signals
*   **16. github_activity_score:** Hard signal for coding activity. Essential since "this role writes code." Handle `-1` (unlinked) as 0. 
*   **1. profile_completeness_score:** Intent and thoroughness marker. 
*   **11. endorsements_received:** Cross-referenced with skill durations to catch honeypot mismatches.
*   **13. expected_salary_range_inr_lpa:** Filter out candidates whose minimum threshold completely eclipses Series A budget boundaries.
*   **14. preferred_work_mode** & **15. willing_to_relocate:** Cross-reference against Noida/Pune hybrid setup. Relocation flag must be True if location is outside target tier-1 tech hubs.
*   **21. verified_email**, **22. verified_phone**, & **23. linkedin_connected:** Multipliers used to combat fraud/synthetic fake data accounts.

---

## 8. Skill Assessment Scores — Objective Proof Weights
The dataset objectively evaluates candidates across 4 key test domains. Because the JD specifically demands NLP/IR depth and explicitly rejects out-of-domain fields without it, we assign specialized weights:

| Assessment Field | JD Alignment Category | Scoring Strategy / Weight |
| :--- | :--- | :--- |
| **NLP** | **Core Must-Have** | **Highest Weight.** Objective proof of embeddings, retrieval, and matching fundamentals. |
| **Fine-tuning LLMs** | **High Score Booster** | **High Weight.** Directly verifies nice-to-have experience with LoRA, QLoRA, and PEFT. |
| **Image Classification** | **Out of Domain** | **Low/Neutral Weight.** Treat as a penalty if high but candidate has 0 score in NLP (catches CV-only candidates). |
| **Speech Recognition** | **Out of Domain** | **Low/Neutral Weight.** Treat as a penalty if high but candidate has 0 score in NLP (catches Speech-only candidates). |

*Note: For the final ranker fusion step, we must calculate the `avg_assessment_score` using only the relevant AI/ML assessment variables.*