import json
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# --- DYNAMIC PATHS & OUTPUTS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "candidates.jsonl")
OUTPUT_FILE_PARQUET = os.path.join(BASE_DIR, "artifacts", "behavioral_scores.parquet")
OUTPUT_FILE_JSON = os.path.join(BASE_DIR, "artifacts", "behavioral_scores.json")

# DYNAMIC DATE
CURRENT_DATE = datetime.now()


def calculate_days_since(date_string):
    if not date_string:
        return 9999
    try:
        date_obj = datetime.strptime(date_string.split("T")[0], "%Y-%m-%d")
        return (CURRENT_DATE - date_obj).days
    except ValueError:
        return 9999


def compute_behavioral_score(signals, location=""):
    if not signals:
        return 0.0

    score_components = []


    # PILLAR 1: Activity & Intent (Weight: 25%)

    activity_score = 0.0

    days_inactive = calculate_days_since(signals.get("last_active_date"))
    if days_inactive <= 7:
        activity_score += 0.40
    elif days_inactive <= 30:
        activity_score += 0.25
    elif days_inactive <= 90:
        activity_score += 0.10

    if signals.get("open_to_work_flag", False):
        activity_score += 0.20

    # Notice Period Logic
    notice_days = signals.get("notice_period_days", 90)
    if notice_days <= 15:
        activity_score += 0.20
    elif notice_days <= 30:
        activity_score += 0.10
    elif notice_days > 60:
        activity_score -= 0.15  # SIGNIFICANT PENALTY APPLIED

    apps = signals.get("applications_submitted_30d", 0)
    if apps > 0: activity_score += min(0.10, apps * 0.02)

    days_since_signup = calculate_days_since(signals.get("signup_date"))
    if 30 < days_since_signup < 9999: activity_score += 0.10

    score_components.append(("activity", min(1.0, max(0.0, activity_score)), 0.25))


    # PILLAR 2: Responsiveness & Follow-through (Weight: 25%)

    resp_score = 0.0

    response_rate = signals.get("recruiter_response_rate", 0.0)
    resp_score += (response_rate * 0.40)

    resp_time = signals.get("avg_response_time_hours", 999)
    if resp_time <= 12:
        resp_score += 0.30
    elif resp_time <= 24:
        resp_score += 0.20
    elif resp_time <= 48:
        resp_score += 0.10

    int_comp = signals.get("interview_completion_rate", 0.0)
    resp_score += (int_comp * 0.20)

    offer_acc = signals.get("offer_acceptance_rate", -1)
    if offer_acc >= 0:
        resp_score += (offer_acc * 0.10)
    else:
        resp_score += 0.05

    score_components.append(("responsiveness", min(1.0, resp_score), 0.25))


    # PILLAR 3: Logistics & Verifications (Weight: 20%)

    logistics_score = 0.0

    if signals.get("willing_to_relocate", False): logistics_score += 0.30

    work_mode = str(signals.get("preferred_work_mode", "")).lower()
    if work_mode in ["hybrid", "onsite", "flexible"]:
        logistics_score += 0.30
    elif work_mode == "remote":
        logistics_score += 0.10

    if signals.get("verified_email", False): logistics_score += 0.10
    if signals.get("verified_phone", False): logistics_score += 0.10
    if signals.get("linkedin_connected", False): logistics_score += 0.10

    sal = signals.get("expected_salary_range_inr_lpa", {})
    if sal.get("min", 0) > 0: logistics_score += 0.10

    score_components.append(("logistics", min(1.0, logistics_score), 0.20))


    # PILLAR 4: Platform Quality, Skills & Traction (Weight: 30%)

    cred_score = 0.0

    completeness = signals.get("profile_completeness_score", 0)
    cred_score += (completeness / 100.0) * 0.25

    github = signals.get("github_activity_score", -1)
    if github > 0: cred_score += (github / 100.0) * 0.25

    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        avg_score = sum(assessments.values()) / len(assessments)
        cred_score += (avg_score / 100.0) * 0.20

    conn = signals.get("connection_count", 0)
    endr = signals.get("endorsements_received", 0)
    if conn > 50 or endr > 10:
        cred_score += 0.15
    elif conn > 10 or endr > 2:
        cred_score += 0.05

    views = signals.get("profile_views_received_30d", 0)
    search = signals.get("search_appearance_30d", 0)
    saves = signals.get("saved_by_recruiters_30d", 0)

    traction = (views * 1) + (search * 0.2) + (saves * 5)
    if traction > 50:
        cred_score += 0.15
    elif traction > 10:
        cred_score += 0.05

    score_components.append(("credibility", min(1.0, cred_score), 0.30))


    # Final Weighted Fusion & Bonuses

    final_score = sum([score * weight for name, score, weight in score_components])

    # LOCATION BONUS (+0.05 for Tier 1 matching JD)
    if location:
        loc_str = str(location).lower()
        tier_1_cities = ["pune", "noida", "hyderabad", "mumbai", "delhi ncr", "delhi"]
        if any(city in loc_str for city in tier_1_cities):
            final_score += 0.05

    return round(max(0.0, min(1.0, final_score)), 4)


def main():
    print("🚀 Initializing Dual-Output Behavioral Scorer (JSON + Parquet)...")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Please ensure the path is correct.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE_PARQUET), exist_ok=True)

    data_rows = []
    json_dict = {}

    print("⏳ Processing candidates...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Scoring 23 signals + Location"):
        if not line.strip():
            continue

        candidate = json.loads(line)
        c_id = candidate.get("candidate_id")
        signals = candidate.get("redrob_signals", {})

        # Extract root-level location for the bonus
        location = candidate.get("location", "")

        # Compute the final modified score
        score = compute_behavioral_score(signals, location)

        # Append for Parquet (Table format)
        data_rows.append({"candidate_id": c_id, "behavioral_score": score})

        # Assign for JSON (O(1) Dictionary format)
        json_dict[c_id] = score

    print("📦 Saving outputs...")

    # 1. Save Parquet (Fast Machine Read)
    df = pd.DataFrame(data_rows)
    df.to_parquet(OUTPUT_FILE_PARQUET, engine="pyarrow", index=False)

    # 2. Save JSON (Human Readability/Debugging)
    with open(OUTPUT_FILE_JSON, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2)

    print(f"✅ Successfully scored {len(df)} candidates.")
    print(f"💾 Saved Parquet to: {OUTPUT_FILE_PARQUET}")
    print(f"💾 Saved JSON to:    {OUTPUT_FILE_JSON}")


if __name__ == "__main__":
    main()