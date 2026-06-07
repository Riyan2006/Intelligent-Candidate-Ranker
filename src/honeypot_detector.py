"""
honeypot_detector.py
--------------------
Detects mathematically impossible (honeypot) candidate profiles.

Three mutually exclusive signal types, all based on hard arithmetic:

  Signal A — YOE vs actual career span
    profile.years_of_experience >> (2026 - earliest_role_start_date)
    e.g., claims 13.7 years experience but earliest career role started 0.8 years ago.

  Signal B — Expert proficiency on skills never used
    3+ skills marked "expert" with duration_months = 0.
    You cannot be expert at something you have spent 0 months using.

  Signal C — Single role duration exceeds total stated YOE
    One role's duration_months > profile.years_of_experience * 12 + 6 month buffer.
    You cannot have spent longer at one job than your total career length.

~67 honeypots found in the 100K dataset (README says "~80").
"""

import os
import json
from collections import defaultdict


CURRENT_YEAR_FLOAT = 2026.0   # approximate "now" for span calculation
SIGNAL_A_EXCESS_THRESHOLD = 8.0   # YOE must exceed career span by this many years
SIGNAL_B_MIN_IMPOSSIBLE_SKILLS = 3  # number of expert+0-duration skills needed to flag
SIGNAL_C_BUFFER_MONTHS = 6          # small rounding buffer for single-role check


def _earliest_career_start_year(career_history: list) -> float | None:
    """
    Returns the decimal year of the earliest start_date found in career_history.
    Returns None if no parseable dates exist.
    """
    starts = []
    for role in career_history:
        raw = role.get("start_date", "")
        if raw and len(raw) >= 7:
            try:
                yr = int(raw[:4])
                mo = int(raw[5:7])
                starts.append(yr + mo / 12.0)
            except (ValueError, IndexError):
                pass
    return min(starts) if starts else None


def detect_honeypots(candidate: dict) -> tuple[bool, str]:
    """
    Returns (True, reason_string) if the profile is a mathematical impossibility.
    Returns (False, "") for all legitimate profiles.

    Designed to produce ~0 false positives — every flagged profile is
    provably impossible regardless of cultural or regional variation.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    yoe = float(profile.get("years_of_experience", 0))

    # ----------------------------------------------------------------
    # Signal A: YOE vs actual career span
    # ----------------------------------------------------------------
    # career_span = years from earliest role start to today (2026).
    # If the profile claims more YOE than the actual career history
    # permits (with a generous 8-year buffer), it's impossible.
    #
    # Why 8 years? Legitimate edge cases:
    #   - Freelance/contract work before first full-time role (adds YOE without a start date)
    #   - Internships not listed in career_history
    #   - Part-time work during education
    # 8 years covers all these generously. Anything beyond that is manufactured.
    # ----------------------------------------------------------------
    earliest_start = _earliest_career_start_year(career)
    if earliest_start is not None:
        career_span_years = CURRENT_YEAR_FLOAT - earliest_start
        yoe_excess = yoe - career_span_years
        if yoe_excess > SIGNAL_A_EXCESS_THRESHOLD:
            return (
                True,
                f"Signal A — YOE impossibility: claims {yoe:.1f}yr but "
                f"career history spans only {career_span_years:.1f}yr "
                f"(excess: {yoe_excess:.1f}yr)"
            )

    # ----------------------------------------------------------------
    # Signal B: Expert skill with zero months of use
    # ----------------------------------------------------------------
    # duration_months = how long the candidate has used this skill.
    # A skill listed as "expert" with duration_months = 0 is a direct
    # internal contradiction: you cannot be expert at something you
    # have never used.
    #
    # We require 3+ such skills to avoid penalising data-entry errors
    # (e.g., one skill accidentally left at 0). If a profile has 3 or
    # more expert skills with 0 months, it was deliberately constructed.
    # ----------------------------------------------------------------
    impossible_expert_skills = [
        s["name"]
        for s in skills
        if s.get("proficiency") == "expert"
        and s.get("duration_months", 1) == 0
    ]
    if len(impossible_expert_skills) >= SIGNAL_B_MIN_IMPOSSIBLE_SKILLS:
        return (
            True,
            f"Signal B — Skill impossibility: {len(impossible_expert_skills)} skills "
            f"marked 'expert' with 0 months of use: "
            f"{', '.join(impossible_expert_skills[:5])}"
        )

    # ----------------------------------------------------------------
    # Signal C: A single role spans longer than total stated YOE
    # ----------------------------------------------------------------
    # If profile says total_yoe = 9.9 years but one role has
    # duration_months = 166 (13.8 years), that's arithmetically
    # impossible — you cannot spend more time at one job than your
    # entire career length.
    #
    # A 6-month buffer absorbs floating-point rounding in the yoe
    # field (e.g., yoe=9.9 stored as 9.87 or 9.93).
    # ----------------------------------------------------------------
    yoe_months = yoe * 12.0
    for role in career:
        role_duration = role.get("duration_months", 0)
        if role_duration > yoe_months + SIGNAL_C_BUFFER_MONTHS:
            return (
                True,
                f"Signal C — Duration impossibility: role '{role.get('title')}' "
                f"at {role.get('company')} lasted {role_duration}mo "
                f"but total YOE = {yoe:.1f}yr ({yoe_months:.0f}mo)"
            )

    return False, ""



# Standalone runner — scans all 100K candidates
if __name__ == "__main__":
    # Import loader from same src/ directory
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from loader import stream_candidates

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    dataset_path = os.path.join(project_root, "data", "candidates.jsonl")
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    output_path = os.path.join(artifacts_dir, "honeypot_ids.txt")

    print("🕵️  Honeypot sweep starting...")
    print(f"   Dataset : {dataset_path}")
    print(f"   Output  : {output_path}")
    print()

    total_processed = 0
    flagged = []
    signal_counters = defaultdict(int)
    signal_key = {"A": "Signal A — YOE vs career span",
                  "B": "Signal B — Expert+0-duration skills",
                  "C": "Signal C — Single role > total YOE"}

    for candidate in stream_candidates(dataset_path):
        total_processed += 1
        is_honeypot, reason = detect_honeypots(candidate)
        if is_honeypot:
            cid = candidate.get("candidate_id", "UNKNOWN")
            flagged.append((cid, reason))
            # Tally by signal letter
            for letter in ("A", "B", "C"):
                if f"Signal {letter}" in reason:
                    signal_counters[letter] += 1
                    break

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for cid, reason in flagged:
            f.write(f"{cid} | {reason}\n")

    # Report
    print("✅  Sweep complete")
    print(f"   Candidates scanned : {total_processed:,}")
    print(f"   Honeypots flagged  : {len(flagged)}")
    print()
    print("📊  Breakdown by signal:")
    for letter, label in signal_key.items():
        print(f"   [{letter}] {label}: {signal_counters[letter]}")
    print()
    print(f"💾  Saved to: {output_path}")