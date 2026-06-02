import json
import os

# Define path to the sample file
sample_path = os.path.join("data", "sample_candidates.json")

# Load the sample dataset
with open(sample_path, "r", encoding="utf-8") as f:
    sample_data = json.load(f)

print("--- DATASET OVERVIEW ---")
print(f"Total candidates in sample: {len(sample_data)}")

# Grab the first candidate to inspect the schema
first_candidate = sample_data[0]
print(f"\nCandidate Object Root Keys:\n{list(first_candidate.keys())}")

print("\n--- SAMPLE BEHAVIORAL SIGNALS ---")
# Print out the Redrob behavioral signals dictionary
signals = first_candidate.get("redrob_signals", {})
for signal_name, value in list(signals.items())[:10]:
    print(f"  {signal_name}: {value}")

print("\n--- SKILL ASSESSMENT SCORES ---")
# Check what fields are being tested objectively
assessments = signals.get("skill_assessment_scores", {})
print(f"Available objective test fields: {list(assessments.keys()) if assessments else 'None found'}")

print("\n--- SPOT-CHECKING PROFILE STRUCTURE ---")
print(f"Headline: {first_candidate.get('headline')}")
print(f"Skills Array Snippet: {first_candidate.get('skills', [])[:3]}")