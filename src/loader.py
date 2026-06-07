import json
import os
from typing import Iterator, Dict, Any


def stream_candidates(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Streams candidates line by line from a JSONL file.
    Yields one candidate dictionary at a time to prevent RAM spikes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: Cannot find dataset at {file_path}")

    print(f"🌊 Starting stream from: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                yield candidate
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed JSON line: {e}")
                continue


# Quick test block to ensure it works when run directly
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Path to the real 100K dataset
    real_dataset_path = os.path.join(project_root, "data", "candidates.jsonl")

    print("--- TESTING STREAMER ON REAL candidates.jsonl ---")
    try:
        count = 0
        # Initialize the generator loop
        for candidate in stream_candidates(real_dataset_path):
            print(f"\nSuccessfully read candidate #{count + 1}")
            print(f"ID: {candidate.get('candidate_id')}")
            print(f"Name/Headline: {candidate.get('profile', {}).get('headline')}")
            print(f"Number of Skills: {len(candidate.get('skills', []))}")
            print(f"Active Date: {candidate.get('redrob_signals', {}).get('last_active_date')}")
            print("-" * 40)

            count += 1
            if count >= 3:  # Stop after 3 rows so we don't stream all 100,000 right now
                break

        print(f"✅ Real-file streaming test completed successfully!")
    except Exception as e:
        print(f"❌ Real-file test failed: {e}")