import pandas as pd
import sys

def validate_submission(csv_path):
    print(f"🔍 Validating {csv_path}...")
    try:
        # Must be UTF-8
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return False

    errors = []

    # Check 1: 4 columns in the correct order
    expected_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    if list(df.columns) != expected_cols:
        errors.append(f"Expected columns {expected_cols}, got {list(df.columns)}")

    # Check 2: Exactly 100 rows
    if len(df) != 100:
        errors.append(f"Expected exactly 100 rows, got {len(df)}")

    # Check 3: Ranks 1-100 used once each
    if 'rank' in df.columns:
        ranks = df['rank'].tolist()
        if sorted(ranks) != list(range(1, 101)):
            errors.append("Ranks must be exactly 1 through 100 with no missing numbers or duplicates.")

    # Check 4: Scores non-increasing (descending order)
    if 'score' in df.columns:
        scores = df['score'].tolist()
        if scores != sorted(scores, reverse=True):
            errors.append("Scores must be in descending (non-increasing) order.")

    # Check 5: No duplicate candidate IDs
    if 'candidate_id' in df.columns:
        if df['candidate_id'].duplicated().any():
            errors.append("Duplicate candidate_ids found. Each ID must be unique.")

    # Output Results
    if errors:
        print("❌ Validation Failed. The autograder would reject this:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("✅ Validation Passed! Perfect format. 100 rows, correct columns, valid ranks and scores.")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <path_to_csv>")
    else:
        validate_submission(sys.argv[1])