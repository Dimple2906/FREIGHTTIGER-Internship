from pathlib import Path
import hashlib
import pandas as pd

from src.pipeline import build_results, write_output


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"

TEMPLATE = DATA / "sample_output_format_v2.csv"
OUTPUT_FILE = OUTPUT / "freight_cost_analysis_final.csv"

THRESHOLD = 0.20


def as_bool(value):
    """Safely interpret boolean or flagged values."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    val_str = str(value).strip().lower()
    return val_str in {"true", "yes", "1", "flagged"}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Output template not found: {TEMPLATE}")

    print("Running FreightTiger Smart Assistant — Evidence-Based Root Cause Analysis...")

    # 1. Run pipeline analysis on the full dataset
    result = build_results(DATA, threshold=THRESHOLD)

    if result.empty:
        raise ValueError("No records were returned by build_results().")

    # 2. Write strict template-compliant output
    write_output(result, TEMPLATE, OUTPUT_FILE)

    # 3. Verify the exported file
    exported = pd.read_csv(OUTPUT_FILE)

    print("\n=== FreightTiger Smart Assistant Summary ===")
    print(f"Total records analyzed: {len(result)}")
    print(f"Records exported: {len(exported)}")

    # Flagged records count
    flagged_mask = result["flagged_status"].isin(["Yes", "No (justified)"]) | result["flagged"].map(as_bool)
    print(f"Total anomaly alerts triggered: {int(flagged_mask.sum())}")

    justified_count = int((result["verdict"] == "justified").sum())
    unexplained_count = int(((result["verdict"] == "unexplained") & flagged_mask).sum())
    normal_count = int((result["verdict"] == "normal").sum())

    print(f"  - Explained / justified anomalies: {justified_count}")
    print(f"  - Unexplained flagged anomalies: {unexplained_count}")
    print(f"  - Normal records within threshold: {normal_count}")

    # Reason completeness check
    missing_reasons = int((exported["reason"].isna() | exported["reason"].astype(str).str.strip().eq("")).sum())
    print(f"Records without a reason in exported CSV: {missing_reasons}")

    print(f"\nOutput File: {OUTPUT_FILE}")
    digest = hashlib.sha256(OUTPUT_FILE.read_bytes()).hexdigest()
    print(f"Output SHA-256 Digest: {digest}")

    print("\n=== Reason Group Distribution ===")
    if "reason_group" in result.columns:
        print(result["reason_group"].value_counts(dropna=False).to_string())

    print("\n=== Sample Records with Generated Explanations ===")
    preferred_cols = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "flagged",
        "matched_note_id",
        "reason",
    ]
    disp_cols = [c for c in preferred_cols if c in exported.columns]
    
    # Display distinct scenarios from the output
    sample_sub = exported[exported["flagged"] != "No"].head(6)
    for _, r in sample_sub.iterrows():
        print(f"\nRoute: {r['route']} | Week: {r['week_of']} | Flagged: {r['flagged']} | Note: {r['matched_note_id']}")
        print(f"Reason: {r['reason']}")

    return result, exported


if __name__ == "__main__":
    main()
