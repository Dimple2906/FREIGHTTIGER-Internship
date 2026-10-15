import pandas as pd
from pathlib import Path

source = Path("output/freight_cost_analysis_final.csv")
destination = Path("output/freight_cost_demo_10.csv")

df = pd.read_csv(source)
df["week_of"] = pd.to_datetime(df["week_of"], errors="coerce")

columns = [
    "route", "week_of", "cost_per_tonne_km",
    "vs_own_history", "vs_similar_routes",
    "flagged", "matched_note_id", "reason"
]

# Actual records to demonstrate distinct scenarios.
targets = [
    ("Delhi-Jaipur", "2024-11-11", "Unexplained historical anomaly"),
    ("Ahmedabad-Mumbai", "2025-01-20", "Justified by matching note N002"),
    ("Chennai-Bangalore", "2025-03-10", "Nearby context, cause unconfirmed"),
    ("Mumbai-Pune", "2025-06-23", "Peer-only anomaly"),
    ("Mumbai-Pune", "2025-09-15", "Peer anomaly; N006 is not justification"),
    ("Mumbai-Pune", "2025-10-27", "Peer anomaly; N010 is not justification"),
    ("Mumbai-Pune", "2025-11-17", "Repeated peer anomaly"),
    ("Mumbai-Pune", "2025-12-08", "Strong repeated peer anomaly"),
    ("Mumbai-Pune", "2024-03-04", "Normal comparison"),
    ("Ahmedabad-Mumbai", "2024-01-01", "No historical baseline"),
]

selected = []

for route, date, scenario in targets:
    match = df[
        df["route"].eq(route)
        & df["week_of"].eq(pd.Timestamp(date))
    ]

    if match.empty:
        print("MISSING:", route, date, "-", scenario)
        continue

    selected.append((match.index[0], scenario))

if len(selected) != 10:
    raise ValueError(
        f"Expected 10 distinct target records; found {len(selected)}. "
        "Check the MISSING messages before proceeding."
    )

demo = df.loc[[idx for idx, _ in selected], columns].copy()
demo["week_of"] = demo["week_of"].dt.strftime("%Y-%m-%d")
demo.to_csv(destination, index=False)

print("\nDemo saved to:", destination)
print("Rows:", len(demo))
print("Columns:", len(demo.columns))
print("\nScenario verification:")

for (_, scenario), (_, row) in zip(selected, demo.iterrows()):
    print(
        f"- {scenario}: {row['route']} | {row['week_of']} "
        f"| {row['flagged']} | note={row['matched_note_id']}"
    )

print("\nFlagged counts:")
print(demo["flagged"].value_counts(dropna=False))
