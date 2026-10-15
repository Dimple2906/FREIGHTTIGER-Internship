# FreightTiger Smart Shipping Cost Assistant — Evidence-Based Root Cause Analysis

An end-to-end freight cost anomaly detection, semantic context evaluation, and root-cause explanation engine built for FreightTiger.

---

## 1. Executive Summary & Objective

The assistant monitors freight transportation rates week by week across all commercial shipping corridors to:
- Spot freight cost inflation per tonne-kilometre (`INR / tonne-km`) that departs significantly from expected baselines.
- Evaluate real-world contextual evidence (fuel price spikes, severe weather, toll changes, festival surcharges, fleet mandates).
- Determine whether an anomaly is genuinely justified by traceable real-world context or remains unexplained.
- Produce evidence-grounded, plain-English explanations and actionable next steps for human freight procurement reviewers.

---

## 2. Comparison Baselines & Precision Specifications

As defined in the engineering brief:
- **Metric**: Cost per tonne-kilometre = `total_cost / (quantity_tonnes * distance_km)`.
- **Grouping**: Shipments are grouped by route corridor (`origin` - `destination`), `route_type` (`Short` / `Medium` / `Long`), and Monday calendar week (`week_of`).
- **vs. Own History**: Trailing 8-week rolling average cost per tonne-km strictly prior to the current week (no look-ahead). When fewer than 8 prior weeks exist, all available prior observations are used. For week 1 of a route where 0 prior weeks exist, historical comparison is marked unavailable and explained as such.
- **vs. Similar Routes (Peer Baseline)**: Volume-weighted average cost per tonne-km across all other routes sharing the same `route_type` in that same week, strictly excluding the route itself.
- **Alert Threshold**: Configured at `+20.0%` deviation above either baseline.

---

## 3. Evidence-Based Anomaly & Context Engine

### Anomaly Distinction Categories
1. **Both Baselines Exceeded**: Cost is elevated >20% against both its historical route average and peer baseline (strong systemic or corridor rate surge).
2. **Historical Baseline Only**: Cost is elevated >20% above historical average, but peers do not show the same rise (route-specific event, carrier rate hike, or billing issue).
3. **Peer Baseline Only**: Cost is elevated >20% above peers of the same route length bucket, while historical comparison remains normal (peer market divergence or route comparability shift).
4. **Consecutive / Sustained Anomalies**: Sequential weekly records where cost continuously breaches baselines (e.g., 10-week sustained surge on Mumbai-Pune in late 2025).
5. **Validated Context Justification**: Route and dates directly overlap an event reporting plausible cost-increasing disruptions (e.g. Note N001 flood or N002 festival).
6. **Contradicted / Non-Justifying Notes**: Notes overlapping dates that explicitly state costs were absorbed without rate changes (N010), returned to normal (N009), demand remained stable (N006), or maintenance without cost impact (N005). These are explicitly rejected as justifications.
7. **Normal Records with Insufficient Baseline**: Records with no prior history (e.g., week 1 of data) are explained with missing baseline guidance rather than being falsely flagged.
8. **Normal Records Within Threshold**: Records within normal variance.

### Transparent Grouping (`reason_group`)
Records sharing the same underlying evidence structure are categorized into stable groups:
- `NOTE_SUPPORTED_<note_id>`: Grounded and supported by validated context note.
- `SUSTAINED_PEER_ANOMALY`: Route exceeds peer baseline for 3+ consecutive weekly records.
- `REPEATED_PEER_ANOMALY`: Route exceeds peer baseline for 2 consecutive weekly records.
- `ISOLATED_PEER_ANOMALY`: Single-week peer baseline exceedance.
- `SUSTAINED_HISTORY_ANOMALY`: Route exceeds historical baseline across consecutive weeks.
- `ISOLATED_HISTORY_ANOMALY`: Single-week historical baseline exceedance.
- `BOTH_BASELINES_EXCEEDED`: Both baselines exceeded without supporting note.
- `POST_EVENT_UNEXPLAINED`: Elevated costs following event reopening/concluding date.
- `CONTRADICTED_NOTE_UNEXPLAINED`: Context note confirms costs were absorbed or normal.
- `NORMAL_WITHIN_THRESHOLD`: Normal operation within configured thresholds.
- `NORMAL_INSUFFICIENT_HISTORY`: Normal record where historical baseline is not yet established.

---

## 4. Setup & Execution

### Prerequisites
Python 3.10+ with `pandas` and `numpy`:
```bash
python -m pip install -r requirements.txt
```

### Run Full Analysis
Processes the full 728 route-week dataset and exports to `output/freight_cost_analysis_final.csv`:
```bash
python run.py
```

### Run 10-Record Demo Extraction
Extracts 10 representative real-world scenarios to `output/freight_cost_demo_10.csv`:
```bash
python create_demo.py
```

### Run Reproducibility Check
Executes 3 independent passes and asserts 100% SHA-256 hash identity:
```bash
python run_three_times.py
```

### Run Test Suite
Executes unit and integration tests covering all 12 validation requirements:
```bash
python -m unittest tests/test_assistant.py
```

---

## 5. Output Contract

Output strictly preserves the column specification of `data/sample_output_format_v2.csv`:
1. `route` (e.g. `Delhi-Jaipur`)
2. `week_of` (Monday date, `YYYY-MM-DD`)
3. `cost_per_tonne_km` (rounded to 2 decimal places)
4. `vs_own_history` (e.g. `+35.5% vs this route's past average`)
5. `vs_similar_routes` (e.g. `+21.0% vs similar-length routes this week`)
6. `flagged` (`No (justified)`, `Yes`, or `No`)
7. `matched_note_id` (note ID if justified; blank `""` if unexplained or normal)
8. `reason` (evidence-based explanation including baseline breakdown and reviewer next steps)

---

## 6. Token & Cost Log

Deterministic rule-and-retrieval pipeline:
- **Total LLM Calls**: 0 (deterministic semantic parsing and grounding engine)
- **Input Tokens**: 0
- **Output Tokens**: 0
- **Estimated Cost**: $0.00
- **Execution Time**: ~1.8 seconds for full 728-record dataset
- **Reproducibility**: 100% byte-for-byte identical across runs (SHA-256 verified)
