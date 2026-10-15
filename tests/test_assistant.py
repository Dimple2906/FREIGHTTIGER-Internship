import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import standardize_shipments, build_weekly_metrics
from src.flagging import detect_anomalies
from src.retrieval import evaluate_context_notes, match_context_note
from src.pipeline import build_results, write_output, generate_explanation_and_action


class TestFreightTigerSmartAssistant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ROOT / "data"
        cls.template_path = cls.data_dir / "sample_output_format_v2.csv"
        cls.shipments = pd.read_csv(cls.data_dir / "shipment_records.csv")
        cls.notes = pd.read_csv(cls.data_dir / "context_notes.csv")
        cls.weekly = build_weekly_metrics(cls.shipments)
        cls.results = build_results(cls.data_dir, threshold=0.20)

    # 1. Both baselines exceeded
    def test_both_baselines_exceeded(self):
        # Delhi-Jaipur on 2024-11-11
        row = self.results[
            (self.results["origin"] == "Delhi")
            & (self.results["destination"] == "Jaipur")
            & (self.results["week_of"] == "2024-11-11")
        ].iloc[0]

        self.assertTrue(row["flagged"])
        self.assertGreater(row["history_increase_pct"], 0.20)
        self.assertGreater(row["peer_increase_pct"], 0.20)
        self.assertEqual(row["verdict"], "unexplained")
        self.assertIn("historical route baseline", row["reason"].lower())
        self.assertIn("similar-route baseline", row["reason"].lower())
        self.assertIn(row["reason_group"], {"BOTH_BASELINES_EXCEEDED", "SUSTAINED_HISTORY_ANOMALY"})
        self.assertIn("revisions", row["recommended_action"].lower())

    # 2. Historical-only anomaly
    def test_historical_only_anomaly(self):
        # Delhi-Jaipur on 2024-11-18 (hist > 20%, peer < 20%)
        row = self.results[
            (self.results["origin"] == "Delhi")
            & (self.results["destination"] == "Jaipur")
            & (self.results["week_of"] == "2024-11-18")
        ].iloc[0]

        self.assertTrue(row["flagged"])
        self.assertGreater(row["history_increase_pct"], 0.20)
        self.assertLessEqual(row["peer_increase_pct"], 0.20)
        self.assertEqual(row["verdict"], "unexplained")
        self.assertIn("historical baseline", row["reason"].lower())

    # 3. Peer-only anomaly
    def test_peer_only_anomaly(self):
        # Mumbai-Pune on 2025-06-23 (peer > 20%, hist < 20%)
        row = self.results[
            (self.results["origin"] == "Mumbai")
            & (self.results["destination"] == "Pune")
            & (self.results["week_of"] == "2025-06-23")
        ].iloc[0]

        self.assertTrue(row["flagged"])
        self.assertGreater(row["peer_increase_pct"], 0.20)
        self.assertLessEqual(row["history_increase_pct"], 0.20)
        self.assertEqual(row["verdict"], "unexplained")
        self.assertIn("similar-route baseline", row["reason"].lower())
        self.assertIn("comparability", row["recommended_action"].lower())

    # 4. Valid route- and date-matched context note
    def test_valid_route_and_date_matched_context_note(self):
        # Ahmedabad-Mumbai on 2025-01-20 (N002)
        row = self.results[
            (self.results["origin"] == "Ahmedabad")
            & (self.results["destination"] == "Mumbai")
            & (self.results["week_of"] == "2025-01-20")
        ].iloc[0]

        self.assertTrue(row["flagged"])
        self.assertEqual(row["verdict"], "justified")
        self.assertEqual(row["matched_note_id"], "N002")
        self.assertEqual(row["flagged_status"], "No (justified)")
        self.assertIn("n002", row["reason"].lower())
        self.assertIn("potentially explained", row["reason"].lower())
        self.assertIn("verify invoice", row["recommended_action"].lower())

    # 5. Irrelevant or expired context note
    def test_irrelevant_or_expired_context_note(self):
        # Chennai-Bangalore on 2025-03-10 is after N001 ended (Mar 8)
        row_post = self.results[
            (self.results["origin"] == "Chennai")
            & (self.results["destination"] == "Bangalore")
            & (self.results["week_of"] == "2025-03-10")
        ].iloc[0]

        self.assertTrue(row_post["flagged"])
        self.assertEqual(row_post["verdict"], "unexplained")
        self.assertEqual(row_post["matched_note_id"], "")
        self.assertEqual(row_post["flagged_status"], "Yes")
        self.assertIn("concluded before this week", row_post["reason"].lower())

        # Note N010 (costs absorbed without rate change) must not justify rate increase
        row_n010 = self.results[
            (self.results["origin"] == "Mumbai")
            & (self.results["destination"] == "Pune")
            & (self.results["week_of"] == "2025-10-27")
        ].iloc[0]
        self.assertEqual(row_n010["verdict"], "unexplained")
        self.assertEqual(row_n010["matched_note_id"], "")
        self.assertIn("absorbed without a rate change", row_n010["reason"].lower())

    # 6. Conflicting context notes
    def test_conflicting_context_notes(self):
        # Create a synthetic note table with conflicting notes for the same route and week
        conflict_notes = pd.DataFrame([
            {
                "note_id": "T001",
                "date": "2025-05-12",
                "applies_to": "Delhi-Jaipur",
                "note": "Emergency fuel surcharge applied by transporters due to acute shortages, increasing freight costs by 25%.",
            },
            {
                "note_id": "T002",
                "date": "2025-05-12",
                "applies_to": "Delhi-Jaipur",
                "note": "State government subsidized freight operations; compliance costs were absorbed by transporters without a rate change.",
            },
        ])

        dummy_row = {
            "origin": "Delhi",
            "destination": "Jaipur",
            "week_of": "2025-05-12",
            "cost_per_tonne_km": 4.50,
            "history_increase_pct": 0.28,
            "peer_increase_pct": 0.25,
            "flagged": True,
        }

        note_eval = evaluate_context_notes(dummy_row, conflict_notes)
        self.assertEqual(note_eval["status"], "conflicting")
        self.assertIsNone(note_eval["matched_note_id"])

        reason, action, group, verdict, nid = generate_explanation_and_action(dummy_row, note_eval)
        self.assertEqual(verdict, "unexplained")
        self.assertEqual(nid, "")
        self.assertIn("conflicting", reason.lower())
        self.assertIn("conflicting context notes", action.lower())

    # 7. Consecutive-week anomalies
    def test_consecutive_week_anomalies(self):
        # Mumbai-Pune from 2025-10-20 to 2025-12-22 is a 10-week streak
        mumbai_pune = self.results[
            (self.results["origin"] == "Mumbai")
            & (self.results["destination"] == "Pune")
            & (self.results["week_of"] >= "2025-10-20")
            & (self.results["week_of"] <= "2025-12-22")
        ]

        self.assertEqual(len(mumbai_pune), 10)
        peer_weeks = mumbai_pune["consecutive_peer_weeks"].tolist()
        self.assertEqual(peer_weeks, list(range(1, 11)))

        # Week 5 (2025-11-17)
        r_w5 = mumbai_pune[mumbai_pune["week_of"] == "2025-11-17"].iloc[0]
        self.assertIn("5 consecutive weekly records", r_w5["reason"].lower())
        self.assertEqual(r_w5["reason_group"], "SUSTAINED_PEER_ANOMALY")

    # 8. Missing historical baseline
    def test_missing_historical_baseline(self):
        # First week of data (2024-01-01) has no prior history
        row_first = self.results[
            (self.results["origin"] == "Ahmedabad")
            & (self.results["destination"] == "Mumbai")
            & (self.results["week_of"] == "2024-01-01")
        ].iloc[0]

        self.assertFalse(row_first["flagged"])
        self.assertTrue(pd.isna(row_first["history_avg"]))
        self.assertEqual(row_first["reason_group"], "NORMAL_INSUFFICIENT_HISTORY")
        self.assertIn("sufficient route baseline has not been established", row_first["reason"].lower())
        self.assertIn("collect additional", row_first["recommended_action"].lower())

    # 9. Missing or malformed values handling
    def test_missing_or_malformed_values(self):
        raw_dirty = pd.DataFrame([
            {"shipment_date": "2024-01-05", "origin": "Delhi", "destination": "Jaipur", "route_type": "Short", "quantity_tonnes": 10.0, "distance_km": 250.0, "freight_cost_inr": 10000},
            {"shipment_date": "invalid-date", "origin": "Delhi", "destination": "Jaipur", "route_type": "Short", "quantity_tonnes": 10.0, "distance_km": 250.0, "freight_cost_inr": 10000},
            {"shipment_date": "2024-01-06", "origin": None, "destination": "Jaipur", "route_type": "Short", "quantity_tonnes": 10.0, "distance_km": 250.0, "freight_cost_inr": 10000},
            {"shipment_date": "2024-01-07", "origin": "Delhi", "destination": "Jaipur", "route_type": "Short", "quantity_tonnes": -5.0, "distance_km": 250.0, "freight_cost_inr": 10000},
            {"shipment_date": "2024-01-08", "origin": "Delhi", "destination": "Jaipur", "route_type": "Short", "quantity_tonnes": 10.0, "distance_km": 0.0, "freight_cost_inr": 10000},
        ])

        cleaned = standardize_shipments(raw_dirty)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned["origin"].iloc[0], "Delhi")

    # 10. Every exported record has a meaningful reason
    def test_every_exported_record_has_meaningful_reason(self):
        tmp_output = ROOT / "output" / "test_output_check.csv"
        write_output(self.results, self.template_path, tmp_output)

        exported = pd.read_csv(tmp_output)
        empty_reasons = (exported["reason"].isna() | exported["reason"].astype(str).str.strip().eq("")).sum()
        self.assertEqual(empty_reasons, 0)

        # Check all reasons have substantial content (>25 chars)
        short_reasons = (exported["reason"].astype(str).str.len() < 25).sum()
        self.assertEqual(short_reasons, 0)

        if tmp_output.exists():
            tmp_output.unlink()

    # 11. Output template compatibility
    def test_output_template_compatibility(self):
        tmp_output = ROOT / "output" / "test_template_compat.csv"
        write_output(self.results, self.template_path, tmp_output)

        template = pd.read_csv(self.template_path, nrows=0)
        exported = pd.read_csv(tmp_output)

        self.assertEqual(list(exported.columns), list(template.columns))
        self.assertIn("route", exported.columns)
        self.assertIn("week_of", exported.columns)
        self.assertIn("cost_per_tonne_km", exported.columns)
        self.assertIn("vs_own_history", exported.columns)
        self.assertIn("vs_similar_routes", exported.columns)
        self.assertIn("flagged", exported.columns)
        self.assertIn("matched_note_id", exported.columns)
        self.assertIn("reason", exported.columns)

        if tmp_output.exists():
            tmp_output.unlink()

    # 12. Full dataset output retains all analyzed records
    def test_full_dataset_retains_all_records(self):
        self.assertEqual(len(self.results), 728)
        output_file = ROOT / "output" / "freight_cost_analysis_final.csv"
        self.assertTrue(output_file.exists())
        exported = pd.read_csv(output_file)
        self.assertEqual(len(exported), 728)


if __name__ == "__main__":
    unittest.main()
