from pathlib import Path
import pandas as pd
import numpy as np

from src.metrics import build_weekly_metrics
from src.flagging import detect_anomalies
from src.retrieval import evaluate_context_notes


def generate_explanation_and_action(row, note_eval, threshold=0.20):
    """
    Generate an evidence-based explanation and recommended reviewer action.

    Strictly distinguishes:
    1. Anomaly above both historical and peer baselines
    2. Anomaly above the historical baseline only
    3. Anomaly above the peer baseline only
    4. Repeated / sustained anomalies across consecutive weeks
    5. Anomaly potentially explained by a validated context note
    6. Anomaly with insufficient evidence, post-event context, or contradicted notes
    7. Normal record with insufficient baseline data
    8. Normal record within threshold

    Never treats correlation as proof of causation.
    """
    flagged = bool(row.get("flagged", False))
    cost = float(row["cost_per_tonne_km"]) if pd.notna(row.get("cost_per_tonne_km")) else None
    cost_str = f"{cost:.2f}" if cost is not None else "N/A"

    hist_pct = row.get("history_increase_pct")
    peer_pct = row.get("peer_increase_pct")
    has_hist = pd.notna(hist_pct)
    has_peer = pd.notna(peer_pct)
    hist_val = float(hist_pct) if has_hist else None
    peer_val = float(peer_pct) if has_peer else None

    consec_peer = int(row.get("consecutive_peer_weeks", 0))
    consec_hist = int(row.get("consecutive_hist_weeks", 0))

    note_status = note_eval.get("status", "none")
    note_id = note_eval.get("matched_note_id")
    event_summary = note_eval.get("event_summary", "")

    thresh_pct = threshold * 100.0

    # ----------------------------------------------------
    # Case 1: Normal record (not flagged)
    # ----------------------------------------------------
    if not flagged:
        if not has_hist:
            # Missing historical baseline (Example E)
            peer_disp = f"{peer_val:+.1%}" if has_peer else "unavailable"
            reason = (
                f"Historical comparison is unavailable because a sufficient route baseline "
                f"has not been established. The current record cannot be assessed against its "
                f"own historical behaviour. Peer comparison is {peer_disp} versus similar routes "
                f"(below alert threshold). Use peer comparison cautiously and collect additional history."
            )
            action = "Collect additional weekly observations to build a reliable route historical baseline."
            group = "NORMAL_INSUFFICIENT_HISTORY"
            verdict = "normal"
            return reason, action, group, verdict, ""

        hist_str = f"{hist_val:+.1%}"
        peer_str = f"{peer_val:+.1%}" if has_peer else "unavailable"
        reason = (
            f"Freight cost of {cost_str} per tonne-km is within expected operational range "
            f"({hist_str} versus route historical baseline; {peer_str} versus similar-length routes). "
            f"Neither comparison exceeded the +{thresh_pct:.0f}% alert threshold."
        )
        action = "Normal operation; continue routine monitoring."
        group = "NORMAL_WITHIN_THRESHOLD"
        verdict = "normal"
        return reason, action, group, verdict, ""

    # ----------------------------------------------------
    # Case 2: Flagged anomaly justified by valid context note (Example C)
    # ----------------------------------------------------
    if note_status == "supported" and note_id:
        hist_str = f"{hist_val:+.1%}" if has_hist else "N/A"
        peer_str = f"{peer_val:+.1%}" if has_peer else "N/A"
        reason = (
            f"Context note {note_id} reports {event_summary} on this corridor during the relevant dates. "
            f"The event overlaps this week's cost elevation ({hist_str} vs historical baseline, "
            f"{peer_str} vs similar routes) and provides a plausible explanation. Mark this as "
            f"potentially explained, subject to invoice verification; do not claim the note proves the full increase."
        )
        action = f"Verify invoice and contracted freight rate against note {note_id} conditions."
        group = f"NOTE_SUPPORTED_{note_id}"
        verdict = "justified"
        return reason, action, group, verdict, note_id

    # ----------------------------------------------------
    # Case 3: Flagged anomaly with conflicting context notes
    # ----------------------------------------------------
    if note_status == "conflicting":
        reason = (
            f"Freight cost of {cost_str} per tonne-km triggered an anomaly alert, but matching context "
            f"notes provide conflicting indications. {note_eval.get('detail', '')} Due to ambiguous "
            f"evidence, cause remains unconfirmed. Review invoices and operational logs."
        )
        action = "Review conflicting context notes and operational logs to clarify rate impact."
        group = "ANOMALY_CONFLICTING_NOTES"
        verdict = "unexplained"
        return reason, action, group, verdict, ""

    # ----------------------------------------------------
    # Case 4: Flagged anomaly post-event / normalizing (e.g. Chennai-Bangalore post-flood)
    # ----------------------------------------------------
    if note_status == "post_event":
        prior_note = note_eval.get("note_id", "prior note")
        hist_str = f"{hist_val:+.1%}" if has_hist else "N/A"
        peer_str = f"{peer_val:+.1%}" if has_peer else "N/A"
        reason = (
            f"The freight cost is {hist_str} above historical baseline and {peer_str} above similar routes. "
            f"Prior disruption note {prior_note} concluded before this week (roads reported passable), "
            f"and subsequent reports confirmed normal conditions restored. Available context does not "
            f"explain continued cost elevation after reopening. Cause unconfirmed; verify whether "
            f"operational detours persisted."
        )
        action = "Verify whether operational detours or invoice backlogs persisted past the reported reopening date."
        group = "POST_EVENT_UNEXPLAINED"
        verdict = "unexplained"
        return reason, action, group, verdict, ""

    # ----------------------------------------------------
    # Case 5: Contradicted note overlapping current week (e.g. N010 compliance absorbed)
    # ----------------------------------------------------
    if note_status == "contradicted":
        c_note = note_eval.get("note_id", "")
        hist_str = f"{hist_val:+.1%}" if has_hist else "N/A"
        peer_str = f"{peer_val:+.1%}" if has_peer else "N/A"

        note_citation = (
            f"Context note {c_note} falls on this period but states compliance costs were "
            f"absorbed without a rate change, so it does not justify an increase."
            if c_note == "N010" else
            f"Context note {c_note} indicates conditions returned to normal or costs were unaffected, "
            f"which does not explain a cost increase."
        )

        if consec_peer >= 2:
            reason = (
                f"The cost is {peer_str} above the similar-route baseline and {hist_str} above historical baseline. "
                f"{note_citation} This is week {consec_peer} of consecutive peer elevation; "
                f"underlying cause remains unconfirmed."
            )
            action = "Audit carrier invoices for unauthorized surcharges and verify compliance terms."
            group = "SUSTAINED_PEER_ANOMALY" if consec_peer >= 3 else "REPEATED_PEER_ANOMALY"
        else:
            reason = (
                f"The cost is {peer_str} above similar routes and {hist_str} versus historical baseline. "
                f"{note_citation} Available data does not establish the cause. "
                f"Check peer comparability and carrier rate structures."
            )
            action = "Check peer comparability, carrier rate structures, and invoice terms."
            group = "CONTRADICTED_NOTE_UNEXPLAINED"

        verdict = "unexplained"
        return reason, action, group, verdict, ""

    # ----------------------------------------------------
    # Case 6: Consecutive / Repeated Peer Anomaly (Example D)
    # ----------------------------------------------------
    if consec_peer >= 2:
        peer_str = f"{peer_val:+.1%}"
        hist_str = f"{hist_val:+.1%}" if has_hist else "unavailable"

        nearby_mention = ""
        if note_status == "nearby_neutral":
            n_nid = note_eval.get("note_id", "")
            nearby_mention = f" Context note {n_nid} mentions stable demand with no disruptions, which does not explain a cost increase."

        reason = (
            f"This route has exceeded the peer baseline in {consec_peer} consecutive weekly records "
            f"(currently {peer_str} above peers and {hist_str} vs history).{nearby_mention} "
            f"The sustained pattern warrants investigation of route-specific pricing, capacity constraints, "
            f"or billing changes. Available data does not yet establish the underlying cause."
        )
        action = "Review consecutive weeks for sustained route-specific pricing, capacity constraints, or billing changes."
        group = "SUSTAINED_PEER_ANOMALY" if consec_peer >= 3 else "REPEATED_PEER_ANOMALY"
        verdict = "unexplained"
        return reason, action, group, verdict, ""

    # ----------------------------------------------------
    # Case 7: Both baselines exceeded without supporting note (Example A)
    # ----------------------------------------------------
    both_exceeded = has_hist and has_peer and (hist_val > threshold) and (peer_val > threshold)
    if both_exceeded:
        hist_str = f"{hist_val:+.1%}"
        peer_str = f"{peer_val:+.1%}"
        reason = (
            f"The freight cost is {hist_str} above its historical route baseline and {peer_str} "
            f"above the similar-route baseline. Both comparisons indicate an unusual increase. "
            f"No validated context note explains the change. Investigate rate revisions, invoices, "
            f"and operational events."
        )
        action = "Investigate rate revisions, carrier invoices, and operational route events."
        group = "BOTH_BASELINES_EXCEEDED"
        verdict = "unexplained"
        return reason, action, group, verdict, ""

    # ----------------------------------------------------
    # Case 8: Peer baseline exceeded only (Example B)
    # ----------------------------------------------------
    peer_only = has_peer and (peer_val > threshold) and (not has_hist or hist_val <= threshold)
    if peer_only:
        peer_str = f"{peer_val:+.1%}"
        hist_str = f"{hist_val:+.1%}" if has_hist else "unavailable"

        nearby_mention = ""
        if note_status == "nearby_neutral":
            n_nid = note_eval.get("note_id", "")
            nearby_mention = f" Context note {n_nid} mentions stable demand with no disruptions, which does not explain a cost increase."

        reason = (
            f"The cost is {peer_str} above the similar-route baseline but only {hist_str} "
            f"above this route's historical baseline. The increase is more pronounced relative "
            f"to comparable routes.{nearby_mention} No validated context note explains the change. "
            f"Check peer comparability and market-wide rate movements."
        )
        action = "Check peer comparability, carrier capacity, and market-wide rate movements."
        group = "ISOLATED_PEER_ANOMALY"
        verdict = "unexplained"
        return reason, action, group, verdict, ""

    # ----------------------------------------------------
    # Case 9: Historical baseline exceeded only
    # ----------------------------------------------------
    hist_only = has_hist and (hist_val > threshold) and (not has_peer or peer_val <= threshold)
    if hist_only:
        hist_str = f"{hist_val:+.1%}"
        peer_str = f"{peer_val:+.1%}" if has_peer else "unavailable"

        if consec_hist >= 2:
            reason = (
                f"This route has exceeded its historical baseline in {consec_hist} consecutive weekly records "
                f"({hist_str} above past average, with peer baseline at {peer_str}). The sustained increase "
                f"over history warrants investigation of route-specific pricing, capacity, or billing changes. "
                f"No validated context note explains the change."
            )
            action = "Audit consecutive weekly carrier invoices and rate contracts for route-specific surcharges."
            group = "SUSTAINED_HISTORY_ANOMALY"
        else:
            reason = (
                f"The freight cost is {hist_str} above this route's historical baseline, while the peer "
                f"comparison ({peer_str}) does not independently cross the alert threshold. The increase "
                f"is route-specific. No validated context note explains the change. Investigate rate "
                f"revisions, invoices, and operational events."
            )
            action = "Investigate route-specific carrier contracts and operational delivery logs."
            group = "ISOLATED_HISTORY_ANOMALY"

        verdict = "unexplained"
        return reason, action, group, verdict, ""

    # Fallback
    reason = (
        f"Freight cost of {cost_str} per tonne-km is flagged, but available comparison values do "
        f"not independently establish a clear cause. Verify threshold calculation and source data."
    )
    action = "Escalate for manual investigation because the cause is unknown."
    group = "ANOMALY_NEEDS_REVIEW"
    verdict = "unexplained"
    return reason, action, group, verdict, ""


def build_results(data_dir, threshold=0.20):
    """
    Execute full anomaly detection, context-note evaluation, grouping,
    and explanation pipeline.
    """
    data_dir = Path(data_dir)

    shipments = pd.read_csv(data_dir / "shipment_records.csv")
    notes = pd.read_csv(data_dir / "context_notes.csv")

    weekly = build_weekly_metrics(shipments)
    result = detect_anomalies(weekly, threshold=threshold)

    reasons = []
    actions = []
    groups = []
    verdicts = []
    matched_ids = []
    flagged_statuses = []

    for _, row in result.iterrows():
        note_eval = evaluate_context_notes(row, notes)
        reason, action, group, verdict, matched_id = generate_explanation_and_action(
            row, note_eval, threshold=threshold
        )

        reasons.append(reason)
        actions.append(action)
        groups.append(group)
        verdicts.append(verdict)
        matched_ids.append(matched_id)

        # Output flagged string contract
        is_fl = bool(row.get("flagged", False))
        if is_fl and verdict == "justified":
            flagged_statuses.append("No (justified)")
        elif is_fl:
            flagged_statuses.append("Yes")
        else:
            flagged_statuses.append("No")

    result["reason"] = reasons
    result["explanation"] = reasons  # Backwards compatibility
    result["recommended_action"] = actions
    result["reason_group"] = groups
    result["verdict"] = verdicts
    result["matched_note_id"] = matched_ids
    result["flagged_status"] = flagged_statuses

    return result


def write_output(result, template_path, output_path):
    """
    Write output matching the exact columns and order of the template.
    Preserves existing output contract.
    """
    template_path = Path(template_path)
    output_path = Path(output_path)

    template = pd.read_csv(template_path, nrows=0)
    output = pd.DataFrame(index=result.index)

    def get_column(*names):
        lookup = {str(c).strip().lower(): c for c in result.columns}
        for name in names:
            col = lookup.get(name.strip().lower())
            if col is not None:
                return result[col]
        return None

    # Route format: Origin-Destination
    origin = get_column("origin")
    destination = get_column("destination")
    if origin is not None and destination is not None:
        output["route"] = (
            origin.fillna("").astype(str).str.strip()
            + "-"
            + destination.fillna("").astype(str).str.strip()
        )

    def format_percentage(val, suffix):
        if pd.isna(val):
            return ""
        try:
            num = float(val)
            return f"{num:+.1%} {suffix}"
        except (TypeError, ValueError):
            return ""

    history_change = get_column("history_increase_pct")
    peer_change = get_column("peer_increase_pct")

    if history_change is not None:
        output["vs_own_history"] = history_change.apply(
            lambda x: format_percentage(x, "vs this route's past average")
        )

    if peer_change is not None:
        output["vs_similar_routes"] = peer_change.apply(
            lambda x: format_percentage(x, "vs similar-length routes this week")
        )

    # Map template columns
    for column in template.columns:
        key = str(column).strip().lower()

        if key == "route" and "route" in output.columns:
            continue

        if key == "week_of":
            src = get_column("week_of")
            if src is not None:
                output[column] = pd.to_datetime(src, errors="coerce").dt.strftime("%Y-%m-%d")
            else:
                output[column] = ""

        elif key == "cost_per_tonne_km":
            src = get_column("cost_per_tonne_km")
            output[column] = src.round(2) if src is not None else ""

        elif key == "vs_own_history":
            if "vs_own_history" not in output.columns:
                output[column] = ""

        elif key == "vs_similar_routes":
            if "vs_similar_routes" not in output.columns:
                output[column] = ""

        elif key == "flagged":
            src = get_column("flagged_status")
            if src is not None:
                output[column] = src
            else:
                flagged_col = get_column("flagged")
                verdict_col = get_column("verdict")
                statuses = []
                for i in result.index:
                    is_fl = bool(flagged_col.loc[i]) if flagged_col is not None else False
                    vd = str(verdict_col.loc[i]).lower() if verdict_col is not None else ""
                    if is_fl and vd == "justified":
                        statuses.append("No (justified)")
                    elif is_fl:
                        statuses.append("Yes")
                    else:
                        statuses.append("No")
                output[column] = statuses

        elif key == "matched_note_id":
            src = get_column("matched_note_id")
            output[column] = src.fillna("").astype(str) if src is not None else ""

        elif key == "reason":
            src = get_column("reason", "explanation", "flag_reason")
            output[column] = src.fillna("").astype(str) if src is not None else ""

        elif key == "reason_group":
            src = get_column("reason_group")
            output[column] = src.fillna("").astype(str) if src is not None else ""

        elif key == "recommended_action":
            src = get_column("recommended_action")
            output[column] = src.fillna("").astype(str) if src is not None else ""

        elif key == "verdict":
            src = get_column("verdict")
            output[column] = src.fillna("").astype(str) if src is not None else ""

        else:
            src = get_column(key)
            output[column] = src.values if src is not None else ""

    # Reindex strictly to template columns
    output = output.reindex(columns=template.columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    return output
