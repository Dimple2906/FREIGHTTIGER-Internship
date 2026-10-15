import re
import pandas as pd


def first_column(df, names):
    """Find the first matching column name ignoring case and whitespace."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.strip().lower() in lookup:
            return lookup[name.strip().lower()]
    return None


def norm(value):
    """Normalize a string by stripping punctuation and case-folding."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def note_date_range(note_text, note_date):
    """
    Extract an explicit date range such as 'from Feb 24 to Mar 8' from note text.
    If no range pattern is found, treat note_date as a single-day event.
    Returns (start_date, end_date) as normalized Timestamps.
    """
    base_date = pd.to_datetime(note_date, errors="coerce")
    if pd.isna(base_date):
        return pd.NaT, pd.NaT

    match = re.search(
        r"from\s+([A-Za-z]+)\s+(\d{1,2})\s+to\s+([A-Za-z]+)\s+(\d{1,2})",
        str(note_text),
        flags=re.IGNORECASE,
    )

    if not match:
        return base_date.normalize(), base_date.normalize()

    start_month, start_day, end_month, end_day = match.groups()
    year = base_date.year

    start = pd.to_datetime(f"{start_month} {start_day} {year}", errors="coerce")
    end = pd.to_datetime(f"{end_month} {end_day} {year}", errors="coerce")

    if pd.isna(start) or pd.isna(end):
        return base_date.normalize(), base_date.normalize()

    # Handle cross-year date ranges (e.g. Dec 28 to Jan 4)
    if end < start:
        end = end + pd.DateOffset(years=1)

    return start.normalize(), end.normalize()


# Phrases indicating plausible cost-increasing disruptions
SUPPORTING_PHRASES = (
    "higher trip costs",
    "higher costs",
    "temporary surcharge",
    "surcharge applied",
    "pushing up transportation costs",
    "increased transportation costs",
    "increased freight costs",
    "costs increased",
    "longer detours and higher trip costs",
    "diesel prices rose",
)

# Phrases that explicitly indicate costs were NOT affected, absorbed, or normalized
CONTRADICTING_PHRASES = (
    "costs were not significantly affected",
    "without a rate change",
    "absorbed by transporters without a rate change",
    "returned to normal conditions",
    "returned to normal",
    "freight movement remained normal",
    "stable demand",
    "no major disruptions",
    "no significant disruptions",
    "not part of this dataset",
    "improved road conditions",
)


def evaluate_context_notes(route_row, notes_df):
    """
    Rigorously evaluate all candidate context notes for a given route-week record.

    Validates:
    - Route scope (exact route corridor vs All Routes)
    - Effective dates (event duration overlapping current week)
    - Semantic support (plausible cost increase vs absorbed/normal/unaffected)
    - Distinguishes supported, contradicted, post-event/normalized, conflicting, or absent evidence.

    Returns a dict with:
      status: 'supported' | 'contradicted' | 'post_event' | 'conflicting' | 'nearby_neutral' | 'none'
      matched_note_id: str or None (only populated if genuinely supported!)
      note_text: str or None
      event_summary: str
      detail: str
    """
    if notes_df is None or notes_df.empty:
        return {
            "status": "none",
            "matched_note_id": None,
            "note_text": None,
            "event_summary": "No context notes dataset available.",
            "detail": "No context notes found.",
        }

    id_col = first_column(notes_df, ["note_id", "id", "context_note_id"])
    text_col = first_column(notes_df, ["note", "context", "note_text", "description", "text"])
    date_col = first_column(notes_df, ["date", "note_date", "event_date"])
    applies_col = first_column(notes_df, ["applies_to"])
    origin_col = first_column(notes_df, ["origin", "source"])
    dest_col = first_column(notes_df, ["destination", "dest"])
    route_col = first_column(notes_df, ["route", "route_name"])

    week_start = pd.to_datetime(route_row.get("week_of"), errors="coerce")
    if pd.isna(week_start):
        return {
            "status": "none",
            "matched_note_id": None,
            "note_text": None,
            "event_summary": "Invalid week date.",
            "detail": "Invalid week date.",
        }

    week_start = week_start.normalize()
    week_end = week_start + pd.Timedelta(days=6)

    origin = str(route_row.get("origin", "")).strip()
    destination = str(route_row.get("destination", "")).strip()
    route_label = f"{origin}-{destination}"
    exact_route = norm(origin + destination)

    overlapping_supporting = []
    overlapping_contradicting = []
    recent_ended = []
    nearby_neutral = []

    for _, n in notes_df.iterrows():
        nid = str(n[id_col]).strip() if id_col and pd.notna(n[id_col]) else "Unknown"
        text = str(n[text_col]).strip() if text_col and pd.notna(n[text_col]) else ""
        if not text or text.casefold() == "nan":
            continue

        # Route matching
        scope_type = None
        if applies_col and pd.notna(n[applies_col]):
            applies_to = norm(n[applies_col])
            if applies_to in {"allroutes", "allroute", "all"}:
                scope_type = "global"
            elif applies_to == exact_route:
                scope_type = "exact"
            else:
                continue
        else:
            n_orig = norm(n[origin_col]) if origin_col and pd.notna(n[origin_col]) else ""
            n_dest = norm(n[dest_col]) if dest_col and pd.notna(n[dest_col]) else ""
            n_rt = norm(n[route_col]) if route_col and pd.notna(n[route_col]) else ""

            if n_rt == exact_route or (n_orig == norm(origin) and n_dest == norm(destination)):
                scope_type = "exact"
            else:
                continue

        note_date = n[date_col] if date_col and pd.notna(n[date_col]) else pd.NaT
        start, end = note_date_range(text, note_date)
        if pd.isna(start) or pd.isna(end):
            continue

        text_lower = text.casefold()
        is_supporting = any(p in text_lower for p in SUPPORTING_PHRASES)
        is_contradicting = any(p in text_lower for p in CONTRADICTING_PHRASES)

        item = {
            "note_id": nid,
            "text": text,
            "scope": scope_type,
            "start": start,
            "end": end,
            "is_supporting": is_supporting,
            "is_contradicting": is_contradicting,
        }

        # Date overlap check
        if start <= week_end and end >= week_start:
            if is_supporting and not is_contradicting:
                overlapping_supporting.append(item)
            elif is_contradicting:
                overlapping_contradicting.append(item)
            else:
                # Neutral note overlapping this week
                nearby_neutral.append(item)
        elif end < week_start:
            # Event concluded before this week
            days_ago = (week_start - end).days
            if days_ago <= 21:
                item["days_ago"] = days_ago
                recent_ended.append(item)
        elif start > week_end:
            # Event begins in future
            days_ahead = (start - week_end).days
            if days_ahead <= 14:
                item["days_ahead"] = days_ahead
                nearby_neutral.append(item)

    # Filter to exact route matches if any exist
    exact_supporting = [x for x in overlapping_supporting if x["scope"] == "exact"]
    if exact_supporting:
        overlapping_supporting = exact_supporting

    exact_contradicting = [x for x in overlapping_contradicting if x["scope"] == "exact"]
    if exact_contradicting:
        overlapping_contradicting = exact_contradicting

    # 1. Conflicting notes
    if overlapping_supporting and overlapping_contradicting:
        sup_ids = ", ".join(x["note_id"] for x in overlapping_supporting)
        con_ids = ", ".join(x["note_id"] for x in overlapping_contradicting)
        return {
            "status": "conflicting",
            "matched_note_id": None,
            "note_text": overlapping_supporting[0]["text"],
            "event_summary": f"Conflicting context: {sup_ids} indicates increase, but {con_ids} indicates costs absorbed or normal.",
            "detail": f"Multiple notes match with conflicting indications ({sup_ids} vs {con_ids}). Evidence is ambiguous; root cause unconfirmed.",
        }

    # 2. Supported by valid context note
    if overlapping_supporting:
        best = overlapping_supporting[0]
        # Short event summary
        summary = _summarize_event(best["note_id"], best["text"])
        return {
            "status": "supported",
            "matched_note_id": best["note_id"],
            "note_text": best["text"],
            "event_summary": summary,
            "detail": f"Matches context note {best['note_id']}: {best['text']}",
        }

    # 3. Contradicted by note overlapping current week (e.g. N010 compliance absorbed, N009 returned to normal)
    if overlapping_contradicting:
        best = overlapping_contradicting[0]
        summary = _summarize_event(best["note_id"], best["text"])
        return {
            "status": "contradicted",
            "matched_note_id": None,
            "note_id": best["note_id"],
            "note_text": best["text"],
            "event_summary": summary,
            "detail": f"Context note {best['note_id']} overlaps this period but explicitly reports that costs were absorbed, normal, or unaffected.",
        }

    # 4. Post-event note: Disruption ended recently, route returned to normal
    exact_recent = [x for x in recent_ended if x["scope"] == "exact"]
    if exact_recent:
        best = exact_recent[0]
        summary = _summarize_event(best["note_id"], best["text"])
        return {
            "status": "post_event",
            "matched_note_id": None,
            "note_id": best["note_id"],
            "note_text": best["text"],
            "days_ago": best["days_ago"],
            "event_summary": summary,
            "detail": f"Disruption note {best['note_id']} concluded prior to this week (roads reported passable); continued cost elevation is not justified by the ended event.",
        }

    # 5. Nearby neutral note (e.g. N006 stable demand)
    if nearby_neutral:
        best = nearby_neutral[0]
        summary = _summarize_event(best["note_id"], best["text"])
        return {
            "status": "nearby_neutral",
            "matched_note_id": None,
            "note_id": best["note_id"],
            "note_text": best["text"],
            "event_summary": summary,
            "detail": f"Context note {best['note_id']} reports stable operational conditions with no disruptions and does not explain a cost rise.",
        }

    # 6. No matching note
    return {
        "status": "none",
        "matched_note_id": None,
        "note_text": None,
        "event_summary": "No matching context note found.",
        "detail": "No matching context note found for this route or date range.",
    }


def _summarize_event(note_id, text):
    """Generate a clean, factual 1-sentence summary of a context note."""
    t = text.lower()
    if "flooding" in t or "flood" in t:
        return "heavy highway flooding and detour disruption"
    if "festival" in t or "surcharge" in t:
        return "regional festival demand and transporter surcharges"
    if "diesel" in t:
        return "nationwide diesel price increase"
    if "toll" in t:
        return "toll plaza commissioning (unaffected routes)"
    if "vehicle tracking" in t or "mandate" in t:
        return "vehicle tracking mandate with costs absorbed by transporters"
    if "maintenance" in t:
        return "scheduled highway maintenance with minor delays"
    if "returned to normal" in t:
        return "highway authorities confirming return to normal road conditions"
    if "resurfacing" in t or "improved road" in t:
        return "improved road conditions following resurfacing"
    if "stable demand" in t or "no major disruptions" in t:
        return "stable freight demand with no major disruptions"
    return f"reported event in note {note_id}"


def match_context_note(route_row, notes_df):
    """
    Backwards-compatible interface.
    Returns (matched_note_id, note_text) only if a genuinely supporting note was matched.
    Otherwise returns (None, None).
    """
    res = evaluate_context_notes(route_row, notes_df)
    if res["status"] == "supported":
        return res["matched_note_id"], res["note_text"]
    return None, None


def match_nearest_context_note(route_row, notes_df, max_days=14):
    """
    Backwards-compatible interface for retrieving nearby context.
    Returns (note_id, note_text, note_date_str) or (None, None, None).
    Note: nearby notes are purely observational and never justify an anomaly.
    """
    res = evaluate_context_notes(route_row, notes_df)
    if res["status"] in {"contradicted", "post_event", "nearby_neutral"}:
        note_id = res.get("note_id")
        text = res.get("note_text")
        # Find note date
        date_col = first_column(notes_df, ["date", "note_date", "event_date"])
        id_col = first_column(notes_df, ["note_id", "id", "context_note_id"])
        matching = notes_df[notes_df[id_col].astype(str).str.strip() == str(note_id)] if id_col else pd.DataFrame()
        dt_str = str(matching[date_col].iloc[0])[:10] if not matching.empty and date_col else ""
        return note_id, text, dt_str
    return None, None, None
