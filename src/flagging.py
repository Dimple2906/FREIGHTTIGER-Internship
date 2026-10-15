import numpy as np
import pandas as pd


def detect_anomalies(weekly, threshold=0.20):
    """
    Detect freight cost anomalies against historical route baseline and
    same-week similar-length peer baseline.

    Distinguishes:
    1. Anomaly above both historical and peer baselines
    2. Anomaly above the historical baseline only
    3. Anomaly above the peer baseline only
    4. Repeated anomalies across consecutive weeks
    5. Normal record with insufficient baseline data
    6. Normal record within threshold

    Also tracks temporal sequences per route to identify consecutive-week
    and sustained cost anomalies.
    """
    df = weekly.copy()

    # Baseline availability
    df["has_history_baseline"] = df["history_avg"].notna() & (df["history_avg"] > 0)
    df["has_peer_baseline"] = df["peer_avg"].notna() & (df["peer_avg"] > 0)

    # Percentage deviations
    df["history_increase_pct"] = np.where(
        df["has_history_baseline"],
        (df["cost_per_tonne_km"] / df["history_avg"]) - 1.0,
        np.nan,
    )

    df["peer_increase_pct"] = np.where(
        df["has_peer_baseline"],
        (df["cost_per_tonne_km"] / df["peer_avg"]) - 1.0,
        np.nan,
    )

    # Alert flags
    hist_flag = df["history_increase_pct"] > threshold
    peer_flag = df["peer_increase_pct"] > threshold

    df["flagged"] = hist_flag.fillna(False) | peer_flag.fillna(False)

    # Ensure chronological order within each route to compute consecutive streaks
    route_keys = [c for c in ["origin", "destination", "route_type"] if c in df.columns]
    if route_keys and "week_of" in df.columns:
        df["_week_dt"] = pd.to_datetime(df["week_of"], errors="coerce")
        df = df.sort_values(route_keys + ["_week_dt"]).reset_index(drop=True)
        df = df.drop(columns=["_week_dt"])

        def _calc_consecutive(series):
            counts = []
            curr = 0
            for val in series:
                if bool(val):
                    curr += 1
                else:
                    curr = 0
                counts.append(curr)
            return pd.Series(counts, index=series.index)

        grouped = df.groupby(route_keys)
        df["consecutive_flagged_weeks"] = grouped["flagged"].transform(_calc_consecutive)
        df["consecutive_peer_weeks"] = grouped[peer_flag.name if peer_flag.name in df.columns else "flagged"].transform(
            lambda s: _calc_consecutive(df.loc[s.index, "peer_increase_pct"] > threshold)
        )
        df["consecutive_hist_weeks"] = grouped["flagged"].transform(
            lambda s: _calc_consecutive(df.loc[s.index, "history_increase_pct"] > threshold)
        )
    else:
        df["consecutive_flagged_weeks"] = np.where(df["flagged"], 1, 0)
        df["consecutive_peer_weeks"] = np.where(peer_flag.fillna(False), 1, 0)
        df["consecutive_hist_weeks"] = np.where(hist_flag.fillna(False), 1, 0)

    # Specific flag reasons
    both = hist_flag.fillna(False) & peer_flag.fillna(False)
    hist_only = hist_flag.fillna(False) & ~peer_flag.fillna(False)
    peer_only = peer_flag.fillna(False) & ~hist_flag.fillna(False)

    df["flag_reason"] = ""
    df.loc[both, "flag_reason"] = "Above own history and peer baseline"
    df.loc[hist_only, "flag_reason"] = "Above own historical baseline"
    df.loc[peer_only, "flag_reason"] = "Above peer baseline"

    # Anomaly category
    conditions = [
        both,
        hist_only,
        peer_only,
        ~df["flagged"] & ~df["has_history_baseline"],
        ~df["flagged"] & df["has_history_baseline"],
    ]
    choices = [
        "BOTH_BASELINES",
        "HISTORY_ONLY",
        "PEER_ONLY",
        "NORMAL_INSUFFICIENT_HISTORY",
        "NORMAL_WITHIN_THRESHOLD",
    ]
    df["anomaly_category"] = np.select(conditions, choices, default="NORMAL_WITHIN_THRESHOLD")

    df["verdict"] = np.where(df["flagged"], "review", "normal")

    return df
