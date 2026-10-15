
import pandas as pd
import numpy as np


# Input CSV column names-ai standard names-ah map pannum
ALIASES = {
    "date": [
        "shipment_date",
        "date",
        "ship_date",
    ],
    "origin": [
        "origin",
        "source",
        "from_location",
        "origin_city",
    ],
    "destination": [
        "destination",
        "dest",
        "to_location",
        "destination_city",
    ],
    "route_type": [
        "route_type",
        "route_category",
        "distance_type",
    ],
    "tonnes": [
        "quantity_tonnes",
        "tonnes",
        "weight_tonnes",
        "load_tonnes",
    ],
    "distance": [
        "distance_km",
        "distance",
        "km",
    ],
    "cost": [
        "freight_cost_inr",
        "freight_cost",
        "total_cost",
        "shipping_cost",
        "cost",
    ],
}


def find_column(df, candidates, required=True):
    """
    CSV-la irukkura column name-ai case-insensitive-ah identify pannum.
    """
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(
            f"Could not find any of {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def standardize_shipments(raw):
    """
    Raw shipment data-ai standard column names and valid numeric values
    ulla DataFrame-ah convert pannum.
    """
    if raw.empty:
        raise ValueError("Shipment input CSV is empty.")

    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]

    mapping = {}

    for target, candidates in ALIASES.items():
        column = find_column(
            df,
            candidates,
            required=(target != "route_type"),
        )

        if column is not None:
            mapping[column] = target

    df = df.rename(columns=mapping)

    # route_type input-la illaina Unknown use pannuvom
    if "route_type" not in df.columns:
        df["route_type"] = "Unknown"

    # Date conversion
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    # Numeric columns conversion
    for column in ["tonnes", "distance", "cost"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Empty and invalid mandatory values remove pannuvom
    df = df.dropna(
        subset=[
            "date",
            "origin",
            "destination",
            "tonnes",
            "distance",
            "cost",
        ]
    ).copy()

    # Invalid quantity, distance, or negative freight cost remove pannuvom
    df = df[
        (df["tonnes"] > 0)
        & (df["distance"] > 0)
        & (df["cost"] >= 0)
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid shipment rows remain after data cleaning."
        )

    # Normalize text columns
    for column in ["origin", "destination"]:
        df[column] = df[column].astype(str).str.strip()

    df["route_type"] = (
        df["route_type"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    # Week start: Monday
    df["week_of"] = (
        df["date"].dt.normalize()
        - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    )

    # Tonne-kilometres = shipment tonnes * distance in km
    df["tonne_km"] = df["tonnes"] * df["distance"]

    return df


def build_weekly_metrics(raw):
    """
    Calculate:
    1. Weekly route cost per tonne-km
    2. Previous 8 available weekly observations baseline
    3. Same-week, same-route-type peer baseline excluding own route
    """
    df = standardize_shipments(raw)

    route_keys = [
        "origin",
        "destination",
        "route_type",
    ]

    week_keys = route_keys + ["week_of"]

    # --------------------------------------------------
    # 1. Aggregate shipments by route and week
    # --------------------------------------------------
    weekly = (
        df.groupby(week_keys, as_index=False)
        .agg(
            total_cost=("cost", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("cost", "size"),
        )
    )

    # Cost per tonne-km
    weekly["cost_per_tonne_km"] = np.where(
        weekly["total_tonne_km"] > 0,
        weekly["total_cost"] / weekly["total_tonne_km"],
        np.nan,
    )

    weekly = weekly.sort_values(
        route_keys + ["week_of"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 2. Own-route historical baseline
    # --------------------------------------------------
    # shift(1) ensures current week is excluded.
    # Rolling window uses the previous 8 available weekly
    # observations, not necessarily 8 calendar weeks.
    route_rate = weekly.groupby(
        route_keys
    )["cost_per_tonne_km"]

    weekly["history_avg"] = route_rate.transform(
        lambda series: (
            series.shift(1)
            .rolling(window=8, min_periods=1)
            .mean()
        )
    )

    weekly["history_weeks_used"] = route_rate.transform(
        lambda series: (
            series.shift(1)
            .rolling(window=8, min_periods=1)
            .count()
        )
    )

    # --------------------------------------------------
    # 3. Peer baseline for same week and route type
    # --------------------------------------------------
    # Use totals so the peer baseline is weighted by
    # each route's tonne-km volume.
    peer_keys = ["route_type", "week_of"]

    peer_totals = (
        weekly.groupby(peer_keys, as_index=False)
        .agg(
            peer_group_cost=("total_cost", "sum"),
            peer_group_tonne_km=("total_tonne_km", "sum"),
            peer_route_count=("cost_per_tonne_km", "count"),
        )
    )

    weekly = weekly.merge(
        peer_totals,
        on=peer_keys,
        how="left",
        validate="many_to_one",
    )

    # Remove the current route from the peer totals
    other_route_cost = (
        weekly["peer_group_cost"] - weekly["total_cost"]
    )

    other_route_tonne_km = (
        weekly["peer_group_tonne_km"]
        - weekly["total_tonne_km"]
    )

    weekly["peer_count"] = (
        weekly["peer_route_count"] - 1
    ).clip(lower=0)

    # No other route means no valid peer baseline
    weekly["peer_avg"] = np.where(
        (weekly["peer_count"] > 0)
        & (other_route_tonne_km > 0),
        other_route_cost / other_route_tonne_km,
        np.nan,
    )

    # Remove intermediate calculation columns
    weekly = weekly.drop(
        columns=[
            "peer_group_cost",
            "peer_group_tonne_km",
            "peer_route_count",
        ]
    )

    return weekly
