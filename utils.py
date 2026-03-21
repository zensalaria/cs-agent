"""Utility functions for ticket data processing."""

from __future__ import annotations

import pandas as pd


def parse_ttc_to_seconds(ttc_str: str) -> float | None:
    """
    Parse a time-to-close string in H:MM:SS format to total seconds.

    Returns None if the value is missing, zero, or unparseable.
    Zero-duration tickets (TTC = 0:00:00) are excluded from percentile
    calculation as they are likely data artefacts.
    """
    if pd.isna(ttc_str) or str(ttc_str).strip() == "":
        return None

    ttc_str = str(ttc_str).strip()

    try:
        parts = ttc_str.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
        else:
            return None

        total = hours * 3600 + minutes * 60 + seconds
        return float(total) if total > 0 else None
    except (ValueError, AttributeError):
        return None


def compute_speed_percentiles(df: pd.DataFrame, ttc_col: str = "TTC_generated") -> pd.Series:
    """
    Compute a speed percentile score for each ticket.

    The percentile represents what percentage of tickets took *equal or less*
    time to close than the given ticket.

    - Lower percentile  → faster resolution → better for the customer
    - Higher percentile → slower resolution → worse for the customer
    - Tickets with missing or zero TTC receive NaN

    Args:
        df: DataFrame containing the TTC column.
        ttc_col: Name of the column holding the HH:MM:SS time-to-close string.

    Returns:
        A Series of float percentile scores (0–100), aligned to df's index.
    """
    ttc_seconds = df[ttc_col].apply(parse_ttc_to_seconds)

    valid_mask = ttc_seconds.notna()
    valid_values = ttc_seconds[valid_mask]

    percentiles = ttc_seconds.copy().astype(float)

    if valid_values.empty:
        return percentiles

    n = len(valid_values)
    for idx in valid_values.index:
        val = valid_values[idx]
        count_lte = (valid_values <= val).sum()
        percentiles[idx] = round((count_lte / n) * 100, 1)

    return percentiles
