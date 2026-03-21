"""Pipeline configuration.

All settings can be overridden via CLI flags in classify_tickets.py.
Default paths assume the script is run from the project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Input / output ────────────────────────────────────────────────────────
    tickets_csv: str = str(
        Path.home() / "Downloads" / "Example tickets - final.csv"
    )
    products_csv: str = str(
        Path.home() / "Downloads" / "archive (1)" / "products.csv"
    )
    output_csv: str = "output/ticket_analysis.csv"

    # ── Anthropic model ───────────────────────────────────────────────────────
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024

    # ── Retry / rate-limit handling ───────────────────────────────────────────
    max_retries: int = 3
    retry_backoff_seconds: float = 5.0

    # ── Resume behaviour ──────────────────────────────────────────────────────
    # If True, existing rows in output_csv are skipped (useful for reruns)
    resume: bool = True

    # ── Row limit (for test runs) ─────────────────────────────────────────────
    # Process only the first N rows. None = process all rows.
    limit: int | None = None

    # ── Column names (ticket CSV) ─────────────────────────────────────────────
    ticket_id_col: str = "Ticket ID"
    ticket_name_col: str = "Ticket name"
    transcript_col: str = "query_transcript"
    ttc_col: str = "TTC_generated"

    # ── Column names (products CSV) ───────────────────────────────────────────
    product_name_col: str = "product_name"
    product_category_col: str = "category"

    # ── Output column names ───────────────────────────────────────────────────
    out_feature_flag: str = "feature_request_flag"
    out_feature_desc: str = "feature_request_description"
    out_product_score: str = "product_quality_score"
    out_product_reason: str = "product_quality_reason"
    out_speed_percentile: str = "speed_percentile_score"
    out_sentiment: str = "sentiment_score"
    out_ceq_score: str = "client_experience_quality_score"
    out_ceq_reason: str = "client_experience_quality_reason"


def load_config(**overrides) -> Config:
    """Return a Config instance, applying any keyword overrides."""
    cfg = Config()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown config key: {key!r}")
    return cfg
