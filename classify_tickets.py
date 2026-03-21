"""
classify_tickets.py — HubSpot ticket classification pipeline.

Reads a HubSpot ticket export CSV, calls Claude once per row to classify
each ticket across three dimensions, and writes an enriched output CSV.

Usage
-----
    python classify_tickets.py

    # Override input / output paths:
    python classify_tickets.py --tickets path/to/tickets.csv --output path/to/out.csv

    # Disable resume (reprocess all tickets from scratch):
    python classify_tickets.py --no-resume

    # Use a different Anthropic model:
    python classify_tickets.py --model claude-opus-4-5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import Config, load_config
from prompts import SYSTEM_PROMPT, build_analysis_prompt
from utils import compute_speed_percentiles

load_dotenv()

# ── Helpers ────────────────────────────────────────────────────────────────────

EMPTY_RESULT: dict = {
    "feature_request_flag": None,
    "feature_request_description": None,
    "product_quality_score": None,
    "product_quality_reason": None,
    "sentiment_score": None,
    "client_experience_quality_score": None,
    "client_experience_quality_reason": None,
}


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences that some models wrap JSON in."""
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _parse_response(raw: str) -> dict:
    """Parse Claude's raw text response into a dict, with error fallback."""
    cleaned = _strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"\n  [WARN] JSON parse failed — storing raw text in reason fields.")
        truncated = cleaned[:300]
        return {
            **EMPTY_RESULT,
            "product_quality_reason": f"PARSE_ERROR: {truncated}",
            "client_experience_quality_reason": f"PARSE_ERROR: {truncated}",
        }


# ── Core classification call ───────────────────────────────────────────────────

def classify_ticket(
    client: anthropic.Anthropic,
    transcript: str,
    speed_percentile: float | None,
    products: list[dict],
    cfg: Config,
) -> dict:
    """
    Send a single ticket to Claude and return parsed classification results.
    Retries up to cfg.max_retries times on API errors with exponential backoff.
    """
    prompt = build_analysis_prompt(transcript, speed_percentile, products)

    for attempt in range(1, cfg.max_retries + 1):
        try:
            message = client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text
            return _parse_response(raw)

        except anthropic.RateLimitError:
            wait = cfg.retry_backoff_seconds * attempt
            print(f"\n  [WARN] Rate limit hit — waiting {wait:.0f}s (attempt {attempt}/{cfg.max_retries})")
            time.sleep(wait)

        except anthropic.APIError as exc:
            wait = cfg.retry_backoff_seconds * attempt
            print(f"\n  [WARN] API error: {exc} — waiting {wait:.0f}s (attempt {attempt}/{cfg.max_retries})")
            time.sleep(wait)

    print(f"\n  [ERROR] All {cfg.max_retries} retries exhausted — storing empty result.")
    return {**EMPTY_RESULT, "product_quality_reason": "API_ERROR", "client_experience_quality_reason": "API_ERROR"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_tickets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    print(f"Loaded {len(df)} tickets from {path}")
    return df


def load_products(path: str, name_col: str, category_col: str) -> list[dict]:
    df = pd.read_csv(path, dtype=str)
    products = [
        {"product_name": row[name_col], "category": row[category_col]}
        for _, row in df.iterrows()
        if pd.notna(row.get(name_col))
    ]
    print(f"Loaded {len(products)} products from {path}")
    return products


def load_existing_results(output_path: str) -> set[int]:
    """Return a set of row indices that have actually been classified.

    A row is considered classified only if at least one of the core
    classification columns contains a non-empty value. This prevents
    rows that were written as placeholders (null values) from being
    treated as already processed.
    """
    p = Path(output_path)
    if not p.exists():
        return set()
    try:
        existing = pd.read_csv(p, dtype=str)
        if "_row_index" not in existing.columns:
            return set()
        # Only count rows where classification actually ran
        classified_mask = existing["client_experience_quality_score"].notna() & (
            existing["client_experience_quality_score"].str.strip() != ""
        )
        return set(existing.loc[classified_mask, "_row_index"].dropna().astype(int).tolist())
    except Exception:
        return set()


# ── Result columns ─────────────────────────────────────────────────────────────

RESULT_COLS = [
    "feature_request_flag",
    "feature_request_description",
    "product_quality_score",
    "product_quality_reason",
    "speed_percentile_score",
    "sentiment_score",
    "client_experience_quality_score",
    "client_experience_quality_reason",
]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(cfg: Config) -> None:
    # Validate API key
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print(
            "[ERROR] ANTHROPIC_API_KEY is not set.\n"
            "Add it to your .env file:  ANTHROPIC_API_KEY=sk-ant-..."
        )
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load data
    tickets_df = load_tickets(cfg.tickets_csv)
    products = load_products(cfg.products_csv, cfg.product_name_col, cfg.product_category_col)

    # Pre-compute speed percentiles for all tickets
    print("Computing speed percentile scores...")
    tickets_df["speed_percentile_score"] = compute_speed_percentiles(tickets_df, cfg.ttc_col)

    # Resume: find already-processed row indices
    already_done: set[int] = set()
    if cfg.resume:
        already_done = load_existing_results(cfg.output_csv)
        if already_done:
            print(f"Resume mode: skipping {len(already_done)} already-processed rows.")

    # Ensure output directory exists
    Path(cfg.output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Load or initialise output DataFrame
    output_path = Path(cfg.output_csv)
    if output_path.exists() and cfg.resume:
        output_df = pd.read_csv(output_path, dtype=str)
    else:
        output_df = pd.DataFrame()

    results_buffer: list[dict] = []

    rows_to_process = [
        (idx, row) for idx, row in tickets_df.iterrows()
        if idx not in already_done
    ]
    if cfg.limit is not None:
        rows_to_process = rows_to_process[: cfg.limit]

    print(f"\nClassifying {len(rows_to_process)} tickets...\n")

    for idx, row in tqdm(rows_to_process, unit="ticket", dynamic_ncols=True):
        transcript = str(row.get(cfg.transcript_col, "")).strip()
        if not transcript:
            tqdm.write(f"  [{idx}] Skipping — no transcript")
            continue

        speed_pct = row.get("speed_percentile_score")
        if pd.notna(speed_pct):
            try:
                speed_pct = float(speed_pct)
            except (ValueError, TypeError):
                speed_pct = None
        else:
            speed_pct = None

        result = classify_ticket(client, transcript, speed_pct, products, cfg)

        # Merge result back
        row_dict = row.to_dict()
        row_dict["_row_index"] = str(idx)
        for col in RESULT_COLS:
            if col == "speed_percentile_score":
                row_dict[col] = speed_pct
            else:
                row_dict[col] = result.get(col)

        results_buffer.append(row_dict)

        ticket_name = str(row.get(cfg.ticket_name_col, idx))[:40]
        fr = "Y" if result.get("feature_request_flag") else "N"
        pq = str(result.get("product_quality_score") or "?")
        ceq = str(result.get("client_experience_quality_score") or "?")
        sent = str(result.get("sentiment_score") or "?")
        tqdm.write(
            f"  [{idx}] {ticket_name:<40} "
            f"FR={fr}  PQ={pq:<6}  Sentiment={sent:<8}  CEQ={ceq}"
        )

        # Write checkpoint every 10 rows to avoid losing progress
        if len(results_buffer) % 10 == 0:
            output_df = _flush_results(output_df, results_buffer, cfg.output_csv, tickets_df)

    # Final flush
    if results_buffer:
        _flush_results(output_df, results_buffer, cfg.output_csv, tickets_df)

    print(f"\nDone. Results saved to {cfg.output_csv}")
    _print_summary(cfg.output_csv)


def _flush_results(
    output_df: pd.DataFrame,
    buffer: list[dict],
    output_path: str,
    original_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge buffer rows into output_df, write to disk, and return the updated DataFrame."""
    new_rows = pd.DataFrame(buffer)

    if "_row_index" in output_df.columns and not output_df.empty:
        output_df = output_df[~output_df["_row_index"].isin(new_rows["_row_index"].astype(str))]
        output_df = pd.concat([output_df, new_rows], ignore_index=True)
    else:
        output_df = new_rows

    output_df.to_csv(output_path, index=False)
    buffer.clear()
    return output_df


def _print_summary(output_path: str) -> None:
    """Print a quick summary of classification results."""
    try:
        df = pd.read_csv(output_path, dtype=str)
        print("\n=== Summary ===")

        fr = df["feature_request_flag"].str.lower().isin(["true", "1", "yes"])
        print(f"Feature requests:  {fr.sum()} / {len(df)}")

        for col, label in [
            ("product_quality_score", "Product quality"),
            ("sentiment_score", "Customer sentiment"),
            ("client_experience_quality_score", "Client experience quality"),
        ]:
            if col in df.columns:
                counts = df[col].value_counts().to_dict()
                print(f"{label}: {counts}")
    except Exception:
        pass


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify HubSpot support tickets using Claude.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = Config()
    parser.add_argument("--tickets", default=defaults.tickets_csv, help="Path to tickets CSV")
    parser.add_argument("--products", default=defaults.products_csv, help="Path to products CSV")
    parser.add_argument("--output", default=defaults.output_csv, help="Output CSV path")
    parser.add_argument("--model", default=defaults.model, help="Anthropic model to use")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip already-processed rows (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Reprocess all rows from scratch",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N tickets (useful for testing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(
        tickets_csv=args.tickets,
        products_csv=args.products,
        output_csv=args.output,
        model=args.model,
        resume=args.resume,
        limit=args.limit,
    )
    run(cfg)
