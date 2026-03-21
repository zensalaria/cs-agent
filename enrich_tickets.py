"""
Enrich HubSpot tickets from a generated CSV export.

Writes:
  query_transcript        → content  (ticket description, native property)
  TTC_generated           → custom_ttc_generated  (custom property)
  generated create date   → custom_create_date_generated  (custom property)

Also creates 100 new tickets from rows where Ticket ID = "gpt_generated".

Usage:
  python3 enrich_tickets.py --create-properties   # run once to create custom properties
  python3 enrich_tickets.py --update-existing     # patch 500 existing tickets
  python3 enrich_tickets.py --create-new          # create 100 new gpt_generated tickets
  python3 enrich_tickets.py --all                 # all three steps in order

NOTE: --create-properties requires the 'crm.schemas.tickets.write' scope on
your Private App (Settings → Private Apps → Auth → Scopes).
"""

import argparse
import csv
import os
import time
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv
import hubspot
from hubspot.crm.tickets.models import SimplePublicObjectInputForCreate as TicketInput

load_dotenv()

TOKEN  = os.environ["HUBSPOT_ACCESS_TOKEN"]
client = hubspot.Client.create(access_token=TOKEN)

CSV_PATH = os.path.expanduser("~/Downloads/Example tickets - final.csv")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enrich_log.csv")
SLEEP    = 0.15

# Internal names for the two custom properties we create
PROP_TTC         = "custom_ttc_generated"
PROP_CREATE_DATE = "custom_create_date_generated"

TICKET_STATUS_LABEL_MAP = {
    "New":               "New",
    "Open":              "New",
    "In Progress":       "Waiting on us",
    "On Hold":           "Waiting on contact",
    "Waiting on contact": "Waiting on contact",
    "Waiting on us":     "Waiting on us",
    "Closed":            "Closed",
}

TICKET_PRIORITY_MAP = {
    "Low":    "LOW",
    "Medium": "MEDIUM",
    "High":   "HIGH",
}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_log_writer      = None
_log_file_handle = None


def init_log():
    global _log_writer, _log_file_handle
    _log_file_handle = open(LOG_FILE, "w", newline="", encoding="utf-8")
    _log_writer = csv.writer(_log_file_handle)
    _log_writer.writerow(["timestamp", "action", "csv_subject", "hs_id", "outcome"])


def log_entry(action, csv_subject, hs_id, outcome):
    if _log_writer:
        _log_writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            action, csv_subject, hs_id, outcome,
        ])


def close_log():
    if _log_file_handle:
        _log_file_handle.close()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def safe_str(value, default=""):
    if pd.isna(value):
        return default
    return str(value).strip()


def fetch_pipeline_stages(object_type):
    pipelines = client.crm.pipelines.pipelines_api.get_all(object_type)
    if not pipelines.results:
        raise ValueError(f"No pipelines found for {object_type}")
    pipeline = pipelines.results[0]
    return {stage.label: stage.id for stage in pipeline.stages}, pipeline.id


def get_all_hs_tickets():
    """Paginate through all HubSpot tickets, returning records with subject property."""
    records = []
    after   = None
    while True:
        page = client.crm.tickets.basic_api.get_page(
            limit=100, after=after, properties=["subject"]
        )
        records.extend(page.results)
        if not page.paging or not page.paging.next:
            break
        after = page.paging.next.after
    return records


# --------------------------------------------------------------------------- #
# Step 1 — Create custom properties
# --------------------------------------------------------------------------- #

def create_custom_properties():
    """
    Create custom ticket properties in HubSpot.
    Safe to re-run — 409 Conflict (already exists) is treated as success.
    Requires crm.schemas.tickets.write scope.
    """
    print("\nCreating custom ticket properties...")
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type":  "application/json",
    }
    url = "https://api.hubapi.com/crm/v3/properties/tickets"

    definitions = [
        {
            "name":        PROP_TTC,
            "label":       "Generated Time to Close",
            "type":        "string",
            "fieldType":   "text",
            "groupName":   "ticketinformation",
            "description": "AI-generated time to close from enrichment CSV (H:MM:SS format)",
        },
        {
            "name":        PROP_CREATE_DATE,
            "label":       "Generated Create Date",
            "type":        "string",
            "fieldType":   "text",
            "groupName":   "ticketinformation",
            "description": "AI-generated create date from enrichment CSV (DD/MM/YYYY HH:MM format)",
        },
    ]

    for defn in definitions:
        resp = requests.post(url, json=defn, headers=headers)
        if resp.status_code == 201:
            print(f"  ✓ Created:         {defn['name']} (\"{defn['label']}\")")
        elif resp.status_code == 409:
            print(f"  ℹ Already exists:  {defn['name']}")
        else:
            print(
                f"  ✗ Failed to create '{defn['name']}': "
                f"{resp.status_code} — {resp.text[:200]}"
            )
        time.sleep(SLEEP)


# --------------------------------------------------------------------------- #
# Step 2 — Update existing tickets
# --------------------------------------------------------------------------- #

def update_existing_tickets(df):
    """
    Match CSV rows to HubSpot tickets using positional matching:
      - Fetch all HubSpot tickets, sort by record ID ascending (= creation order)
      - Sort CSV rows by their file position
      - Zip them together 1-to-1

    Subject-based matching is not used here because every ticket subject in
    the dataset appears 16–44 times, making unique matching impossible.
    Any row where the subjects don't match after pairing is flagged in the log.
    """
    existing_df = df[df["Ticket ID"] != "gpt_generated"].reset_index(drop=True)
    print(f"\nUpdating {len(existing_df)} existing tickets (positional matching)...")

    print("  Fetching all tickets from HubSpot...")
    hs_tickets = get_all_hs_tickets()
    # Sort by numeric HubSpot ID ascending = creation order
    hs_tickets.sort(key=lambda r: int(r.id))
    print(f"  {len(hs_tickets)} tickets found in HubSpot.")

    if len(hs_tickets) != len(existing_df):
        print(
            f"  ⚠ Count mismatch: {len(hs_tickets)} HubSpot tickets vs "
            f"{len(existing_df)} CSV rows — some rows may not be matched."
        )

    updates       = []
    mismatch_count = 0

    for i, row in existing_df.iterrows():
        if i >= len(hs_tickets):
            log_entry("update", safe_str(row["Ticket name"]), "", "no_hs_record")
            continue

        hs_record = hs_tickets[i]
        hs_id     = hs_record.id
        hs_subj   = (hs_record.properties.get("subject") or "").strip()
        csv_subj  = safe_str(row["Ticket name"])

        if hs_subj != csv_subj:
            # Subjects don't match at this position — log but still update
            # (the subjects repeat heavily, so minor ordering differences are expected)
            log_entry("update", csv_subj, hs_id, f"subject_mismatch:hs='{hs_subj}'")
            mismatch_count += 1

        props = {
            "content":        safe_str(row["query_transcript"]),
            PROP_TTC:         safe_str(row["TTC_generated"]),
            PROP_CREATE_DATE: safe_str(row["generated create date"]),
        }
        updates.append({"id": hs_id, "properties": props})
        log_entry("update", csv_subj, hs_id, "updated")

    # Batch PATCH in groups of 100 (HubSpot batch update limit)
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type":  "application/json",
    }
    batch_url  = "https://api.hubapi.com/crm/v3/objects/tickets/batch/update"
    batch_size = 100

    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]
        resp  = requests.post(batch_url, json={"inputs": batch}, headers=headers)
        if resp.status_code in (200, 207):
            print(f"  ✓ Batch patched {len(batch)} tickets  (rows {i+1}–{i+len(batch)})")
        else:
            print(
                f"  ✗ Batch update failed: "
                f"{resp.status_code} — {resp.text[:300]}"
            )
        time.sleep(SLEEP)

    print(f"\n  Results:")
    print(f"    Patched:                    {len(updates)}")
    print(f"    Subject mismatches (logged): {mismatch_count}")
    if mismatch_count:
        print(f"    → See {LOG_FILE} for subject mismatch details.")


# --------------------------------------------------------------------------- #
# Step 3 — Create new tickets
# --------------------------------------------------------------------------- #

def create_new_tickets(df):
    new_df = df[df["Ticket ID"] == "gpt_generated"].copy()
    print(f"\nCreating {len(new_df)} new tickets...")

    stage_map, pipeline_id = fetch_pipeline_stages("tickets")
    created = 0

    for _, row in new_df.iterrows():
        subject      = safe_str(row["Ticket name"])
        dataset_status = safe_str(row["Ticket status"])
        hs_label     = TICKET_STATUS_LABEL_MAP.get(dataset_status, "New")
        hs_stage_id  = stage_map.get(hs_label, list(stage_map.values())[0])

        props = {
            "subject":          subject,
            "hs_pipeline":      pipeline_id,
            "hs_pipeline_stage": hs_stage_id,
            "hs_ticket_priority": TICKET_PRIORITY_MAP.get(
                safe_str(row["Priority"]), "MEDIUM"
            ),
            "content":          safe_str(row["query_transcript"]),
            PROP_TTC:           safe_str(row["TTC_generated"]),
            PROP_CREATE_DATE:   safe_str(row["generated create date"]),
        }

        try:
            result = client.crm.tickets.basic_api.create(
                simple_public_object_input_for_create=TicketInput(properties=props)
            )
            created += 1
            log_entry("create", subject, result.id, "created")
            print(f"  ✓ {subject[:75]}")
        except Exception as e:
            print(f"  ✗ {subject[:75]}: {e}")
            log_entry("create", subject, "", f"error: {e}")

        time.sleep(SLEEP)

    print(f"  → {created}/{len(new_df)} new tickets created.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich HubSpot tickets from generated CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 enrich_tickets.py --create-properties\n"
            "  python3 enrich_tickets.py --update-existing\n"
            "  python3 enrich_tickets.py --create-new\n"
            "  python3 enrich_tickets.py --all\n"
        ),
    )
    parser.add_argument(
        "--create-properties", action="store_true",
        help="Create custom HubSpot ticket properties (run once; needs crm.schemas.tickets.write scope)"
    )
    parser.add_argument(
        "--update-existing", action="store_true",
        help="Patch 500 existing tickets with transcript + generated fields"
    )
    parser.add_argument(
        "--create-new", action="store_true",
        help="Create 100 new tickets from gpt_generated rows"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run --create-properties, --update-existing, and --create-new in order"
    )
    args = parser.parse_args()

    if args.all:
        args.create_properties = True
        args.update_existing   = True
        args.create_new        = True

    if not any([args.create_properties, args.update_existing, args.create_new]):
        parser.print_help()
        raise SystemExit("\nNo action selected. Pass at least one flag, e.g. --all")

    init_log()
    try:
        df = None
        if args.update_existing or args.create_new:
            print(f"Loading CSV: {CSV_PATH}")
            df = pd.read_csv(CSV_PATH)
            print(f"  {len(df)} rows loaded  "
                  f"({(df['Ticket ID'] != 'gpt_generated').sum()} existing, "
                  f"{(df['Ticket ID'] == 'gpt_generated').sum()} new)\n")

        if args.create_properties:
            create_custom_properties()

        if args.update_existing:
            update_existing_tickets(df)

        if args.create_new:
            create_new_tickets(df)

    finally:
        close_log()

    print(f"\nLog saved to: {LOG_FILE}")
