"""
Upload B2B SaaS HubSpot dataset to a HubSpot account.

Upload order: Companies → Contacts → Deals → Tickets
Associations are created in bulk after each object type is uploaded.

Control which sections run by editing RUN_SECTIONS below.
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
from hubspot.crm.companies.models import SimplePublicObjectInputForCreate as CompanyInput
from hubspot.crm.contacts.models import SimplePublicObjectInputForCreate as ContactInput
from hubspot.crm.deals.models import SimplePublicObjectInputForCreate as DealInput
from hubspot.crm.tickets.models import SimplePublicObjectInputForCreate as TicketInput
from hubspot.crm.products.models import SimplePublicObjectInputForCreate as ProductInput

load_dotenv()

TOKEN = os.environ["HUBSPOT_ACCESS_TOKEN"]
client = hubspot.Client.create(access_token=TOKEN)

DATA_DIR = os.path.expanduser("~/Downloads/archive (1)")

# Seconds to wait between individual API calls.
# HubSpot free/dev accounts allow 100 requests per 10 seconds.
SLEEP = 0.15


# --------------------------------------------------------------------------- #
# Log file — records every property mapping decision
# --------------------------------------------------------------------------- #
LOG_FILE = os.path.join(os.path.dirname(__file__), "upload_log.csv")

# --------------------------------------------------------------------------- #
# Industry mapping: CSV free-text value → HubSpot enum value
# Any value NOT in this map will be stored in the company description field
# instead of the industry enum, and flagged in the log.
# --------------------------------------------------------------------------- #
INDUSTRY_MAP = {
    "SaaS":       "COMPUTER_SOFTWARE",
    "Healthcare":  "HOSPITAL_HEALTH_CARE",
    "EdTech":      "E_LEARNING",
    "Fintech":     "FINANCIAL_SERVICES",
    "E-commerce":  "INTERNET",
}

# --------------------------------------------------------------------------- #
# Dataset stage/status → HubSpot pipeline stage label mappings
# --------------------------------------------------------------------------- #

DEAL_STAGE_LABEL_MAP = {
    "Prospecting": "Appointment Scheduled",
    "Demo Scheduled": "Presentation Scheduled",
    "Negotiation": "Contract Sent",
    "Closed Won": "Closed Won",
    "Closed Lost": "Closed Lost",
}

TICKET_STATUS_LABEL_MAP = {
    "New": "New",
    "Open": "New",
    "In Progress": "Waiting on us",
    "On Hold": "Waiting on contact",
    "Closed": "Closed",
}

TICKET_PRIORITY_MAP = {
    "Low": "LOW",
    "Medium": "MEDIUM",
    "High": "HIGH",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def get_all_hs_records(api, properties):
    """
    Paginate through all records using a CRM basic_api object.
    Returns a flat list of all records.
    """
    records = []
    after = None
    while True:
        page = api.get_page(limit=100, after=after, properties=properties)
        records.extend(page.results)
        if not page.paging or not page.paging.next:
            break
        after = page.paging.next.after
    return records


def rebuild_company_id_map(companies_df):
    """
    Fetch all companies from HubSpot and match back to CSV by name.
    Returns {local_company_id: hs_company_id}.
    """
    print("  Fetching companies from HubSpot...")
    hs_records = get_all_hs_records(client.crm.companies.basic_api, ["name"])
    name_to_hs = {r.properties.get("name", ""): r.id for r in hs_records}

    id_map = {}
    for _, row in companies_df.iterrows():
        name = safe_str(row["company_name"])
        if name in name_to_hs:
            id_map[int(row["company_id"])] = name_to_hs[name]
        else:
            print(f"    ⚠ Company not found in HubSpot: '{name}'")
    print(f"  Matched {len(id_map)}/{len(companies_df)} companies.")
    return id_map


def rebuild_contact_id_map(contacts_df):
    """
    Fetch all contacts from HubSpot and match back to CSV by email.
    Returns {local_contact_id: hs_contact_id}.
    """
    print("  Fetching contacts from HubSpot...")
    hs_records = get_all_hs_records(client.crm.contacts.basic_api, ["email"])
    email_to_hs = {r.properties.get("email", ""): r.id for r in hs_records}

    id_map = {}
    for _, row in contacts_df.iterrows():
        email = safe_str(row["email"])
        if email in email_to_hs:
            id_map[int(row["contact_id"])] = email_to_hs[email]
        else:
            print(f"    ⚠ Contact not found in HubSpot: '{email}'")
    print(f"  Matched {len(id_map)}/{len(contacts_df)} contacts.")
    return id_map


def rebuild_deal_id_map(deals_df):
    """
    Fetch all deals from HubSpot and match back to CSV by deal name.
    Returns {local_deal_id: hs_deal_id}.
    """
    print("  Fetching deals from HubSpot...")
    hs_records = get_all_hs_records(client.crm.deals.basic_api, ["dealname"])
    name_to_hs = {r.properties.get("dealname", ""): r.id for r in hs_records}

    id_map = {}
    for _, row in deals_df.iterrows():
        name = safe_str(row["deal_name"])
        if name in name_to_hs:
            id_map[int(row["deal_id"])] = name_to_hs[name]
        else:
            print(f"    ⚠ Deal not found in HubSpot: '{name}'")
    print(f"  Matched {len(id_map)}/{len(deals_df)} deals.")
    return id_map


def rebuild_ticket_id_map(tickets_df):
    """
    Fetch all tickets from HubSpot, sort by HubSpot record ID (ascending =
    creation order), and match positionally to CSV rows sorted by ticket_id.
    This works because tickets were created in sequence with no failures.
    Returns {local_ticket_id: hs_ticket_id}.
    """
    print("  Fetching tickets from HubSpot...")
    hs_records = get_all_hs_records(client.crm.tickets.basic_api, ["subject"])

    # Sort by numeric HubSpot ID — reflects creation order
    hs_records.sort(key=lambda r: int(r.id))

    # Sort CSV by ticket_id to match that same creation order
    sorted_csv = tickets_df.sort_values("ticket_id").reset_index(drop=True)

    if len(hs_records) != len(sorted_csv):
        print(
            f"    ⚠ Count mismatch: {len(hs_records)} HubSpot tickets "
            f"vs {len(sorted_csv)} CSV rows — positional matching may be off."
        )

    id_map = {}
    for i, row in sorted_csv.iterrows():
        if i < len(hs_records):
            id_map[int(row["ticket_id"])] = hs_records[i].id

    print(f"  Matched {len(id_map)}/{len(tickets_df)} tickets.")
    return id_map


def backfill_company_associations(companies_df, contacts_df, deals_df, tickets_df):
    """
    Rebuild HubSpot ID maps for all object types and create the company
    associations that were skipped during the original upload (because
    companies uploaded last / failed first time).
    """
    print("\nRebuilding ID maps from HubSpot...")
    company_id_map = rebuild_company_id_map(companies_df)
    contact_id_map = rebuild_contact_id_map(contacts_df)
    deal_id_map    = rebuild_deal_id_map(deals_df)
    ticket_id_map  = rebuild_ticket_id_map(tickets_df)

    # Contact → Company
    print("\nCreating Contact → Company associations...")
    contact_company_pairs = []
    for _, row in contacts_df.iterrows():
        local_contact_id  = int(row["contact_id"])
        local_company_id  = int(row["company_id"])
        if local_contact_id in contact_id_map and local_company_id in company_id_map:
            contact_company_pairs.append(
                (contact_id_map[local_contact_id], company_id_map[local_company_id])
            )
    batch_associate("contacts", "companies", contact_company_pairs)

    # Deal → Company
    print("Creating Deal → Company associations...")
    deal_company_pairs = []
    for _, row in deals_df.iterrows():
        local_deal_id    = int(row["deal_id"])
        local_company_id = int(row["company_id"])
        if local_deal_id in deal_id_map and local_company_id in company_id_map:
            deal_company_pairs.append(
                (deal_id_map[local_deal_id], company_id_map[local_company_id])
            )
    batch_associate("deals", "companies", deal_company_pairs)

    # Ticket → Company
    print("Creating Ticket → Company associations...")
    ticket_company_pairs = []
    for _, row in tickets_df.iterrows():
        local_ticket_id  = int(row["ticket_id"])
        local_company_id = int(row["associated_company_id"])
        if local_ticket_id in ticket_id_map and local_company_id in company_id_map:
            ticket_company_pairs.append(
                (ticket_id_map[local_ticket_id], company_id_map[local_company_id])
            )
    batch_associate("tickets", "companies", ticket_company_pairs)

    print(
        f"\nBackfill complete:"
        f"\n  Contact → Company pairs: {len(contact_company_pairs)}"
        f"\n  Deal    → Company pairs: {len(deal_company_pairs)}"
        f"\n  Ticket  → Company pairs: {len(ticket_company_pairs)}"
    )


def fetch_pipeline_stages(object_type):
    """
    Returns (stage_label_to_id dict, pipeline_id) for the first pipeline
    of the given object type.
    """
    pipelines = client.crm.pipelines.pipelines_api.get_all(object_type)
    if not pipelines.results:
        raise ValueError(f"No pipelines found for object type: {object_type}")
    pipeline = pipelines.results[0]
    stage_map = {stage.label: stage.id for stage in pipeline.stages}
    return stage_map, pipeline.id


def batch_associate(from_type, to_type, pairs):
    """
    Create default (unlabeled) associations in bulk via the v4 API.
    pairs: list of (from_hubspot_id, to_hubspot_id)
    Processes in batches of 2,000 (the API maximum).
    """
    if not pairs:
        return

    batch_size = 2000
    url = (
        f"https://api.hubapi.com/crm/v4/associations"
        f"/{from_type}/{to_type}/batch/associate/default"
    )
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        payload = {
            "inputs": [
                {"from": {"id": str(f)}, "to": {"id": str(t)}}
                for f, t in batch
            ]
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        time.sleep(SLEEP)
        print(f"    Associated {len(batch)} {from_type} → {to_type}")


def safe_str(value, default=""):
    """Convert a value to string, returning default if NaN/None."""
    if pd.isna(value):
        return default
    return str(value).strip()


_log_writer = None
_log_file_handle = None

def init_log():
    """Open the log file and write the header row."""
    global _log_writer, _log_file_handle
    _log_file_handle = open(LOG_FILE, "w", newline="", encoding="utf-8")
    _log_writer = csv.writer(_log_file_handle)
    _log_writer.writerow([
        "timestamp", "object_type", "record_name",
        "property", "original_value", "mapped_to", "outcome"
    ])


def log_entry(object_type, record_name, property_name, original_value, mapped_to, outcome):
    """
    Write one row to the log.
    outcome: "mapped" | "fallback_to_description" | "skipped"
    """
    if _log_writer:
        _log_writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            object_type, record_name, property_name,
            original_value, mapped_to, outcome,
        ])


def close_log():
    if _log_file_handle:
        _log_file_handle.close()


def to_hs_timestamp(date_str):
    """
    Convert a YYYY-MM-DD string to a millisecond epoch timestamp,
    which is what HubSpot datetime properties expect.
    """
    return int(pd.Timestamp(date_str).timestamp() * 1000)


# --------------------------------------------------------------------------- #
# Upload functions
# --------------------------------------------------------------------------- #

def upload_companies(df):
    print(f"\nUploading {len(df)} companies...")
    id_map = {}  # local company_id → HubSpot record id

    for _, row in df.iterrows():
        company_name = safe_str(row["company_name"])
        raw_industry = safe_str(row["industry"])
        hs_industry = INDUSTRY_MAP.get(raw_industry)

        props = {
            "name": company_name,
            "annualrevenue": safe_str(row["annual_revenue"]),
            "numberofemployees": str(int(row["num_employees"])),
            "country": safe_str(row["country"]),
        }

        if hs_industry:
            # We have a valid HubSpot enum value — use it
            props["industry"] = hs_industry
            # Still record the original label in description so nothing is lost
            props["description"] = f"Original industry: {raw_industry}"
            log_entry("company", company_name, "industry", raw_industry, hs_industry, "mapped")
            print(f"  ✓ Company: {company_name}  [{raw_industry} → {hs_industry}]")
        else:
            # No mapping — store the raw value in description instead
            props["description"] = f"Original industry: {raw_industry}"
            log_entry("company", company_name, "industry", raw_industry, "description field", "fallback_to_description")
            print(f"  ⚠ Company: {company_name}  [{raw_industry} → no enum match, stored in description]")

        try:
            result = client.crm.companies.basic_api.create(
                simple_public_object_input_for_create=CompanyInput(properties=props)
            )
            id_map[int(row["company_id"])] = result.id
        except Exception as e:
            print(f"  ✗ Failed to create company '{company_name}': {e}")
            log_entry("company", company_name, "industry", raw_industry, "N/A", f"error: {e}")

        time.sleep(SLEEP)

    print(f"  → {len(id_map)} companies created.")
    return id_map


def upload_contacts(df, company_id_map):
    print(f"\nUploading {len(df)} contacts...")
    id_map = {}  # local contact_id → HubSpot record id
    assoc_pairs = []  # (contact_hs_id, company_hs_id)

    for _, row in df.iterrows():
        props = {
            "firstname": safe_str(row["first_name"]),
            "lastname": safe_str(row["last_name"]),
            "email": safe_str(row["email"]),
            "phone": safe_str(row["phone"]),
            "jobtitle": safe_str(row["job_title"]),
        }
        try:
            result = client.crm.contacts.basic_api.create(
                simple_public_object_input_for_create=ContactInput(properties=props)
            )
            contact_hs_id = result.id
            id_map[int(row["contact_id"])] = contact_hs_id
            print(f"  ✓ Contact: {props['firstname']} {props['lastname']}")

            local_company_id = int(row["company_id"])
            if local_company_id in company_id_map:
                assoc_pairs.append((contact_hs_id, company_id_map[local_company_id]))

        except Exception as e:
            print(f"  ✗ Failed to create contact '{props['email']}': {e}")

        time.sleep(SLEEP)

    print(f"  → {len(id_map)} contacts created.")
    batch_associate("contacts", "companies", assoc_pairs)
    return id_map


def upload_deals(df, company_id_map, contact_id_map, stage_map, pipeline_id):
    print(f"\nUploading {len(df)} deals...")
    id_map = {}
    company_pairs = []
    contact_pairs = []

    for _, row in df.iterrows():
        dataset_stage = safe_str(row["stage"])
        hs_label = DEAL_STAGE_LABEL_MAP.get(dataset_stage, "Appointment Scheduled")
        hs_stage_id = stage_map.get(hs_label, list(stage_map.values())[0])

        props = {
            "dealname": safe_str(row["deal_name"]),
            "amount": safe_str(row["amount"]),
            "dealstage": hs_stage_id,
            "pipeline": pipeline_id,
            "closedate": str(to_hs_timestamp(row["close_date"])),
        }
        try:
            result = client.crm.deals.basic_api.create(
                simple_public_object_input_for_create=DealInput(properties=props)
            )
            deal_hs_id = result.id
            id_map[int(row["deal_id"])] = deal_hs_id
            print(f"  ✓ Deal: {props['dealname']} ({dataset_stage})")

            local_company_id = int(row["company_id"])
            if local_company_id in company_id_map:
                company_pairs.append((deal_hs_id, company_id_map[local_company_id]))

            local_contact_id = int(row["contact_id"])
            if local_contact_id in contact_id_map:
                contact_pairs.append((deal_hs_id, contact_id_map[local_contact_id]))

        except Exception as e:
            print(f"  ✗ Failed to create deal '{row['deal_name']}': {e}")

        time.sleep(SLEEP)

    print(f"  → {len(id_map)} deals created.")
    batch_associate("deals", "companies", company_pairs)
    batch_associate("deals", "contacts", contact_pairs)
    return id_map


def upload_products(df):
    """
    Upload products to the HubSpot product library.
    HubSpot has no built-in category field, so the original category value
    is stored in the description property and logged.

    NOTE: Requires the 'crm.objects.products.write' scope on your Private App.
    If you get a 403 error, add that scope in HubSpot → Settings → Private Apps.
    """
    print(f"\nUploading {len(df)} products...")
    id_map = {}

    for _, row in df.iterrows():
        product_name = safe_str(row["product_name"])
        category     = safe_str(row["category"])

        props = {
            "name":        product_name,
            "price":       safe_str(row["price"]),
            "description": f"Category: {category}",
        }

        try:
            result = client.crm.products.basic_api.create(
                simple_public_object_input_for_create=ProductInput(properties=props)
            )
            id_map[int(row["product_id"])] = result.id
            log_entry(
                "product", product_name, "category", category,
                "description field", "fallback_to_description"
            )
            print(f"  ✓ Product: {product_name}  [category '{category}' → description]")
        except Exception as e:
            print(f"  ✗ Failed to create product '{product_name}': {e}")
            log_entry("product", product_name, "category", category, "N/A", f"error: {e}")

        time.sleep(SLEEP)

    print(f"  → {len(id_map)} products created.")
    return id_map


def upload_tickets(df, contact_id_map, company_id_map, stage_map, pipeline_id):
    print(f"\nUploading {len(df)} tickets...")
    id_map = {}
    contact_pairs = []
    company_pairs = []

    for _, row in df.iterrows():
        dataset_status = safe_str(row["status"])
        hs_label = TICKET_STATUS_LABEL_MAP.get(dataset_status, "New")
        hs_stage_id = stage_map.get(hs_label, list(stage_map.values())[0])

        props = {
            "subject": safe_str(row["subject"]),
            "hs_pipeline": pipeline_id,
            "hs_pipeline_stage": hs_stage_id,
            "hs_ticket_priority": TICKET_PRIORITY_MAP.get(
                safe_str(row["priority"]), "MEDIUM"
            ),
        }
        try:
            result = client.crm.tickets.basic_api.create(
                simple_public_object_input_for_create=TicketInput(properties=props)
            )
            ticket_hs_id = result.id
            id_map[int(row["ticket_id"])] = ticket_hs_id
            print(f"  ✓ Ticket: {props['subject'][:60]}")

            local_contact_id = int(row["associated_contact_id"])
            if local_contact_id in contact_id_map:
                contact_pairs.append((ticket_hs_id, contact_id_map[local_contact_id]))

            local_company_id = int(row["associated_company_id"])
            if local_company_id in company_id_map:
                company_pairs.append((ticket_hs_id, company_id_map[local_company_id]))

        except Exception as e:
            print(f"  ✗ Failed to create ticket '{row['subject']}': {e}")

        time.sleep(SLEEP)

    print(f"  → {len(id_map)} tickets created.")
    batch_associate("tickets", "contacts", contact_pairs)
    batch_associate("tickets", "companies", company_pairs)
    return id_map


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload HubSpot dataset. Pass one or more flags to choose what to upload."
    )
    parser.add_argument("--products",             action="store_true", help="Upload products")
    parser.add_argument("--companies",            action="store_true", help="Upload companies")
    parser.add_argument("--contacts",             action="store_true", help="Upload contacts (+ contact→company associations)")
    parser.add_argument("--deals",                action="store_true", help="Upload deals (+ deal→company and deal→contact associations)")
    parser.add_argument("--tickets",              action="store_true", help="Upload tickets (+ ticket→contact and ticket→company associations)")
    parser.add_argument("--backfill-associations", action="store_true", help="Re-create company associations for already-uploaded contacts, deals, and tickets")
    parser.add_argument("--all",                  action="store_true", help="Run all sections in the correct order")
    args = parser.parse_args()

    RUN_SECTIONS = set()
    if args.all:
        RUN_SECTIONS = {"products", "companies", "contacts", "deals", "tickets"}
    else:
        if args.products:             RUN_SECTIONS.add("products")
        if args.companies:            RUN_SECTIONS.add("companies")
        if args.contacts:             RUN_SECTIONS.add("contacts")
        if args.deals:                RUN_SECTIONS.add("deals")
        if args.tickets:              RUN_SECTIONS.add("tickets")
        if args.backfill_associations: RUN_SECTIONS.add("backfill_associations")

    if not RUN_SECTIONS:
        parser.print_help()
        raise SystemExit("\nNo sections selected. Pass at least one flag, e.g. --products")

    print(f"Running: {sorted(RUN_SECTIONS)}")
    print(f"Log file: {LOG_FILE}\n")

    init_log()

    try:
        print("Loading CSV files...")
        companies_df = pd.read_csv(f"{DATA_DIR}/companies.csv")
        contacts_df  = pd.read_csv(f"{DATA_DIR}/contacts.csv")
        deals_df     = pd.read_csv(f"{DATA_DIR}/deals.csv")
        tickets_df   = pd.read_csv(f"{DATA_DIR}/tickets.csv")
        products_df  = pd.read_csv(f"{DATA_DIR}/products.csv")

        company_id_map = {}
        contact_id_map = {}
        deal_id_map    = {}
        ticket_id_map  = {}
        product_id_map = {}

        if "products" in RUN_SECTIONS:
            product_id_map = upload_products(products_df)

        if "companies" in RUN_SECTIONS:
            company_id_map = upload_companies(companies_df)

        if "contacts" in RUN_SECTIONS:
            contact_id_map = upload_contacts(contacts_df, company_id_map)

        if "deals" in RUN_SECTIONS or "tickets" in RUN_SECTIONS:
            print("\nFetching HubSpot pipeline stages...")
            deal_stage_map, deal_pipeline_id = fetch_pipeline_stages("deals")
            ticket_stage_map, ticket_pipeline_id = fetch_pipeline_stages("tickets")
            print(f"  Deal stages:   {list(deal_stage_map.keys())}")
            print(f"  Ticket stages: {list(ticket_stage_map.keys())}")

        if "deals" in RUN_SECTIONS:
            deal_id_map = upload_deals(
                deals_df, company_id_map, contact_id_map, deal_stage_map, deal_pipeline_id
            )

        if "tickets" in RUN_SECTIONS:
            ticket_id_map = upload_tickets(
                tickets_df, contact_id_map, company_id_map, ticket_stage_map, ticket_pipeline_id
            )

        if "backfill_associations" in RUN_SECTIONS:
            backfill_company_associations(companies_df, contacts_df, deals_df, tickets_df)

    finally:
        close_log()

    print("\n--- Done ---")
    if "products" in RUN_SECTIONS:
        print(f"  Products uploaded:  {len(product_id_map)}")
    if "companies" in RUN_SECTIONS:
        print(f"  Companies uploaded: {len(company_id_map)}")
    if "contacts" in RUN_SECTIONS:
        print(f"  Contacts uploaded:  {len(contact_id_map)}")
    if "deals" in RUN_SECTIONS:
        print(f"  Deals uploaded:     {len(deal_id_map)}")
    if "tickets" in RUN_SECTIONS:
        print(f"  Tickets uploaded:   {len(ticket_id_map)}")
    print(f"\nFull property mapping log saved to: {LOG_FILE}")
