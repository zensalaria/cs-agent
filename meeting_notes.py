"""
meeting_notes.py — Process Gemini meeting notes from Google Drive.

Pulls a Google Doc, finds the calendar event to get attendee emails,
creates a HubSpot Call engagement associated with matching contacts,
drafts a follow-up email in Gmail via Claude, and saves a summary
to output/meeting_notes.json for the dashboard.

Usage
-----
    # Process the most recent Google Doc:
    python3 meeting_notes.py

    # Process a specific file by name:
    python3 meeting_notes.py --file-name "Zen / Alex - Alpaca"

    # List the 10 most recent Google Docs so you can pick one:
    python3 meeting_notes.py --list
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path

import anthropic
import requests as _requests
from dotenv import load_dotenv
from hubspot import HubSpot
from hubspot.crm.contacts.models import PublicObjectSearchRequest, Filter, FilterGroup
from hubspot.crm.objects.calls.models import (
    SimplePublicObjectInputForCreate as CallInput,
    PublicAssociationsForObject,
    AssociationSpec,
    PublicObjectId,
)

from google_auth import get_drive_service, get_gmail_service, get_calendar_service
from meeting_prompts import FOLLOW_UP_SYSTEM, build_follow_up_prompt

load_dotenv()

OUTPUT_JSON = Path("output/meeting_notes.json")


# ── Google Drive helpers ──────────────────────────────────────────────────────

def list_recent_docs(drive, max_results: int = 10) -> list[dict]:
    """Return the most recent Google Docs from Drive."""
    results = drive.files().list(
        q="mimeType='application/vnd.google-apps.document'",
        orderBy="modifiedTime desc",
        pageSize=max_results,
        fields="files(id, name, modifiedTime)",
    ).execute()
    return results.get("files", [])


def find_doc_by_name(drive, name: str) -> dict | None:
    """Search for a Google Doc whose name contains the given string."""
    escaped = name.replace("'", "\\'")
    results = drive.files().list(
        q=f"mimeType='application/vnd.google-apps.document' and name contains '{escaped}'",
        orderBy="modifiedTime desc",
        pageSize=5,
        fields="files(id, name, modifiedTime)",
    ).execute()
    files = results.get("files", [])
    return files[0] if files else None


def export_doc_text(drive, file_id: str) -> str:
    """Export a Google Doc as plain text."""
    return drive.files().export(fileId=file_id, mimeType="text/plain").execute().decode("utf-8")


# ── Google Calendar helpers ───────────────────────────────────────────────────

def _parse_datetime_from_doc_name(doc_name: str) -> datetime | None:
    """Try to extract a date/time from a Gemini notes doc name.

    Expected format fragment: '2026/03/31 15:56 BST'
    """
    match = re.search(r"(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2})", doc_name)
    if not match:
        return None
    date_str, time_str = match.group(1), match.group(2)
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M")
    except ValueError:
        return None


def find_calendar_event(calendar_service, around: datetime, window_minutes: int = 120) -> dict | None:
    """Find a calendar event near the given time. Returns the best match or None."""
    time_min = (around - timedelta(minutes=window_minutes)).isoformat() + "Z"
    time_max = (around + timedelta(minutes=window_minutes)).isoformat() + "Z"

    events_result = calendar_service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
        maxResults=10,
    ).execute()
    events = events_result.get("items", [])

    # Prefer events that have a Google Meet conference link
    for event in events:
        conf = event.get("conferenceData", {})
        entry_points = conf.get("entryPoints", [])
        if any(ep.get("entryPointType") == "video" for ep in entry_points):
            return event

    return events[0] if events else None


def get_attendee_emails(event: dict, exclude_self: bool = True) -> list[dict]:
    """Extract attendees from a calendar event.

    Returns list of dicts with 'email' and 'name' keys.
    If exclude_self is True, the organiser / self is excluded.
    """
    attendees = []
    for att in event.get("attendees", []):
        if exclude_self and att.get("self"):
            continue
        attendees.append({
            "email": att.get("email", ""),
            "name": att.get("displayName", att.get("email", "").split("@")[0]),
        })
    return attendees


# ── Claude helpers ────────────────────────────────────────────────────────────

def _strip_json_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _call_claude(client: anthropic.Anthropic, system: str, prompt: str, model: str,
                 max_retries: int = 3, backoff: float = 5.0) -> dict:
    """Send a prompt to Claude and parse the JSON response. Retries on transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text
            return json.loads(_strip_json_fences(raw))
        except (anthropic.OverloadedError, anthropic.RateLimitError, anthropic.APIError) as exc:
            wait = backoff * attempt
            print(f"  [WARN] {type(exc).__name__} — retrying in {wait:.0f}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError("Claude API: all retries exhausted.")


# ── HubSpot helpers ───────────────────────────────────────────────────────────

def _get_portal_id(access_token: str) -> str | None:
    """Fetch the HubSpot portal ID for constructing dashboard links."""
    try:
        r = _requests.get(
            "https://api.hubapi.com/account-info/v3/details",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        return str(r.json().get("portalId", ""))
    except Exception:
        return None


def _find_contact_by_email(hs: HubSpot, email: str) -> dict | None:
    """Search HubSpot for a contact by email. Returns dict with id and company, or None."""
    search_request = PublicObjectSearchRequest(
        filter_groups=[FilterGroup(filters=[
            Filter(property_name="email", operator="EQ", value=email),
        ])],
        properties=["email", "firstname", "lastname", "company"],
        limit=1,
    )
    try:
        results = hs.crm.contacts.search_api.do_search(public_object_search_request=search_request)
        if results.results:
            contact = results.results[0]
            props = contact.properties or {}
            return {
                "id": contact.id,
                "firstname": props.get("firstname", ""),
                "lastname": props.get("lastname", ""),
                "company": props.get("company", ""),
            }
    except Exception as exc:
        print(f"  [WARN] HubSpot search for {email} failed: {exc}")
    return None


def create_hubspot_call(
    access_token: str,
    notes_text: str,
    doc_name: str,
    attendee_emails: list[str],
) -> tuple[str | None, list[dict]]:
    """Create a Call engagement in HubSpot associated with matching contacts.

    Returns (call_id, matched_contacts) where matched_contacts is a list of
    dicts with id, firstname, lastname, company.
    """
    hs = HubSpot(access_token=access_token)

    matched_contacts = []
    for email in attendee_emails:
        contact = _find_contact_by_email(hs, email)
        if contact:
            matched_contacts.append(contact)
            print(f"  Matched: {email} → {contact['firstname']} {contact['lastname']} ({contact['company']})")
        else:
            print(f"  [WARN] No HubSpot contact for {email}")

    associations = [
        PublicAssociationsForObject(
            to=PublicObjectId(id=c["id"]),
            types=[
                AssociationSpec(
                    association_category="HUBSPOT_DEFINED",
                    association_type_id=194,
                )
            ],
        )
        for c in matched_contacts
    ]

    call = hs.crm.objects.calls.basic_api.create(
        simple_public_object_input_for_create=CallInput(
            properties={
                "hs_timestamp": str(int(time.time() * 1000)),
                "hs_call_title": doc_name,
                "hs_call_body": notes_text,
                "hs_call_direction": "INBOUND",
                "hs_call_status": "COMPLETED",
            },
            associations=associations if associations else None,
        )
    )
    return call.id, matched_contacts


# ── Gmail helpers ─────────────────────────────────────────────────────────────

def create_gmail_draft(
    gmail_service,
    to_emails: list[str],
    subject: str,
    body: str,
) -> str | None:
    """Create a draft in Gmail. Returns the draft ID."""
    message = MIMEText(body)
    message["subject"] = subject
    if to_emails:
        message["to"] = ", ".join(to_emails)

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    draft = gmail_service.users().drafts().create(
        userId="me",
        body={"message": {"raw": raw}},
    ).execute()
    return draft.get("id")


# ── Local JSON persistence ────────────────────────────────────────────────────

def load_existing_notes() -> list[dict]:
    if OUTPUT_JSON.exists():
        return json.loads(OUTPUT_JSON.read_text())
    return []


def save_notes(notes: list[dict]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(notes, indent=2))


def get_already_processed() -> set[str]:
    """Return the set of source_doc names already in the JSON."""
    return {n["source_doc"] for n in load_existing_notes() if "source_doc" in n}


# ── Core processing (importable by dashboard) ────────────────────────────────

def process_doc(
    doc: dict,
    drive,
    calendar_service,
    gmail_service,
    claude_client: anthropic.Anthropic,
    hubspot_token: str,
    model: str = "claude-sonnet-4-6",
) -> dict | None:
    """Process a single Google Doc through the full pipeline.

    Returns the meeting notes dict for the JSON, or None on failure.
    """
    try:
        print(f"Processing: {doc['name']}")
        notes_text = export_doc_text(drive, doc["id"])
        print(f"  {len(notes_text)} characters extracted.")

        # Calendar lookup
        meeting_dt = _parse_datetime_from_doc_name(doc["name"])
        if not meeting_dt:
            meeting_dt = datetime.fromisoformat(
                doc["modifiedTime"].replace("Z", "+00:00")
            ).replace(tzinfo=None)

        event = find_calendar_event(calendar_service, meeting_dt)
        if event:
            attendees = get_attendee_emails(event)
            attendee_emails = [a["email"] for a in attendees]
            attendee_names = [a["name"] for a in attendees]
        else:
            attendee_emails, attendee_names = [], []

        meeting_date = meeting_dt.strftime("%Y-%m-%d")

        # HubSpot call
        portal_id = _get_portal_id(hubspot_token)
        call_id, matched_contacts = create_hubspot_call(
            hubspot_token, notes_text, doc["name"], attendee_emails,
        )

        company = matched_contacts[0].get("company", "") if matched_contacts else ""
        participant_display = ", ".join(attendee_names) if attendee_names else "Unknown"

        # Follow-up email via Claude
        prompt = build_follow_up_prompt(
            participant=participant_display,
            company=company,
            notes_text=notes_text,
        )
        email_data = _call_claude(claude_client, FOLLOW_UP_SYSTEM, prompt, model)

        draft_id = create_gmail_draft(
            gmail_service,
            to_emails=attendee_emails,
            subject=email_data.get("subject", f"Follow-up: Meeting with {participant_display}"),
            body=email_data.get("body", ""),
        )

        summary_snippet = notes_text.strip().replace("\n", " ")[:200]

        return {
            "date": meeting_date,
            "participant": participant_display,
            "participant_emails": attendee_emails,
            "company": company,
            "summary": summary_snippet,
            "follow_up_done": False,
            "hubspot_call_id": call_id,
            "hubspot_portal_id": portal_id,
            "gmail_draft_id": draft_id,
            "source_doc": doc["name"],
        }
    except Exception as exc:
        print(f"  [ERROR] Failed to process {doc['name']}: {exc}")
        return None


# ── Main pipeline (CLI) ──────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    hubspot_token = os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip()
    if not hubspot_token:
        print("[ERROR] HUBSPOT_ACCESS_TOKEN not set in .env")
        sys.exit(1)

    claude = anthropic.Anthropic(api_key=api_key)
    model = args.model

    print("Connecting to Google Drive...")
    drive = get_drive_service()

    if args.list:
        docs = list_recent_docs(drive)
        if not docs:
            print("No Google Docs found in your Drive.")
            return
        print("\nRecent Google Docs:\n")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc['name']}  ({doc['modifiedTime'][:10]})")
        print(f"\nRe-run with:  python3 meeting_notes.py --file-name \"<name>\"")
        return

    if args.file_name:
        doc = find_doc_by_name(drive, args.file_name)
    else:
        docs = list_recent_docs(drive, max_results=1)
        doc = docs[0] if docs else None

    if not doc:
        print("[ERROR] No matching Google Doc found.")
        sys.exit(1)

    calendar = get_calendar_service()
    gmail = get_gmail_service()

    result = process_doc(doc, drive, calendar, gmail, claude, hubspot_token, model)
    if result:
        existing = load_existing_notes()
        existing.append(result)
        save_notes(existing)
        print(f"Saved to {OUTPUT_JSON}")
        print("\nDone!")
    else:
        print("\n[ERROR] Processing failed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Gemini meeting notes from Google Drive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file-name",
        default=None,
        help="Search for a Google Doc containing this name (uses most recent if omitted)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent Google Docs and exit",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model to use for follow-up email generation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
