"""
Google OAuth 2.0 authentication for Drive, Gmail, and Calendar.

First run opens a browser for consent and caches a token.json file.
Subsequent runs reuse / auto-refresh the cached token.

Usage
-----
    from google_auth import get_drive_service, get_gmail_service, get_calendar_service

    drive    = get_drive_service()      # googleapiclient.discovery.Resource
    gmail    = get_gmail_service()      # googleapiclient.discovery.Resource
    calendar = get_calendar_service()   # googleapiclient.discovery.Resource
"""

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.readonly",
]

_PROJECT_ROOT = Path(__file__).resolve().parent
CREDENTIALS_FILE = _PROJECT_ROOT / "credentials.json"
TOKEN_FILE = _PROJECT_ROOT / "token.json"


def _get_credentials() -> Credentials:
    """Load cached credentials or run the OAuth consent flow."""
    creds: Credentials | None = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json())
            return creds
        except Exception:
            pass

    # Scopes changed or no valid token — run full consent flow
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CREDENTIALS_FILE}. Download your OAuth client JSON from "
            "Google Cloud Console and save it as credentials.json in the project root."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json())
    return creds


def get_drive_service():
    """Return an authenticated Google Drive API v3 service."""
    return build("drive", "v3", credentials=_get_credentials())


def get_gmail_service():
    """Return an authenticated Gmail API v1 service."""
    return build("gmail", "v1", credentials=_get_credentials())


def get_calendar_service():
    """Return an authenticated Google Calendar API v3 service."""
    return build("calendar", "v3", credentials=_get_credentials())
