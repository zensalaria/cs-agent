"""
Prompts for the meeting notes pipeline.

Claude is used only for generating the follow-up email draft.
Meeting data (participants, company) comes from Google Calendar + HubSpot.
"""

FOLLOW_UP_SYSTEM = """\
You are a customer success manager drafting a professional follow-up email \
after a meeting. Keep the tone warm but the messaging should be as concise and clear as possible. The email should:
- Thank the participant for their time
- List any action items that were agreed upon


Respond with valid JSON and nothing else — no markdown fences."""


def build_follow_up_prompt(
    participant: str,
    company: str,
    notes_text: str,
) -> str:
    return f"""\
Draft a follow-up email for the meeting described below.
Use the meeting notes to identify key discussion points and action items.

Participant(s): {participant}
Company: {company}

--- MEETING NOTES ---
{notes_text[:6000]}
--- END ---

Return a single JSON object with these keys:
- "subject": the email subject line
- "body": the full email body (plain text, use \\n for line breaks)"""
