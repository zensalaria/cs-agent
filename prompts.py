"""Prompt templates for ticket classification."""

from __future__ import annotations


SYSTEM_PROMPT = """You are a customer success analyst classifying B2B SaaS support tickets.
You will be given a support conversation transcript and must evaluate it on three dimensions.
Return ONLY valid JSON — no markdown, no explanation, no extra text."""


def build_analysis_prompt(
    transcript: str,
    speed_percentile: float | None,
    products: list[dict],
) -> str:
    """
    Build the user-turn prompt for a single ticket.

    Args:
        transcript:       The raw conversation text from the ticket.
        speed_percentile: Pre-computed percentile (0–100). Lower = faster = better.
                          None if TTC data was unavailable.
        products:         List of dicts with keys 'product_name' and 'category'.
    """
    # ── Speed context block ───────────────────────────────────────────────────
    if speed_percentile is not None:
        speed_block = (
            f"\nThis ticket's speed percentile score is {speed_percentile:.1f}. "
            "This means it was resolved faster than or equal to "
            f"{speed_percentile:.1f}% of all tickets. "
            "Lower = faster = better for the customer. "
            "A score below 33 is fast, 33–66 is average, above 66 is slow."
        )
    else:
        speed_block = (
            "\nTime-to-close data was not available for this ticket."
        )

    # ── Products context block ────────────────────────────────────────────────
    if products:
        product_lines = "\n".join(
            f"  - {p['product_name']} ({p['category']})" for p in products
        )
        products_block = f"\nCore product catalogue:\n{product_lines}"
    else:
        products_block = ""

    return f"""Analyse this support ticket on three dimensions.
{speed_block}
{products_block}

## CHECK 1: Feature Request Detection
Does this interaction contain a feature request from the customer (not the support agent)?
A feature request is when a customer asks for new functionality, improvements, or changes to
the product. Bug reports that implicitly suggest a fix also count if the customer explicitly
asks for something to be added or changed.

## CHECK 2: Quality of Product Offering
Rate what this interaction reveals about the quality of the company's core product offering.
Only score this if the ticket is related to one of the products listed in the core product
catalogue above. Use "Null" for general questions, onboarding issues, billing queries, or
anything that does not reflect the product's intrinsic quality.

Rate as:
- "High"   = interaction suggests the product is working well (customer satisfied, no issues)
- "Medium" = mixed signals or minor issues that were resolved
- "Low"    = interaction reveals a significant product problem (bug, poor UX, missing feature
              causing frustration)
- "Null"   = not relevant to core product quality

## CHECK 3: Client Experience Quality Score
Rate the overall quality of the customer's experience in this interaction. Consider:
- Customer sentiment — did the customer seem satisfied, frustrated, or neutral?
- Resolution quality — was the issue actually resolved or left open?
- Response completeness — did the agent answer all questions raised?
- Agent professionalism — was the tone helpful and professional?
- Speed — use the speed percentile score provided above as an input (lower percentile = faster
  = positive contribution to experience)

Also determine the customer's sentiment on its own:
- Focus only on the customer's messages, not the agent's
- Consider the customer's tone, wording, and implied satisfaction
- Consider whether the outcome was resolved

Return this exact JSON structure:
{{
  "feature_request_flag": true or false,
  "feature_request_description": "concise description of the request, or null if none",
  "product_quality_score": "High" or "Medium" or "Low" or "Null",
  "product_quality_reason": "concise reason for the score",
  "sentiment_score": "positive" or "neutral" or "negative",
  "client_experience_quality_score": "High" or "Medium" or "Low",
  "client_experience_quality_reason": "concise reason, referencing speed and sentiment"
}}

TRANSCRIPT:
{transcript}"""
