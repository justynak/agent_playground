"""LLM summarization and independent verification of per-article summaries."""

import json
import sys
from typing import Any

from openai import OpenAI

from .config import (
    CATEGORIES,
    DEEPSEEK_API_KEY,
    MAX_ARTICLE_CHARS,
    MAX_SUMMARY_ATTEMPTS,
    SUMMARIZE_TOOL,
    VERIFY_TOOL,
)
from .fetching import fetch_article_content

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


def validate_summary_payload(payload: dict) -> dict[str, str] | None:
    category = payload.get("category")
    summary = payload.get("summary")
    if not isinstance(category, str) or category not in CATEGORIES:
        return None
    if not isinstance(summary, str) or not summary.strip():
        return None
    return {"category": category, "summary": summary.strip()}


def generate_summary_payload(
    article: dict,
    content: str,
    feedback: list[str] | None = None,
    truncated: bool = False,
) -> dict[str, str] | None:
    """Ask the model for one structured, categorized summary of the given article text.

    If `feedback` (unsupported claims from a prior failed verification) is given, the
    model is asked to correct them rather than starting from scratch.

    If `truncated` is True the article text was cut off, so the model is told to
    summarize only what is present and not infer anything about the missing portion.
    """
    system_prompt = (
        "You are an EU AI Act policy analyst. You will be given the full text of one news article.\n"
        "Summarize it in 2-4 sentences, using only information present in the article text below — "
        "do not add outside knowledge or speculation.\n"
        f"Then call record_summary with your summary and the single best-fitting category from: "
        f"{', '.join(CATEGORIES)}."
    )
    user_content = f"Article title: {article['title']}\n\nArticle text:\n{content}"
    if truncated:
        user_content += (
            "\n\nNOTE: the article text above was cut off at a length limit and the ending is missing. "
            "Summarize only what is present; do not state or imply anything about the parts you cannot see."
        )
    if feedback:
        claims = "\n".join(f"- {claim}" for claim in feedback)
        user_content += (
            "\n\nYour previous summary made claims not supported by the article text above:\n"
            f"{claims}\n"
            "Write a corrected summary that avoids these unsupported claims."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=[SUMMARIZE_TOOL],
            tool_choice={"type": "function", "function": {"name": "record_summary"}},
        )
        payload = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    except Exception as e:
        print(f"  [summarize] generation call failed: {e}", file=sys.stderr)
        return None

    return validate_summary_payload(payload)


def validate_verification_payload(payload: dict) -> dict[str, Any] | None:
    faithful = payload.get("faithful")
    claims = payload.get("unsupported_claims")
    if not isinstance(faithful, bool):
        return None
    if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
        return None
    return {"faithful": faithful, "unsupported_claims": claims}


def verify_summary(content: str, summary: str, truncated: bool = False) -> dict[str, Any] | None:
    """Judge whether `summary` is fully supported by the article text `content`.

    If `truncated` is True the source text was cut off; the judge is told that claims
    about anything beyond the shown text cannot be confirmed either way, so a summary
    should not be approved on the assumption the missing text backs it up.

    Returns None if the judge call itself fails or returns something unparseable —
    callers must treat that as "could not verify" (i.e. not faithful), not as a pass.
    """
    system_prompt = (
        "You are a strict fact-checker. You will be given an article's full text and a summary of it.\n"
        "Check whether every factual claim in the summary — dates, numbers, names, actions, outcomes — "
        "is directly supported by the article text. Assume the summary is wrong until you find clear "
        "support for each claim; do not give it the benefit of the doubt.\n"
        "Call record_verification with your verdict."
    )
    user_content = f"Article text:\n{content}\n\nSummary to check:\n{summary}"
    if truncated:
        user_content += (
            "\n\nNOTE: the article text above was cut off at a length limit. A claim can only be "
            "counted as supported if the shown text backs it up; do not assume the missing remainder supports it."
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=[VERIFY_TOOL],
            tool_choice={"type": "function", "function": {"name": "record_verification"}},
        )
        payload = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    except Exception as e:
        print(f"  [verify] verification call failed: {e}", file=sys.stderr)
        return None

    return validate_verification_payload(payload)


def summarize_article(article: dict, max_attempts: int = MAX_SUMMARY_ATTEMPTS) -> dict[str, str] | None:
    """Summarize a single article, verifying the summary against the article text before accepting it.

    Regenerates the summary (with feedback on what was unsupported) up to `max_attempts`
    times. If no attempt passes verification, the article is rejected and None is returned
    — callers are expected to skip such articles rather than publish an unverified summary.
    """
    content_result = fetch_article_content(article["url"])
    if "error" in content_result:
        print(f"  [summarize] skipping {article['url']}: {content_result['error']}", file=sys.stderr)
        return None
    content = content_result["content"]
    truncated = content_result.get("truncated", False)
    if truncated:
        print(f"  [summarize] {article['url']}: article truncated at {MAX_ARTICLE_CHARS} chars", file=sys.stderr)

    feedback: list[str] | None = None
    for attempt in range(1, max_attempts + 1):
        generated = generate_summary_payload(article, content, feedback, truncated=truncated)
        if generated is None:
            print(f"  [summarize] {article['url']} attempt {attempt}/{max_attempts}: generation failed", file=sys.stderr)
            feedback = None
            continue

        verification = verify_summary(content, generated["summary"], truncated=truncated)
        if verification is not None and verification["faithful"]:
            return {"title": article["title"], "url": article["url"], **generated}

        unsupported = verification["unsupported_claims"] if verification else []
        reason = unsupported or "verification call failed"
        print(
            f"  [summarize] {article['url']} attempt {attempt}/{max_attempts} failed verification: {reason}",
            file=sys.stderr,
        )
        feedback = unsupported

    print(f"  [summarize] rejected {article['url']} after {max_attempts} failed attempt(s)", file=sys.stderr)
    return None


def summarize_articles(articles: list[dict]) -> list[dict]:
    summaries = []
    for article in articles:
        summary = summarize_article(article)
        if summary:
            summaries.append(summary)
    return summaries
