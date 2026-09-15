"""LLM summarization and independent verification of per-article summaries."""

import json
import logging
import time
from typing import Any

from openai import OpenAI
from opentelemetry import trace

from .config import (
    CATEGORIES,
    DEEPSEEK_API_KEY,
    MAX_ARTICLE_CHARS,
    MAX_SUMMARY_ATTEMPTS,
    SUMMARIZE_TOOL,
    VERIFY_TOOL,
)
from .fetching import fetch_article_content
from .telemetry import (
    articles_processed_total,
    llm_call_duration_seconds,
    verification_attempts_total,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

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

    with tracer.start_as_current_span("summarization.generate") as span:
        span.set_attribute("llm.model", "deepseek-chat")
        span.set_attribute("llm.retry", feedback is not None)
        start = time.monotonic()
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=[SUMMARIZE_TOOL],
                tool_choice={"type": "function", "function": {"name": "record_summary"}},
            )
            payload = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.warning("[summarize] generation call failed: %s", e)
            return None
        finally:
            llm_call_duration_seconds.record(time.monotonic() - start, {"call_type": "generate"})

        result = validate_summary_payload(payload)
        span.set_attribute("llm.valid_payload", result is not None)
        if result is not None:
            span.set_attribute("summary.category", result["category"])
        return result


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

    with tracer.start_as_current_span("summarization.verify") as span:
        span.set_attribute("llm.model", "deepseek-chat")
        start = time.monotonic()
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=[VERIFY_TOOL],
                tool_choice={"type": "function", "function": {"name": "record_verification"}},
            )
            payload = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.warning("[verify] verification call failed: %s", e)
            return None
        finally:
            llm_call_duration_seconds.record(time.monotonic() - start, {"call_type": "verify"})

        result = validate_verification_payload(payload)
        if result is not None:
            span.set_attribute("verification.faithful", result["faithful"])
            span.set_attribute("verification.unsupported_claims_count", len(result["unsupported_claims"]))
        return result


def summarize_article(article: dict, max_attempts: int = MAX_SUMMARY_ATTEMPTS) -> dict[str, str] | None:
    """Summarize a single article, verifying the summary against the article text before accepting it.

    Regenerates the summary (with feedback on what was unsupported) up to `max_attempts`
    times. If no attempt passes verification, the article is rejected and None is returned
    — callers are expected to skip such articles rather than publish an unverified summary.
    """
    with tracer.start_as_current_span("summarization.summarize_article") as span:
        span.set_attribute("article.url", article["url"])
        span.set_attribute("article.title", article["title"])

        content_result = fetch_article_content(article["url"])
        if "error" in content_result:
            span.set_attribute("summarize.outcome", "rejected_fetch_error")
            span.add_event("fetch_failed", {"error": content_result["error"]})
            logger.info("[summarize] skipping %s: %s", article["url"], content_result["error"])
            articles_processed_total.add(1, {"outcome": "rejected_fetch_error"})
            return None
        content = content_result["content"]
        truncated = content_result.get("truncated", False)
        span.set_attribute("article.truncated", truncated)
        if truncated:
            logger.info("[summarize] %s: article truncated at %d chars", article["url"], MAX_ARTICLE_CHARS)

        feedback: list[str] | None = None
        for attempt in range(1, max_attempts + 1):
            generated = generate_summary_payload(article, content, feedback, truncated=truncated)
            if generated is None:
                span.add_event("attempt", {"attempt": attempt, "result": "generation_failed"})
                logger.info("[summarize] %s attempt %d/%d: generation failed", article["url"], attempt, max_attempts)
                verification_attempts_total.add(1, {"result": "generation_failed"})
                feedback = None
                continue

            verification = verify_summary(content, generated["summary"], truncated=truncated)
            if verification is not None and verification["faithful"]:
                span.add_event("attempt", {"attempt": attempt, "result": "faithful"})
                span.set_attribute("summarize.outcome", "accepted")
                span.set_attribute("summarize.attempts_used", attempt)
                span.set_attribute("summary.category", generated["category"])
                verification_attempts_total.add(1, {"result": "faithful"})
                articles_processed_total.add(1, {"outcome": "accepted"})
                return {"title": article["title"], "url": article["url"], **generated}

            unsupported = verification["unsupported_claims"] if verification else []
            result_label = "unfaithful" if verification else "verify_failed"
            reason = unsupported or "verification call failed"
            span.add_event("attempt", {
                "attempt": attempt,
                "result": result_label,
                "unsupported_claims": unsupported,
            })
            logger.info(
                "[summarize] %s attempt %d/%d failed verification: %s", article["url"], attempt, max_attempts, reason
            )
            verification_attempts_total.add(1, {"result": result_label})
            feedback = unsupported

        span.set_attribute("summarize.outcome", "rejected_verification")
        span.set_attribute("summarize.attempts_used", max_attempts)
        logger.info("[summarize] rejected %s after %d failed attempt(s)", article["url"], max_attempts)
        articles_processed_total.add(1, {"outcome": "rejected_verification"})
        return None


def summarize_articles(articles: list[dict]) -> list[dict]:
    summaries = []
    for article in articles:
        summary = summarize_article(article)
        if summary:
            summaries.append(summary)
    return summaries
