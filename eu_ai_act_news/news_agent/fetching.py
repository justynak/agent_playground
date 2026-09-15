"""Article fetching: allowlist enforcement, SSRF-safe redirects, HTML text extraction."""

import logging
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from opentelemetry import trace

from .config import ALLOWED_DOMAINS, MAX_ARTICLE_CHARS, MAX_REDIRECTS

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Class names commonly used for the main body of an article, tried in order.
CONTENT_CLASS_HINTS = ["article-body", "article__body", "story-body", "post-content", "entry-content"]


def is_allowed_domain(url: str) -> bool:
    domain = urlparse(url).netloc.removeprefix("www.")
    return any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS)


def fetch_following_safe_redirects(url: str, timeout: int = 15) -> requests.Response:
    """GET `url`, following redirects manually and re-validating the domain on every hop.

    `requests` follows redirects by default and would silently land on any host an
    allowlisted server points at (SSRF). Here we disable auto-redirects and walk each
    hop ourselves, refusing any `Location` that leaves the allowlist.
    """
    headers = {"User-Agent": USER_AGENT}
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not is_allowed_domain(current):
            domain = urlparse(current).netloc.removeprefix("www.")
            raise ValueError(f"Redirect left the allowlist: {domain}")
        resp = requests.get(current, timeout=timeout, headers=headers, allow_redirects=False)
        if resp.is_redirect:
            location = resp.headers.get("Location")
            if not location:
                # A redirect with no Location cannot be followed; return it rather than loop.
                return resp
            current = urljoin(current, location)
            continue
        return resp
    raise ValueError(f"Too many redirects (>{MAX_REDIRECTS}) starting from {url}")


def fetch_article_content(url: str) -> dict[str, Any]:
    with tracer.start_as_current_span("fetching.fetch_article_content") as span:
        span.set_attribute("article.url", url)

        if not is_allowed_domain(url):
            domain = urlparse(url).netloc.removeprefix("www.")
            error = f"Domain not in allowlist: {domain}"
            span.set_attribute("article.error", error)
            span.set_status(trace.StatusCode.ERROR, error)
            logger.warning("[fetch] %s: %s", url, error)
            return {"error": error, "url": url}

        try:
            resp = fetch_following_safe_redirects(url)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            main_content = (
                soup.find("article")
                or soup.find("main")
                or soup.find(attrs={"class": lambda c: c and any(
                    x in " ".join(c) for x in CONTENT_CLASS_HINTS
                )})
                or soup.body
            )

            text = (main_content or soup).get_text(separator=" ", strip=True)
            text = " ".join(text.split())
            truncated = len(text) > MAX_ARTICLE_CHARS
            span.set_attribute("article.content_chars", min(len(text), MAX_ARTICLE_CHARS))
            span.set_attribute("article.truncated", truncated)
            return {
                "url": url,
                "content": text[:MAX_ARTICLE_CHARS],
                "truncated": truncated,
            }
        except Exception as e:
            span.set_attribute("article.error", str(e))
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.warning("[fetch] %s: failed: %s", url, e)
            return {"error": str(e), "url": url}
