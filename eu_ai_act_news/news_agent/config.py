"""Static configuration: environment, feed sources, keywords, model schemas, tunables."""

import os
import re

from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]

RSS_FEEDS = [
    {"name": "Euractiv", "url": "https://www.euractiv.com/feed/"},
    {"name": "European Parliament", "url": "https://www.europarl.europa.eu/rss/doc/latest-news/en.xml"},
    {"name": "Politico Europe", "url": "https://www.politico.eu/feed/"},
    {"name": "Reuters Technology", "url": "https://feeds.reuters.com/reuters/technologyNews"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
    {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
    {"name": "Wired", "url": "https://www.wired.com/feed/rss"},
    {"name": "BBC Technology", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml"},
    {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/"},
]

ALLOWED_DOMAINS = {
    "euractiv.com",
    "europarl.europa.eu",
    "politico.eu",
    "reuters.com",
    "techcrunch.com",
    "theverge.com",
    "wired.com",
    "bbc.co.uk",
    "bbc.com",
    "venturebeat.com",
    "ec.europa.eu",
    "consilium.europa.eu",
    "eur-lex.europa.eu",
}

AI_ACT_KEYWORDS = [
    "eu ai act", "ai act", "artificial intelligence act",
    "ai regulation", "eu ai regulation", "general purpose ai",
    "gpai", "ai liability", "ai office", "eu ai office",
    "eu ai", "europe ai", "european ai",
]

# Word-boundary matching. Joining a keyword's words with `\s*[- ]\s*` tolerates
# "ai act", "ai-act" and "AI Act" variants, while the surrounding \b stops partial
# substring hits like "Thai activist" (contains "ai act") or "AI action plan".
KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        r"\s*[- ]\s*".join(re.escape(word) for word in keyword.split())
        for keyword in AI_ACT_KEYWORDS
    ) + r")\b",
    re.IGNORECASE,
)

CATEGORIES = [
    "Key Developments",
    "Legislative & Regulatory Updates",
    "Industry & Compliance",
    "Research & Expert Opinion",
]

SUMMARIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "record_summary",
        "description": "Records a structured summary for a single article.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The single best-fitting category for this article.",
                    "enum": CATEGORIES,
                },
                "summary": {
                    "type": "string",
                    "description": "2-4 sentence summary of the article, grounded only in the provided article text.",
                },
            },
            "required": ["category", "summary"],
        },
    },
}

VERIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "record_verification",
        "description": "Records whether a summary is fully supported by the article text it claims to summarize.",
        "parameters": {
            "type": "object",
            "properties": {
                "faithful": {
                    "type": "boolean",
                    "description": "True only if every factual claim in the summary is directly supported by the article text.",
                },
                "unsupported_claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Claims in the summary not supported by the article text. Empty if faithful is true.",
                },
            },
            "required": ["faithful", "unsupported_claims"],
        },
    },
}

# Graded rather than boolean: a yes/no gate pushes borderline "mentions the AI Act once"
# articles to whichever side the model leans that day, while an explicit "passing" grade
# gives those a named bucket to land in. Only grades in ACCEPTED_RELEVANCE are published.
RELEVANCE_LEVELS = ["primary", "substantive", "passing", "none"]
ACCEPTED_RELEVANCE = {"primary", "substantive"}

RELEVANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "record_relevance",
        "description": "Records how relevant one news article is to the EU AI Act specifically.",
        "parameters": {
            "type": "object",
            # `reasoning` is listed first so the model writes its justification before
            # committing to a grade, rather than rationalising a grade it already picked.
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "1-2 sentences: what the article is mainly about, and how (if at all) "
                        "it involves the EU AI Act."
                    ),
                },
                "relevance": {
                    "type": "string",
                    "enum": RELEVANCE_LEVELS,
                    "description": (
                        "primary: the EU AI Act is the main subject. "
                        "substantive: a significant part of the article discusses the EU AI Act. "
                        "passing: the EU AI Act is only name-dropped or mentioned in a sentence or two. "
                        "none: the article does not involve the EU AI Act."
                    ),
                },
            },
            "required": ["reasoning", "relevance"],
        },
    },
}

MAX_SUMMARY_ATTEMPTS = 3

# Cap on how much article text we feed the model. Raised from the original 4000 so
# most articles fit whole; when it still truncates, summarize_article refuses to
# summarize the cut-off article rather than verifying against partial evidence.
MAX_ARTICLE_CHARS = 12000

MAX_REDIRECTS = 5

ISSUES_PER_PAGE = 100
MAX_ISSUE_PAGES = 10
