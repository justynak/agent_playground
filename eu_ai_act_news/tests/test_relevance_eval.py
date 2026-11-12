"""Labelled eval set for the EU AI Act relevance gate, run against the live model.

Each case is a short synthetic article plus the expected gate decision (keep/drop).
The hard negatives are the ones that slip past the keyword prefilter: they all mention
"European AI", "AI regulation" or even the AI Act itself, but are not about the Act.

When the gate gets a real article wrong in production, add it here (title + the relevant
paragraphs) so the fix can be measured and the regression stays caught.

Run with: make integration_tests
"""

import pytest

from news_agent.collection import is_relevant
from news_agent.config import ACCEPTED_RELEVANCE
from news_agent.summarization import assess_relevance

KEEP, DROP = True, False

CASES = [
    # --- should be kept -------------------------------------------------------------
    (
        "gpai_code_of_practice",
        KEEP,
        "Commission publishes final General-Purpose AI Code of Practice",
        (
            "The European Commission has published the final version of the General-Purpose AI Code of "
            "Practice, a voluntary tool to help providers of models such as GPT and Gemini comply with "
            "the EU AI Act's transparency, copyright and safety obligations. Signatories will benefit "
            "from reduced administrative burden, the AI Office said, while providers that do not sign "
            "must demonstrate compliance by other means."
        ),
    ),
    (
        "omnibus_delay",
        KEEP,
        "EU proposes delaying high-risk AI rules under digital omnibus",
        (
            "The Commission's digital omnibus package proposes pushing back the application date of the "
            "AI Act's high-risk obligations by up to 16 months, tying it to the availability of "
            "harmonised standards. Civil-society groups warned the move weakens the regulation, while "
            "industry associations welcomed the extra time to prepare conformity assessments."
        ),
    ),
    (
        "company_compliance",
        KEEP,
        "Banks race to classify credit-scoring models ahead of AI Act deadline",
        (
            "European lenders are auditing their credit-scoring systems, which the EU AI Act classifies "
            "as high-risk. Compliance teams say the hardest part is the Act's requirements on data "
            "governance and human oversight, and several banks have hired dedicated AI Act officers. "
            "National supervisors are expected to coordinate enforcement with the AI Office."
        ),
    ),
    # --- should be dropped (hard negatives) -----------------------------------------
    (
        "ai_stock_selloff",
        DROP,
        "European AI stocks tumble as chipmaker selloff spreads",
        (
            "Shares in European AI companies fell sharply on Tuesday as a selloff in US chipmakers "
            "spread to Europe. ASML dropped 6% and SAP lost 4%, as investors questioned whether heavy "
            "spending on AI infrastructure will pay off. Analysts said valuations had run ahead of "
            "earnings and expected volatility to continue into the next quarter."
        ),
    ),
    (
        # Real regression: published in the 2026-11-07 digest (issue #46). It passed the keyword
        # prefilter on "AI regulation". Text is the pipeline's own verified summary of
        # https://www.bbc.com/news/articles/crlykkzwwpydo, as the BBC page can't be fetched here.
        "bbc_us_midterms_ai_selloff",
        DROP,
        "Why the US midterms sparked an AI stock selloff",
        (
            "A steep sell-off hit US stock markets, most dramatically on the Nasdaq, after sweeping "
            "Democratic gains in the midterm elections gave the party full control of Congress. Several "
            "leading Democrats campaigned against unhindered expansion of data centres, and fears this will "
            "lead to more AI regulation appear to have concerned investors, though worries about the "
            "industry's prospects had been growing for some time. Democrats are now expected to challenge "
            "the president's technology agenda and introduce legislation that could curb the industry's "
            "expansion, with Texas governor-elect Gina Hinojosa pledging a moratorium on new data centre "
            "construction. Analysts quoted say traders are likely to be more selective rather than expect an "
            "immediate widespread recovery, while some argue valuations had been inflated by expectations of "
            "\"almost infinite AI growth.\""
        ),
    ),
    (
        "funding_round_name_drop",
        DROP,
        "French AI startup raises €600m to build European AI champion",
        (
            "Paris-based model developer announced a €600m funding round led by US venture firms, "
            "valuing the company at €6bn. The money will fund new data centres and hiring. In a "
            "single line, the CEO said the company was 'comfortable with the AI Act', before focusing "
            "on plans to launch a new multilingual model next spring."
        ),
    ),
    (
        "us_state_law",
        DROP,
        "California passes landmark AI regulation bill",
        (
            "California lawmakers passed a bill requiring developers of large AI models to publish "
            "safety frameworks and report serious incidents to the state. The governor has 30 days to "
            "sign it. Supporters said the law fills a gap left by the lack of federal AI regulation in "
            "the United States."
        ),
    ),
    (
        "gdpr_fine",
        DROP,
        "Irish regulator fines social network €250m over EU users' data transfers",
        (
            "Ireland's Data Protection Commission fined the company €250m under the GDPR for "
            "transferring European users' personal data to the US without adequate safeguards. The "
            "company said it would appeal. The decision follows a long-running investigation into its "
            "advertising business. A spokesperson added that the regulator keeps the EU AI Office informed "
            "of its work in general."
        ),
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize(("case_id", "expected_keep", "title", "text"), CASES, ids=[c[0] for c in CASES])
def test_relevance_gate(case_id, expected_keep, title, text):
    result = assess_relevance({"title": title, "url": f"https://example.com/{case_id}"}, text)
    assert result is not None, "relevance call failed"
    kept = result["relevance"] in ACCEPTED_RELEVANCE
    assert kept is expected_keep, f"graded {result['relevance']!r}: {result['reasoning']}"


def test_hard_negatives_actually_pass_the_keyword_prefilter():
    """Guards the eval set itself: a hard negative the prefilter already drops tests nothing."""
    for case_id, expected_keep, title, text in CASES:
        if expected_keep is DROP:
            assert is_relevant(title, text), f"{case_id} would never reach the relevance gate"
