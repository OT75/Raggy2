"""
Labeled routing eval for Raggy's mode router — generalized across multiple
documents.

The whole generalization mechanism is one dict: DOCUMENTS maps a PDF
filename to its own labeled questions. Every test function loops over that
dict. To test a new document, drop its PDF in tests/fixtures/ and add one
entry to DOCUMENTS below — no other code changes.

This still makes real Groq API calls and needs GROQ_API_KEY set. Tests
auto-skip cleanly if it's absent, so CI stays green either way.

Run locally:
    cd backend
    pytest tests/test_routing.py -v -s
"""
import os
import sys
import functools
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_engine import get_pdf_text, get_text_chunks, get_vectorstore, answer_question

load_dotenv(override=True)

API_CHOICE = "Groq (Free)"
API_KEY = os.getenv("GROQ_API_KEY")
FIXTURES_DIR = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="Set GROQ_API_KEY to run the routing eval — it makes real LLM calls.",
)


# ============================================================================
# THE WHOLE GENERALIZATION MECHANISM: one dict.
# filename -> {"rag_cases": [...], "fallback_cases": [...]}
#
# rag_cases:      questions this document DOES answer, with a fragment that
#                 must appear in the retrieved source chunks.
# fallback_cases: plausible-sounding questions this document does NOT cover.
#
# An optional "xfail" key on any case marks it as a known, accepted miss
# (see Meridian's two entries below) rather than hiding or deleting it.
# ============================================================================
DOCUMENTS = {
    "meridian-internship-handbook.pdf": {
        "rag_cases": [
            {"q": "What is the monthly stipend for the internship?", "expect": "2,450"},
            {"q": "Who is the Track Lead for Perception Systems?", "expect": "Foss"},
            {"q": "How many interns are on the Applied Robotics Control team?", "expect": "4 interns"},
            {"q": "What do Generative Tooling interns receive instead of a sensor kit?", "expect": "GPU"},
            {"q": "When is the mid-program review?", "expect": "May 18"},
            {"q": "When is the final presentation?", "expect": "July 28"},
            {
                "q": "What is the one-time relocation allowance?", "expect": "900",
                "xfail": "Scores 1.527, just over the 1.5 threshold. Raising the threshold "
                         "to catch this would also let the 'remote work' fallback case "
                         "(1.147) through as a false grounded answer -- the worse failure.",
            },
            {"q": "How many hours per week do interns work?", "expect": "37.5"},
            {"q": "Who scores the final presentations besides the Track Leads?", "expect": "external reviewer"},
            {"q": "Who should I contact with program coordination questions?", "expect": "internships@meridianrobotics"},
        ],
        "fallback_cases": [
            {"q": "What is the WiFi password?"},
            {"q": "Is there a gym on campus?"},
            {"q": "What is the dress code?"},
            {
                "q": "Can interns work fully remotely?",
                "xfail": "Scores 1.147, lower than several genuinely answerable questions "
                         "in this document. Accepted, documented miss -- see rag_engine.py.",
            },
            {"q": "What is the visitor parking policy?"},
        ],
    },

    "aurora-cloud-storage-policy.pdf": {
        "rag_cases": [
            {"q": "What is the monthly price of the Professional plan?", "expect": "12.99"},
            {"q": "How much storage does the Enterprise plan include?", "expect": "10 TB"},
            {"q": "How many devices can a Starter plan use?", "expect": "2"},
            {"q": "What is the support response time for Enterprise customers?", "expect": "1 business hour"},
            {"q": "How long is the refund eligibility window?", "expect": "14"},
            {"q": "How much data can be uploaded and still qualify for a refund?", "expect": "10 GB"},
            {"q": "How long is data kept after cancellation before deletion?", "expect": "30 days"},
            {"q": "How long does full deletion take after that 30-day window?", "expect": "72 hours"},
            {"q": "How long does an account transfer take?", "expect": "10 business days"},
            {
                "q": "Where should security vulnerabilities be reported?", "expect": "security@aurorastorage.example",
                "xfail": "Scores 1.884, worse than every genuinely irrelevant Aurora fallback question "
                         "(1.69-1.75). Likely cause: this sentence sits inside a 'Contact' section that's "
                         "otherwise entirely about general billing/sales, so the embedded chunk's vector is "
                         "diluted by surrounding off-topic contact info. A second, independently-found "
                         "instance of the same category of limit found in the Meridian eval -- evidence this "
                         "is a real property of the approach, not a one-document fluke.",
            },
        ],
        "fallback_cases": [
            {"q": "Do you offer a student discount?"},
            {"q": "Is there a mobile app available?"},
            {"q": "Can I pay with cryptocurrency?"},
            {"q": "Do you support single sign-on (SSO) login?"},
            {"q": "Is there a free trial available?"},
        ],
    },
}

# Doc-independent — general mode is a property of having NO vectorstore at
# all, not of any document's content, so these aren't tied to DOCUMENTS.
GENERAL_CASES = [
    "Hi, how are you?",
    "What's a good icebreaker question for a new team?",
    "Can you explain what a REST API is?",
    "What's the weather usually like in the fall?",
    "Recommend a book about machine learning.",
]


@functools.lru_cache(maxsize=None)
def _vectorstore_for(filename):
    """Builds (and caches) one vectorstore per document, so the 10+ questions
    against the same file don't each re-embed it from scratch."""
    with open(FIXTURES_DIR / filename, "rb") as f:
        raw_text = get_pdf_text([f])
    chunks = get_text_chunks(raw_text)
    return get_vectorstore(chunks, API_CHOICE, API_KEY)


def _param(filename, case, *keys):
    """Builds one pytest.param from a case dict, attaching an xfail mark if
    the case declares one. `keys` picks which dict fields become argvalues."""
    marks = [pytest.mark.xfail(reason=case["xfail"], strict=True)] if "xfail" in case else []
    values = tuple(case[k] for k in keys)
    return pytest.param(filename, *values, marks=marks, id=f"{filename}::{case['q'][:40]}")


RAG_PARAMS = [
    _param(filename, case, "q", "expect")
    for filename, spec in DOCUMENTS.items()
    for case in spec["rag_cases"]
]

FALLBACK_PARAMS = [
    _param(filename, case, "q")
    for filename, spec in DOCUMENTS.items()
    for case in spec["fallback_cases"]
]


@pytest.mark.parametrize("filename,question,expected_fragment", RAG_PARAMS)
def test_rag_mode_and_grounding(filename, question, expected_fragment):
    vectorstore = _vectorstore_for(filename)
    answer, sources, mode, debug = answer_question(vectorstore, question, API_CHOICE, API_KEY)
    print(f"\n[RAG]      {filename}  score={debug['best_score']}  mode={mode}  Q: {question}")

    assert mode == "rag", f"expected 'rag', got '{mode}' — score={debug['best_score']} — Q: {question}"

    combined_sources = " ".join(doc.page_content for doc in sources)
    assert expected_fragment.lower() in combined_sources.lower(), (
        f"retrieval didn't surface a chunk containing '{expected_fragment}' for: {question}"
    )


@pytest.mark.parametrize("filename,question", FALLBACK_PARAMS)
def test_fallback_mode(filename, question):
    vectorstore = _vectorstore_for(filename)
    _, _, mode, debug = answer_question(vectorstore, question, API_CHOICE, API_KEY)
    print(f"\n[FALLBACK] {filename}  score={debug['best_score']}  mode={mode}  Q: {question}")

    assert mode == "general_fallback", (
        f"expected 'general_fallback', got '{mode}' — score={debug['best_score']} — Q: {question}"
    )


@pytest.mark.parametrize("question", GENERAL_CASES)
def test_general_mode_with_no_documents(question):
    _, _, mode, _ = answer_question(None, question, API_CHOICE, API_KEY)
    assert mode == "general", f"expected 'general', got '{mode}' for: {question}"