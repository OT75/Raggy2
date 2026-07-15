"""
Labeled routing eval for Raggy's mode router.

Tests two things, per the JD's own language ("separate what's working from
what's hype"):
  1. Does the router pick the correct mode (general / general_fallback / rag)
     for each labeled question?
  2. For answerable questions, does retrieval actually surface the chunk that
     contains the answer — not just "some mode was picked," but "the right
     evidence was found"?

This makes real Groq API calls and needs GROQ_API_KEY set — it's an
integration eval, not a pure unit test, since the router's decision depends
on a live similarity score and a live LLM call. Tests auto-skip if no key is
present, so CI stays green either way rather than failing on a missing secret.

Run locally:
    cd backend
    pytest tests/test_routing.py -v
"""
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_engine import get_pdf_text, get_text_chunks, get_vectorstore, answer_question

load_dotenv(override=True)

API_CHOICE = "Groq (Free)"
API_KEY = os.getenv("GROQ_API_KEY")

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "meridian-internship-handbook.pdf"

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="Set GROQ_API_KEY to run the routing eval — it makes real LLM calls.",
)


@pytest.fixture(scope="module")
def handbook_vectorstore():
    with open(FIXTURE_PDF, "rb") as f:
        raw_text = get_pdf_text([f])
    chunks = get_text_chunks(raw_text)
    return get_vectorstore(chunks, API_CHOICE, API_KEY)


# (question, a fragment that MUST appear in the retrieved source chunks —
# proof retrieval found the right evidence, not just that mode routing agreed)
RAG_CASES = [
    ("What is the monthly stipend for the internship?", "2,450"),
    ("Who is the Track Lead for Perception Systems?", "Foss"),
    ("How many interns are on the Applied Robotics Control team?", "4 interns"),
    ("What do Generative Tooling interns receive instead of a sensor kit?", "GPU"),
    ("When is the mid-program review?", "May 18"),
    ("When is the final presentation?", "July 28"),
    pytest.param(
        "What is the one-time relocation allowance?", "900",
        marks=pytest.mark.xfail(
            reason="Known score overlap: this question scores 1.527, just over the 1.5 "
                   "threshold. Raising the threshold to catch it would also let the "
                   "'remote work' fallback case (1.147) through as false-positive RAG — "
                   "the worse failure mode. Accepted, documented miss, not a regression.",
            strict=True,
        ),
    ),
    ("How many hours per week do interns work?", "37.5"),
    ("Who scores the final presentations besides the Track Leads?", "external reviewer"),
    ("Who should I contact with program coordination questions?", "internships@meridianrobotics"),
]

# Plausible HR-sounding questions the handbook genuinely doesn't cover.
FALLBACK_CASES = [
    "What is the WiFi password?",
    "Is there a gym on campus?",
    "What is the dress code?",
    pytest.param(
        "Can interns work fully remotely?",
        marks=pytest.mark.xfail(
            reason="Known score overlap: this question scores 1.147, lower than several "
                   "genuinely answerable questions. score_threshold=1.5 was set to favor "
                   "not letting irrelevant questions masquerade as grounded, which means "
                   "this specific borderline case is an accepted, documented miss — not a "
                   "regression. See rag_engine.answer_question docstring.",
            strict=True,
        ),
    ),
    "What is the visitor parking policy?",
]

# No documents loaded at all — general mode is a property of the SESSION
# (no vectorstore), not of the question's content.
GENERAL_CASES = [
    "Hi, how are you?",
    "What's a good icebreaker question for a new team?",
    "Can you explain what a REST API is?",
    "What's the weather usually like in the fall?",
    "Recommend a book about machine learning.",
]


@pytest.mark.parametrize("question,expected_fragment", RAG_CASES)
def test_rag_mode_and_grounding(handbook_vectorstore, question, expected_fragment):
    answer, sources, mode, debug = answer_question(
        handbook_vectorstore, question, API_CHOICE, API_KEY
    )
    print(f"\n[RAG case]      score={debug['best_score']:.3f}  mode={mode}  Q: {question}")

    assert mode == "rag", f"expected 'rag', got '{mode}' — score={debug['best_score']} — Q: {question}"

    combined_sources = " ".join(doc.page_content for doc in sources)
    assert expected_fragment.lower() in combined_sources.lower(), (
        f"retrieval didn't surface a chunk containing '{expected_fragment}' for: {question}"
    )


@pytest.mark.parametrize("question", FALLBACK_CASES)
def test_fallback_mode(handbook_vectorstore, question):
    _, _, mode, debug = answer_question(handbook_vectorstore, question, API_CHOICE, API_KEY)
    print(f"\n[FALLBACK case] score={debug['best_score']:.3f}  mode={mode}  Q: {question}")

    assert mode == "general_fallback", (
        f"expected 'general_fallback', got '{mode}' — score={debug['best_score']} — Q: {question}"
    )


@pytest.mark.parametrize("question", GENERAL_CASES)
def test_general_mode_with_no_documents(question):
    _, _, mode, debug = answer_question(None, question, API_CHOICE, API_KEY)
    # best_score is legitimately None here — no vectorstore means no retrieval
    # ever runs, so there's nothing to score. That's correct, not a bug.
    print(f"\n[GENERAL case]  score=n/a    mode={mode}  Q: {question}")

    assert mode == "general", f"expected 'general', got '{mode}' for: {question}"