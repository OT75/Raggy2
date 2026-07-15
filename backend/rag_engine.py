from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_SYSTEM_PROMPT = (
    "You are Raggy, a document assistant with three response modes: "
    "(1) general chat when no documents are loaded, "
    "(2) general knowledge with a disclosed fallback when documents are loaded "
    "but don't cover the question, and "
    "(3) retrieval-augmented answers grounded in uploaded documents when they do. "
    "Do not mention these modes, your architecture, or how you work unless the user "
    "explicitly asks what you are, how you work, or something equivalent. "
    "For all other messages, respond naturally as a normal conversational assistant."
)
FALLBACK_INSTRUCTION = (
    "The uploaded documents don't contain information relevant to this specific question. "
    "Answer using general knowledge only. Do NOT invent specific facts, numbers, names, "
    "passwords, codes, or other concrete details as if they belong to a real document, "
    "organization, or person you don't actually have information about. "
    "If the question requires specific information you don't have, say so plainly "
    "instead of guessing or making something up."
)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)


def get_embeddings(api_choice, api_key):
    """Returns the embeddings model alone, so callers can build a fresh FAISS
    index OR add texts to an existing one without re-embedding what's already indexed."""
    if api_choice == "OpenAI" and api_key:
        return OpenAIEmbeddings(api_key=api_key)
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_vectorstore(text_chunks, api_choice, api_key):
    embeddings = get_embeddings(api_choice, api_key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_llm(api_choice, api_key):
    if api_choice == "OpenAI":
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, api_key=api_key)
    else:
        return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5, api_key=api_key)


def _build_history_messages(chat_history, max_turns=6):
    """Convert session-state message dicts into (role, content) tuples LangChain accepts."""
    if not chat_history:
        return []
    trimmed = chat_history[-max_turns:]
    role_map = {"user": "human", "assistant": "ai"}
    return [(role_map[m["role"]], m["content"]) for m in trimmed if m["role"] in role_map]


def contextualize_question(llm, question, history_messages):
    """Rewrite a follow-up question into a standalone one using recent chat history,
    so retrieval isn't searching with an ambiguous fragment like 'what is the policy?'
    Deliberately conservative: only resolves pronouns/references, never expands scope."""
    if not history_messages:
        return question

    history_text = "\n".join(
        f"{'User' if role == 'human' else 'Assistant'}: {content}"
        for role, content in history_messages
    )

    rewrite_prompt = f"""Given the conversation history and a follow-up question, decide if the follow-up contains a pronoun or vague reference (like "it", "that", "this", "the same one") that depends on the history to understand.

If it does NOT need history to be understood on its own, return the follow-up question EXACTLY as written, with no changes.

If it DOES need history, rewrite it minimally — only replace the pronoun/reference with what it refers to. Do not add new topics, do not add hedging language, do not make it longer than necessary. Keep it as close to the original wording as possible.

History:
{history_text}

Follow-up question: {question}

Rewritten question (or the original if no rewrite is needed):"""

    response = llm.invoke(rewrite_prompt)
    rewritten = response.content.strip()

    # Guardrail: if the rewrite ballooned in length, the model probably over-elaborated
    # instead of doing a minimal edit — fall back to the original question rather than
    # risk searching with a distorted query.
    if len(rewritten) > len(question) * 2.5:
        return question

    return rewritten


def answer_question(vectorstore, question, api_choice, api_key, chat_history=None, score_threshold=1.0):
    llm = get_llm(api_choice, api_key)
    history_messages = _build_history_messages(chat_history)

    debug_info = {"search_query": None, "best_score": None}

    if vectorstore is None:
        messages = [("system", BASE_SYSTEM_PROMPT)] + history_messages + [("human", question)]
        response = llm.invoke(messages)
        return response.content, [], "general", debug_info

    search_query = contextualize_question(llm, question, history_messages)
    debug_info["search_query"] = search_query

    docs_with_scores = vectorstore.similarity_search_with_score(search_query, k=4)

    if not docs_with_scores:
        fallback_system = f"{BASE_SYSTEM_PROMPT}\n\n{FALLBACK_INSTRUCTION}"
        messages = [("system", fallback_system)] + history_messages + [("human", question)]
        response = llm.invoke(messages)
        return response.content, [], "general_fallback", debug_info

    best_score = float(min(score for _, score in docs_with_scores))
    debug_info["best_score"] = best_score

    if best_score > score_threshold:
        fallback_system = f"{BASE_SYSTEM_PROMPT}\n\n{FALLBACK_INSTRUCTION}"
        messages = [("system", fallback_system)] + history_messages + [("human", question)]
        response = llm.invoke(messages)
        return response.content, [], "general_fallback", debug_info

    docs = [doc for doc, _ in docs_with_scores]
    context = "\n\n".join(doc.page_content for doc in docs)
    grounding_instruction = (
        "Answer the question using ONLY the context below. "
        "Be concise and cite specifics from the context where relevant.\n\n"
        f"Context:\n{context}"
    )
    system_msg = f"{BASE_SYSTEM_PROMPT}\n\n{grounding_instruction}"
    messages = [("system", system_msg)] + history_messages + [("human", question)]
    response = llm.invoke(messages)
    return response.content, docs, "rag", debug_info