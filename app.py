import io
import re
from pathlib import Path
from typing import List, Dict

import fitz 
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "CaseLens AI"
MAX_CHARS_FOR_LLM = 3600

st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner=False)
def load_generator():
    from transformers import pipeline

    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
    )


def extract_pdf(file_bytes: bytes) -> List[Dict]:
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


def extract_txt(file_bytes: bytes) -> List[Dict]:
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    return [{"page": 1, "text": text}] if text else []


def chunk_pages(pages: List[Dict], max_chars: int = 1000) -> List[Dict]:
    chunks = []
    for page in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page["text"]) if p.strip()]
        current = []
        current_len = 0
        chunk_no = 1

        for paragraph in paragraphs:
            if current and current_len + len(paragraph) > max_chars:
                text = "\n\n".join(current)
                chunks.append(
                    {
                        "page": page["page"],
                        "chunk": chunk_no,
                        "label": f"Page {page['page']} · Chunk {chunk_no}",
                        "text": text,
                    }
                )
                chunk_no += 1
                current = []
                current_len = 0
            current.append(paragraph)
            current_len += len(paragraph)

        if current:
            chunks.append(
                {
                    "page": page["page"],
                    "chunk": chunk_no,
                    "label": f"Page {page['page']} · Chunk {chunk_no}",
                    "text": "\n\n".join(current),
                }
            )
    return chunks


def sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def extractive_summary(text: str, max_sentences: int = 6) -> str:
    sents = sentences(text)
    if not sents:
        return text[:1200]
    chosen = sents[:max_sentences]
    return " ".join(chosen)


def ai_generate(prompt: str, max_new_tokens: int = 180) -> str:
    try:
        generator = load_generator()
        result = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return result[0]["generated_text"].strip()
    except Exception:
        return ""


def summarize_document(text: str) -> str:
    source = text[:MAX_CHARS_FOR_LLM]
    prompt = (
        "Summarize this legal document in clear, neutral language. "
        "Focus on parties, dispute, important facts, decision, and reasoning. "
        "Do not invent information.\n\nDOCUMENT:\n" + source
    )
    generated = ai_generate(prompt)
    return generated or extractive_summary(text)


def timeline_items(text: str, limit: int = 10) -> List[str]:
    date_patterns = [
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]
    results = []
    for sent in sentences(text):
        if any(re.search(pattern, sent, flags=re.I) for pattern in date_patterns):
            results.append(sent)
            if len(results) >= limit:
                break
    return results


def key_facts(text: str, limit: int = 7) -> List[str]:
    sents = sentences(text)
    keywords = ("court", "plaintiff", "defendant", "claim", "agreement", "ordered", "held", "found", "evidence", "judgment")
    ranked = sorted(
        sents,
        key=lambda s: (sum(k in s.lower() for k in keywords), min(len(s), 300)),
        reverse=True,
    )
    return ranked[:limit]


def retrieve(question: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    if not chunks:
        return []
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=12000)
    matrix = vectorizer.fit_transform(corpus + [question])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    indices = sims.argsort()[::-1][:top_k]
    return [dict(chunks[i], score=float(sims[i])) for i in indices]


def answer_question(question: str, retrieved: List[Dict]) -> str:
    if not retrieved:
        return "I could not find relevant context in the uploaded document."
    context = "\n\n".join(f"[{c['label']}] {c['text']}" for c in retrieved)
    context = context[:MAX_CHARS_FOR_LLM]
    prompt = (
        "Answer the question using only the legal context below. "
        "If the answer is not supported by the context, say that it is not available. "
        "Be concise and do not invent facts.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
    )
    generated = ai_generate(prompt, max_new_tokens=140)
    if generated:
        return generated
    return "The most relevant passages are shown below. The local AI model could not load, so please review the cited context directly."


def load_sample() -> bytes:
    return Path("sample_case.txt").read_bytes()


st.title("⚖️ CaseLens AI")
st.caption("Open-source legal document intelligence MVP — summary, timeline, retrieval, Q&A and citations.")
st.info("Research/demo tool only. It does not provide legal advice.")

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    use_sample = st.button("Use included sample case")
    st.divider()
    st.markdown("**Open-source stack**")
    st.write("Streamlit · PyMuPDF · scikit-learn TF-IDF · FLAN-T5 Small")
    st.caption("The model is loaded only when an AI action is requested. If model loading fails, the app falls back to extractive output.")

if use_sample:
    st.session_state["sample_bytes"] = load_sample()
    st.session_state["sample_name"] = "sample_case.txt"

file_bytes = None
file_name = None
if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_name = uploaded.name
elif "sample_bytes" in st.session_state:
    file_bytes = st.session_state["sample_bytes"]
    file_name = st.session_state.get("sample_name", "sample_case.txt")

if not file_bytes:
    st.subheader("Start here")
    st.write("Upload a public legal PDF/TXT file, or click **Use included sample case** in the sidebar.")
    st.stop()

try:
    if file_name.lower().endswith(".pdf"):
        pages = extract_pdf(file_bytes)
    else:
        pages = extract_txt(file_bytes)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()

if not pages:
    st.warning("No readable text was found in this document.")
    st.stop()

chunks = chunk_pages(pages)
full_text = "\n\n".join(p["text"] for p in pages)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Pages", len(pages))
metric2.metric("Text chunks", len(chunks))
metric3.metric("Characters", f"{len(full_text):,}")

tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Timeline & Facts", "Ask the Document", "Source Text"])

with tab1:
    st.subheader("Document summary")
    if st.button("Generate summary", type="primary"):
        with st.spinner("Generating with the open-source model..."):
            st.session_state["summary"] = summarize_document(full_text)
    if "summary" in st.session_state:
        st.write(st.session_state["summary"])
    else:
        st.caption("Click **Generate summary**. The first run may take longer while the model downloads.")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Timeline")
        items = timeline_items(full_text)
        if items:
            for item in items:
                st.markdown(f"- {item}")
        else:
            st.caption("No explicit dates were detected in the document.")
    with col2:
        st.subheader("Key facts")
        for fact in key_facts(full_text):
            st.markdown(f"- {fact}")

with tab3:
    st.subheader("Ask a grounded question")
    question = st.text_input("Question", placeholder="What did the court decide and why?")
    if st.button("Find answer", type="primary", disabled=not bool(question.strip())):
        found = retrieve(question, chunks, top_k=3)
        with st.spinner("Retrieving relevant context and generating an answer..."):
            answer = answer_question(question, found)
        st.session_state["last_answer"] = answer
        st.session_state["last_sources"] = found

    if "last_answer" in st.session_state:
        st.markdown("### Answer")
        st.write(st.session_state["last_answer"])
        st.markdown("### Citations / retrieved evidence")
        for source in st.session_state.get("last_sources", []):
            with st.expander(f"{source['label']} · relevance {source['score']:.2f}"):
                st.write(source["text"])

with tab4:
    st.subheader(file_name)
    for page in pages:
        with st.expander(f"Page {page['page']}", expanded=page["page"] == 1):
            st.text(page["text"])

st.divider()
st.caption("CaseLens AI MVP · Built with open-source components · No paid API required")
