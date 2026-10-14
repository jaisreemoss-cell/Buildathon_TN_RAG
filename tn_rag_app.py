import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_FILE = "scraped_schemes.txt"
INDEX_DIR = "faiss_index"
SOURCE_URL = "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 8                 # default number of scheme chunks retrieved per question
MAX_CHUNK_CHARS = 2000    # safety net: any scheme longer than this is split further

SCHEME_HEADING_PATTERN = (
    r"(?=\b[1-5]\.\s+[A-Z][a-z])"
    r"|(?=Promoting SRI)"
    r"|(?=Distribution of Conoweeder and Marker to 3000)"
)

EXAMPLE_QUESTIONS = [
    "What schemes are available for paddy farmers?",
    "Pumpset replacement subsidy for SC/ST farmers below 5 HP?",
    "Which districts are covered under NFSM rice?",
    "What is the hire charge for a bulldozer?",
    "What support is there for SRI cultivation?",
]

PROMPT_TEMPLATE = """You are a helpful agricultural assistant for Tamil Nadu paddy farmers.
Answer the question strictly based ONLY on the retrieved context below.
If the answer cannot be found in the context, say clearly that the information is not available in the database.
Do not use outside knowledge or web search.
When you mention a subsidy amount, name the scheme it belongs to.
Use short paragraphs or bullet points so the answer is easy to scan.

Context:
{context}

Question:
{question}

Answer:"""

# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TN Paddy Scheme Finder", page_icon="🌾", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind+Madurai:wght@400;500;700&display=swap');

.stApp { font-family: 'Hind Madurai', 'Segoe UI', Arial, sans-serif; }

/* Hero: rows of young paddy, with a harvest-gold edge */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 14px;
    border-bottom: 6px solid #D8A630;
    background-color: #24552F;
    background-image: repeating-linear-gradient(
        104deg,
        rgba(255, 255, 255, 0.00) 0px,
        rgba(255, 255, 255, 0.00) 10px,
        rgba(156, 200, 92, 0.22) 10px,
        rgba(156, 200, 92, 0.22) 14px
    );
    padding: 2.2rem 2rem 1.8rem 2rem;
    margin-bottom: 1.4rem;
}
.hero h1 {
    color: #F4F8F1;
    font-family: 'Hind Madurai', sans-serif;
    font-weight: 700;
    font-size: 2.3rem;
    line-height: 1.1;
    margin: 0 0 0.5rem 0;
    padding: 0;
}
.hero p {
    color: #DCE8D3;
    font-size: 1.05rem;
    line-height: 1.5;
    max-width: 34rem;
    margin: 0;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.6rem 0.9rem;
}

/* Sidebar */
[data-testid="stSidebar"] h3 {
    color: #24552F;
    font-weight: 700;
    margin-top: 0.4rem;
}
.kb-stat {
    font-size: 2rem;
    font-weight: 700;
    color: #24552F;
    line-height: 1;
}
.kb-label { color: #5A6B5E; font-size: 0.92rem; }

.footnote { color: #5A6B5E; font-size: 0.85rem; margin-top: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Add it to your .env file, then restart the app.")
    st.stop()


# ---------------------------------------------------------------------------
# Chunking: one chunk per scheme
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\bTop\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_by_scheme(text: str) -> list[Document]:
    parts = [p.strip() for p in re.split(SCHEME_HEADING_PATTERN, text)]
    parts = [p for p in parts if len(p) > 60]
    return [
        Document(
            page_content=part,
            metadata={
                "state": "Tamil Nadu",
                "source": SOURCE_URL,
                "chunk_id": i,
                "heading": part[:80],
            },
        )
        for i, part in enumerate(parts)
    ]


def enforce_max_size(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=200)
    result = []
    for doc in docs:
        if len(doc.page_content) <= MAX_CHUNK_CHARS:
            result.append(doc)
        else:
            result.extend(splitter.split_documents([doc]))
    return result


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def index_is_stale() -> bool:
    if not os.path.exists(INDEX_DIR):
        return True
    return os.path.getmtime(SOURCE_FILE) > os.path.getmtime(INDEX_DIR)


def build_vector_store(embeddings: OpenAIEmbeddings) -> FAISS:
    documents = TextLoader(SOURCE_FILE, encoding="utf-8").load()
    docs = enforce_max_size(split_by_scheme(clean_text(documents[0].page_content)))
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    return vectorstore


@st.cache_resource(show_spinner=False)
def load_vector_store() -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if index_is_stale():
        return build_vector_store(embeddings)
    # Safe here because this index was created locally by this app.
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


@st.cache_resource(show_spinner=False)
def load_chain():
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


if not os.path.exists(SOURCE_FILE):
    st.error(f"'{SOURCE_FILE}' was not found. Run the scraper first, then reload this page.")
    st.stop()

with st.spinner("Loading the schemes database..."):
    vectorstore = load_vector_store()
chain = load_chain()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Knowledge base")
    st.markdown(
        f'<div class="kb-stat">{vectorstore.index.ntotal}</div>'
        f'<div class="kb-label">scheme sections indexed from the TNAU paddy schemes page</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    top_k = st.slider(
        "Scheme sections to search per question",
        min_value=3,
        max_value=12,
        value=TOP_K,
        help="Higher values help with questions like 'list all schemes'. Lower values use fewer tokens.",
    )

    if st.button("Rebuild index"):
        with st.spinner("Rebuilding the index from scraped_schemes.txt..."):
            st.cache_resource.clear()
            build_vector_store(OpenAIEmbeddings(model=EMBEDDING_MODEL))
        st.success("Index rebuilt.")
        st.rerun()

    st.markdown("### Try asking")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"example_{q}"):
            st.session_state.pending_question = q

    st.markdown("### Conversation")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Paddy scheme finder</h1>
  <p>Ask about Tamil Nadu government schemes and subsidies for paddy farmers.
  Answers come only from the TNAU agritech schemes page.</p>
</div>
""",
    unsafe_allow_html=True,
)


def show_sources(sources: list[str]):
    with st.expander(f"Scheme sections used ({len(sources)})"):
        for heading in sources:
            st.markdown(f"- {heading}…")


if not st.session_state.messages:
    st.info("Type a question below, or pick one of the examples in the sidebar.")

for message in st.session_state.messages:
    avatar = "🌾" if message["role"] == "assistant" else "🧑‍🌾"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

typed_question = st.chat_input("Ask about a scheme, subsidy, district or eligibility rule")
question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🌾"):
        try:
            with st.spinner("Searching the schemes database..."):
                docs = vectorstore.similarity_search(question, k=top_k)
            answer = st.write_stream(
                chain.stream({"context": format_docs(docs), "question": question})
            )
            sources = [doc.metadata.get("heading", "Untitled section") for doc in docs]
            show_sources(sources)
        except Exception as e:
            answer = f"The answer could not be generated: {e}. Check your API key and network connection, then try again."
            sources = []
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

st.markdown(
    '<p class="footnote">Figures come from the TNAU agritech portal and may not reflect '
    "current government rates. Confirm with your local Agricultural Extension Centre before applying.</p>",
    unsafe_allow_html=True,
)