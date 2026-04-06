import os
import time
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from llama_cpp import Llama

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "Chroma_db")
MODEL_PATH = os.path.join(BASE_DIR, "Models", "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")

MAX_TOKENS = 400
TEMPERATURE = 0.1
TOP_K = 5

STOP_TOKENS = ["<|eot_id|>","<|end_header_id|>","Question:","Question :","Cordialement","Note :","Merci","[Votre nom]","Vous :","\n\n\n","Reponse :","___","Assistant expert"]

st.set_page_config(page_title="Agent RAG TGR", page_icon="🏛️", layout="wide")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})

@st.cache_resource
def load_database(_embeddings):
    return Chroma(persist_directory=DB_DIR, embedding_function=_embeddings)

@st.cache_resource
def load_llm():
    return Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, n_batch=256, verbose=False)

def detect_language(text):
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total = len(text.strip())
    if total == 0:
        return "fr"
    return "ar" if arabic_chars > total * 0.3 else "fr"

def search_docs(database, query):
    return database.similarity_search("query: " + query, k=TOP_K)

def make_prompt(query, documents):
    parts = []
    for i, doc in enumerate(documents):
        text = doc.page_content.replace("passage: ", "")
        source = os.path.basename(doc.metadata.get("source", "inconnu"))
        page = doc.metadata.get("page", "?")
        parts.append(f"[Doc {i+1} - {source} p{page}]\n{text}")
    context = "\n\n".join(parts)
    lang = detect_language(query)
    if lang == "ar":
        sys_msg = "أنت مساعد خبير في الخزينة العامة للمملكة المغربية. أجب بالعربية فقط. استخدم فقط المعلومات في السياق. كن مختصراً. لا تكرر."
        usr_msg = f"السياق:\n{context}\n\nالسؤال: {query}"
    else:
        sys_msg = "Vous etes assistant expert de la TGR. Repondez directement. Utilisez uniquement le contexte. Soyez concis. Ne dites jamais Cordialement ou Merci."
        usr_msg = f"Contexte:\n{context}\n\nQuestion: {query}"
    return f"<|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{usr_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def get_response(model, prompt):
    output = model(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=0.9, repeat_penalty=1.2, stop=STOP_TOKENS)
    text = output["choices"][0]["text"].strip().rstrip("(").strip()
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if any(s.startswith(x) for x in ["_____", "Note :", "Cordialement", "Merci", "[Votre", "Assistant expert"]):
            break
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    return result if result else "Information non trouvee."

def get_sources(documents):
    seen = set()
    sources = []
    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", "inconnu"))
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append(f"{source} (page {page})")
    return sources

with st.sidebar:
    st.title("🏛️ Agent RAG TGR")
    st.markdown("---")
    st.markdown("Agent intelligent pour les documents de la TGR")
    st.markdown("---")
    st.markdown("🇫🇷 Francais | 🇲🇦 Arabe")
    st.markdown("---")
    st.markdown("Llama-3-8B | E5-multilingual | ChromaDB")
    if st.button("Effacer historique"):
        st.session_state.messages = []
        st.rerun()

st.title("🏛️ Agent RAG - Tresorerie Generale du Royaume")
st.markdown("Posez vos questions en **francais** ou en **arabe**")

with st.spinner("Chargement..."):
    embeddings = load_embeddings()
    database = load_database(embeddings)
    model = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.write(f"📄 {s}")
        if msg.get("time"):
            st.caption(msg["time"])

query = st.chat_input("Posez votre question ici...")
if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        lang = detect_language(query)
        st.caption(f"Langue : {'🇲🇦 Arabe' if lang == 'ar' else '🇫🇷 Francais'}")
        with st.spinner("Recherche en cours..."):
            start = time.time()
            docs = search_docs(database, query)
            prompt = make_prompt(query, docs)
            response = get_response(model, prompt)
            elapsed = time.time() - start
        st.write(response)
        sources = get_sources(docs)
        with st.expander("📚 Sources"):
            for s in sources:
                st.write(f"📄 {s}")
        time_msg = f"⏱️ {elapsed:.1f}s"
        st.caption(time_msg)
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources, "time": time_msg})