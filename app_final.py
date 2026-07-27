import os, io, fitz, docx
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- SECRETS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TEAM_PASSWORD = "Meta@Team2025"
ADMIN_PASSWORD = "SOP@2024"
PERSIST_DIR = "./chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

st.set_page_config(page_title="SOP Assistant", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.block-container { padding-bottom: 120px !important; }
.chat-bubble-user { background: #0084ff; color: white; padding: 12px 18px; border-radius: 18px 18px 4px 18px; margin: 8px 0 8px auto; max-width: 75%; width: fit-content; }
.chat-bubble-ai { background: white; color: #111; padding: 14px 18px; border-radius: 18px 18px 18px 4px; margin: 8px auto 8px 0; max-width: 80%; width: fit-content; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.source-box { background: #eef2ff; border-left: 3px solid #0084ff; padding: 6px 10px; margin-top: 10px; border-radius: 6px; font-size: 11px; }
div[data-testid="stChatInput"] { position: fixed !important; bottom: 20px !important; left: 50% !important; transform: translateX(-50%) !important; width: 70% !important; max-width: 800px !important; z-index: 9999 !important; background: white !important; border-radius: 28px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important; }
</style>
""", unsafe_allow_html=True)

# --- AUTH ---
if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.role = None

if not st.session_state.auth:
    st.title("🔒 SOP Assistant")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if pwd == TEAM_PASSWORD:
            st.session_state.auth = True
            st.session_state.role = "Team"
            st.rerun()
        elif pwd == ADMIN_PASSWORD:
            st.session_state.auth = True
            st.session_state.role = "Admin"
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()

# --- EMBEDDINGS & DB ---
@st.cache_resource
def get_emb():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )

def get_vs():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_emb())

def extract_docx_text(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    full = []
    for p in doc.paragraphs:
        if p.text.strip():
            full.append(p.text.strip())
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join([c.text.strip() for c in row.cells if c.text.strip()])
            if row_text:
                full.append(row_text)
    return "\n".join(full)

def get_llm():
    if not GROQ_API_KEY:
        return None
    return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY, temperature=0.1)

# --- SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I am SOP Assistant 🤖\n\nAsk me anything from your SOPs."}]

st.sidebar.title(f"👤 {st.session_state.role}")
if st.sidebar.button("Logout"):
    st.session_state.auth = False
    st.session_state.role = None
    st.rerun()

st.title("🤖 SOP Assistant")
tab1, tab2 = st.tabs(["💬 Chat", "📤 Upload (Admin)"])

with tab2:
    if st.session_state.role != "Admin":
        st.warning("Admin only can upload SOPs")
    else:
        files = st.file_uploader("Upload SOPs", type=["pdf","docx","txt"], accept_multiple_files=True)
        if files and st.button("Process & Save", type="primary"):
            texts = []
            metas = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            for f in files:
                b = f.getvalue()
                txt = ""
                if f.name.endswith(".pdf"):
                    doc = fitz.open(stream=b, filetype="pdf")
                    txt = "".join([(p.get_text() or "") for p in doc])
                elif f.name.endswith(".docx"):
                    txt = extract_docx_text(b)
                else:
                    txt = b.decode("utf-8", errors="ignore")
                
                if len(txt) < 20:
                    continue
                
                clean_chunks = [c for c in splitter.split_text(txt) if len(c.strip()) > 20]
                texts.extend(clean_chunks)
                metas.extend([{"source": f.name} for _ in clean_chunks])

            if texts:
                vs = get_vs()
                vs.add_texts(texts=texts, metadatas=metas)
                vs.persist()
                st.success(f"✅ Saved {len(texts)} chunks from {len(files)} files!")
            else:
                st.error("No text found in files")

with tab1:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">{msg["content"]}</div>', unsafe_allow_html=True)

    prompt = st.chat_input("Ask your SOP...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-bubble-user">{prompt}</div>', unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            try:
                vs = get_vs()
                docs = vs.similarity_search(prompt, k=3)
                
                if not docs:
                    answer = "No SOPs found. Please ask admin to upload SOP documents in Upload tab."
                else:
                    context = "\n\n".join([d.page_content for d in docs])
                    sources = ", ".join(set([d.metadata.get("source","") for d in docs]))
                    
                    llm = get_llm()
                    if not llm:
                        answer = f"**Context from SOPs:**\n\n{context}\n\n*Add GROQ_API_KEY in Streamlit Secrets to get AI answers*"
                    else:
                        template = """You are SOP Assistant. Answer ONLY from given context.

Context:
{context}

Question: {question}

If answer not in context, say "Not found in SOPs". Be concise and helpful.

Answer:"""
                        prompt_template = ChatPromptTemplate.from_template(template)
                        chain = prompt_template | llm
                        res = chain.invoke({"context": context, "question": prompt})
                        answer = res.content + f'\n\n<div class="source-box">📄 Sources: {sources}</div>'

            except Exception as e:
                answer = f"Error: {str(e)} - If first time, upload SOPs in Upload tab."

        st.markdown(f'<div class="chat-bubble-ai">{answer}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
