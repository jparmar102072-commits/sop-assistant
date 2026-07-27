import os, fitz, docx, streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# HF uses secrets - works both local and HF
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "gsk_YOUR_KEY_HERE")
KNN = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
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
footer {display:none !important}
</style>
""", unsafe_allow_html=True)

if "auth" not in st.session_state:
    st.session_state.auth=False; st.session_state.role=None
if not st.session_state.auth:
    st.title("🔒 SOP Assistant")
    pwd=st.text_input("Password", type="password")
    if st.button("Login"):
        if pwd==TEAM_PASSWORD: st.session_state.auth=True; st.session_state.role="Team"; st.rerun()
        elif pwd==ADMIN_PASSWORD: st.session_state.auth=True; st.session_state.role="Admin"; st.rerun()
        else: st.error("Wrong")
    st.stop()

st.sidebar.title(f"👤 {st.session_state.role}")
if st.sidebar.button("Clear Chat"):
    st.session_state.messages=[{"role":"assistant","content":"Hi! Ask me from SOPs 🤖"}]; st.rerun()
if st.sidebar.button("Logout"): st.session_state.auth=False; st.rerun()

@st.cache_resource
def get_emb():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':True, 'batch_size':64})

def get_vs():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_emb())

def extract_docx_text(file_bytes):
    import io
    doc = docx.Document(io.BytesIO(file_bytes))
    full=[]
    for p in doc.paragraphs:
        if p.text.strip(): full.append(p.text.strip())
    for table in doc.tables:
        for row in table.rows:
            row_text=" | ".join([c.text.strip() for c in row.cells if c.text.strip()])
            if row_text: full.append(row_text)
    return "\n".join(full)

if "messages" not in st.session_state:
    st.session_state.messages=[{"role":"assistant","content":"Hi! I am SOP Assistant 🤖\n\nAsk me anything from your SOPs."}]

st.title("🤖 SOP Assistant")
tab1, tab2 = st.tabs(["💬 Chat", "📤 Upload"])

with tab2:
    if st.session_state.role!="Admin": st.warning("Admin only")
    else:
        files=st.file_uploader("Upload", type=["pdf","docx","txt"], accept_multiple_files=True)
        if files and st.button("Process & Save", type="primary"):
            texts=[]; metas=[]
            for f in files:
                b=f.getvalue(); txt=""
                if f.name.endswith(".pdf"):
                    doc=fitz.open(stream=b, filetype="pdf")
                    txt="".join([(p.get_text() or "") for p in doc])
                elif f.name.endswith(".docx"): txt=extract_docx_text(b)
                else: txt=b.decode("utf-8", errors="ignore")
                if len(txt)<20: continue
                splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                clean=[c for c in splitter.split_text(txt) if len(c.strip())>10]
                texts.extend(clean