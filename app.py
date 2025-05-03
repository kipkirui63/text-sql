# app.py
import os
import base64
import requests
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# App config
st.set_page_config(page_title="Voice-to-SQL", layout="wide", initial_sidebar_state="expanded")

# ================== Dark Mode Styling ==================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #fafafa;
}
button {
    background-color: #1f77b4 !important;
    color: white !important;
    border-radius: 5px !important;
}
h1, h2, h3 {
    color: #61dafb;
}
textarea, .stTextInput, .stTextArea {
    background-color: #262730 !important;
    color: white !important;
}
table {
    background-color: #1e1e1e;
}
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Voice & Text SQL Assistant")

DB_PATH = "my_database.db"

# ================= DB Setup ====================
def connect_to_db():
    return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

def add_file_to_db(uploaded_file):
    file_name = uploaded_file.name
    table_name = os.path.splitext(file_name)[0]

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.warning("⚠️ Only .csv or .xlsx files are supported.")
        return

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    st.success(f"✅ File '{file_name}' added as table '{table_name}'")

# Upload placeholder
with st.sidebar:
    st.header("📂 Upload New File")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        add_file_to_db(uploaded_file)

# Connect DB
db = connect_to_db()
llm = ChatOpenAI(temperature=0, model_name="gpt-4")
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True, return_intermediate_steps=True)

# Query history
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ================= Schema Explorer ====================
with st.sidebar:
    st.header("📊 Available Tables")
    try:
        tables = db.get_usable_table_names()
        for table in sorted(tables):
            st.markdown(f"- `{table}`")
            with sqlite3.connect(DB_PATH) as conn:
                cols = pd.read_sql_query(f"PRAGMA table_info('{table}')", conn)
                col_names = cols['name'].tolist()
                st.caption(f"    Columns: {', '.join(col_names)}")
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch tables: {e}")

    st.header("🕘 Query History")
    for i, q in enumerate(st.session_state.query_history[::-1][:10]):
        st.markdown(f"{i+1}. {q}")

# ================= Voice Recorder ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔊 Click and Speak your query")
    custom_html = """
    <script>
    let mediaRecorder;
    let audioChunks = [];
    let stream;

    async function startRecording() {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const arrayBuffer = await audioBlob.arrayBuffer();
            const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

            window.parent.postMessage({ type: 'audioBase64', data: base64Audio }, '*');
            audioChunks = [];
        };

        mediaRecorder.start();
        document.getElementById('status').innerText = '🎙️ Recording...';
    }

    function stopRecording() {
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
        document.getElementById('status').innerText = '✅ Recording stopped';
    }
    </script>
    <button onclick="startRecording()">🎤 Start</button>
    <button onclick="stopRecording()">⏹️ Stop & Transcribe</button>
    <p id="status"></p>
    """
    st.components.v1.html(custom_html, height=150)

with col2:
    st.markdown("### ⌨️ Or Type your query")
    typed_query = st.text_input("Query", placeholder="Type your question here...", label_visibility="collapsed")
    if typed_query:
        result = db_chain(typed_query)
        sql_query = result['intermediate_steps'][0]
        if isinstance(sql_query, dict) and 'SQLQuery' in sql_query:
            sql_query = sql_query['SQLQuery']
        sql_query = str(sql_query)

        if any(word in sql_query.lower() for word in ["drop", "delete", "update"]):
            st.error("❌ Dangerous SQL blocked.")
        else:
            try:
                df = pd.read_sql_query(sql_query, sqlite3.connect(DB_PATH))
                st.session_state.query_history.append(typed_query)
                st.success("✅ Query executed!")
                st.code(sql_query)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download CSV", csv, "query_results.csv", "text/csv")
            except Exception as e:
                st.error(f"⚠️ Failed to execute query: {e}")

# ================= Transcription Logic ====================
if "base64_audio" not in st.session_state:
    st.session_state.base64_audio = ""

def transcribe_audio(base64_audio):
    audio_bytes = base64.b64decode(base64_audio)
    response = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": "whisper-1"}
    )
    return response.json().get("text", "")

# JS listener
st.components.v1.html("""
<script>
window.addEventListener("message", (event) => {
    if (event.data.type === 'audioBase64') {
        const audio = event.data.data;
        const queryParams = new URLSearchParams(window.location.search);
        queryParams.set("audio", audio);
        window.location.search = queryParams.toString();
    }
});
</script>
""", height=0)

# Handle audio transcription
audio_base64 = st.query_params.get("audio")
if audio_base64:
    transcription = transcribe_audio(audio_base64[0])
    st.text_area("📝 Whisper Transcript", transcription)

    if transcription:
        result = db_chain(transcription)
        sql_query = result['intermediate_steps'][0]
        if isinstance(sql_query, dict) and 'SQLQuery' in sql_query:
            sql_query = sql_query['SQLQuery']
        sql_query = str(sql_query)

        if any(word in sql_query.lower() for word in ["drop", "delete", "update"]):
            st.error("❌ Dangerous SQL blocked.")
        else:
            try:
                df = pd.read_sql_query(sql_query, sqlite3.connect(DB_PATH))
                st.session_state.query_history.append(transcription)
                st.success("✅ Query executed!")
                st.code(sql_query)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download CSV", csv, "query_results.csv", "text/csv")
            except Exception as e:
                st.error(f"⚠️ Failed to execute query: {e}")