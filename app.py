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

db = connect_to_db()
llm = ChatOpenAI(temperature=0, model_name="gpt-4")
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True, return_intermediate_steps=True)

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
    typed_query = st.text_input("", placeholder="Type your question here...")
    if typed_query:
        result = db_chain(typed_query)
        sql_query = result['intermediate_steps'][0]

        if any(word in sql_query.lower() for word in ["drop", "delete", "update"]):
            st.error("❌ Dangerous SQL blocked.")
        else:
            st.success("✅ Query executed successfully!")
            st.markdown("**Generated SQL:**")
            st.code(sql_query)
            st.dataframe(result['result'])

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

# JS listener to get audio from browser
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
audio_base64 = st.experimental_get_query_params().get("audio")
if audio_base64:
    transcription = transcribe_audio(audio_base64[0])
    st.text_area("📝 Whisper Transcript", transcription)

    if transcription:
        result = db_chain(transcription)
        sql_query = result['intermediate_steps'][0]

        if any(word in sql_query.lower() for word in ["drop", "delete", "update"]):
            st.error("❌ Dangerous SQL blocked.")
        else:
            st.success("✅ Query executed successfully!")
            st.markdown("**Generated SQL:**")
            st.code(sql_query)
            st.dataframe(result['result'])
else:
    st.info("🎤 Speak and click 'Stop & Transcribe' to query.")
