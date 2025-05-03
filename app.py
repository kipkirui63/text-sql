import os
import io
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import openai

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# App Config
st.set_page_config(page_title="Text-to-SQL Agent with Voice Input", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #111827;
        color: #f3f4f6;
    }
    .stApp {
        background-color: #111827;
    }
    .stTextInput input {
        background-color: #1f2937;
        color: white;
        font-size: 16px;
    }
    .stSelectbox div {
        background-color: #1f2937;
        color: white;
    }
    .stDataFrame {
        background-color: #1f2937;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Text-to-SQL Agent with Voice Input")

DB_PATH = "my_database.db"
history_file = "query_history.csv"
uploaded_log = "upload_log.csv"

# === Setup DB ===
def connect_db():
    return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

def add_file(uploaded_file):
    table = os.path.splitext(uploaded_file.name)[0].strip().replace(" ", "_")
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.warning("⚠️ Only CSV and XLSX are supported.")
        return

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame([[uploaded_file.name, now]], columns=["filename", "uploaded_at"])
    if os.path.exists(uploaded_log):
        log_df.to_csv(uploaded_log, mode="a", header=False, index=False)
    else:
        log_df.to_csv(uploaded_log, index=False)

    st.success(f"✅ Uploaded '{uploaded_file.name}' as '{table}'")

# === Sidebar: Table Preview ===
with st.sidebar:
    st.header("📊 Table Preview")
    db = connect_db()
    tables = db.get_usable_table_names()
    selected_table = st.selectbox("Preview table", sorted(tables))

    if selected_table:
        conn = sqlite3.connect(DB_PATH)
        preview = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 5", conn)
        st.dataframe(preview)

    st.markdown("---")
    st.subheader("📌 Schema")
    for table in sorted(tables):
        st.markdown(f"**🗂️ {table}**")
        df = pd.read_sql(f"PRAGMA table_info({table});", conn)
        schema_info = "\n".join([f"- `{row['name']}` ({row['type']})" for _, row in df.iterrows()])
        st.markdown(schema_info)

    st.markdown("---")
    st.subheader("🕓 Upload History")
    if os.path.exists(uploaded_log):
        logs = pd.read_csv(uploaded_log)
        st.dataframe(logs.tail(5))

    st.markdown("---")
    st.subheader("🕘 Query History")
    if os.path.exists(history_file):
        queries = pd.read_csv(history_file)
        if st.button("🧹 Clear History"):
            os.remove(history_file)
            st.success("History cleared.")
        else:
            st.dataframe(queries.tail(5))

# === LLM Setup ===
llm = ChatOpenAI(temperature=0, model_name="gpt-4")
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, return_intermediate_steps=True, verbose=False)

# === 🎤 Voice Capture ===
st.subheader("🎤 Speak your question (voice-to-SQL)")
text_result = st.empty()
recorded_audio = webrtc_streamer(
    key="voice",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)

if recorded_audio.audio_receiver:
    audio = recorded_audio.audio_receiver.get_frames(timeout=5)[0]
    audio_bytes = audio.to_ndarray().tobytes()
    with st.spinner("Transcribing voice..."):
        response = openai.Audio.transcribe("whisper-1", file=io.BytesIO(audio_bytes), filename="input.wav")
        spoken_text = response["text"]
        text_result.text_area("🎙️ Transcribed text", spoken_text, height=80)
else:
    spoken_text = None

# === 🧠 Text-to-SQL Input ===
st.subheader("🔎 Ask your question")
default_hint = "🎤 Speak or type your question (e.g. show top 5 products)..."
question = st.text_input("Your question", value=spoken_text or "", placeholder=default_hint)

if question:
    try:
        result = db_chain(question)
        sql = result["intermediate_steps"][0]

        if any(x in sql.lower() for x in ["drop", "delete", "alter", "truncate"]):
            st.error("⛔ Destructive query blocked.")
        else:
            try:
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql_query(sql, conn)
                st.success("✅ Query successful!")
                st.dataframe(df)
            except:
                st.markdown(f"**Result:** {result['result']}")

            st.markdown("---")
            st.markdown("🧾 **Generated SQL:**")
            st.code(sql, language="sql")

            # Save history
            new_log = pd.DataFrame([[datetime.now(), question, sql]], columns=["time", "query", "sql"])
            if os.path.exists(history_file):
                new_log.to_csv(history_file, mode="a", header=False, index=False)
            else:
                new_log.to_csv(history_file, index=False)

    except Exception as e:
        st.error(f"❌ Error: {e}")
