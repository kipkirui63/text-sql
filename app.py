import os
import sqlite3
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DB_PATH = "my_database.db"

# App config
st.set_page_config(page_title="Text-to-SQL Agent", layout="wide")
st.markdown("<h1 style='color:#58a6ff;'>🧠 AI SQL Assistant</h1>", unsafe_allow_html=True)

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# =========================
# 🎤 Voice Input with Whisper
# =========================
def transcribe_audio(audio_bytes):
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_bytes)
        return transcript["text"]
    except Exception as e:
        return f"Error transcribing: {e}"

with st.expander("🎙️ Speak your question (Click 'Start')"):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        ),
        audio_receiver_size=512,
    )

    if webrtc_ctx.audio_receiver:
        audio_bytes = b"".join([frame.to_ndarray().tobytes() for frame in webrtc_ctx.audio_receiver.iter_frames()])
        if audio_bytes:
            with st.spinner("Transcribing..."):
                with open("temp.wav", "wb") as f:
                    f.write(audio_bytes)
                with open("temp.wav", "rb") as f:
                    transcript = transcribe_audio(f)
                    st.success(f"🗣️ You said: {transcript}")
                    st.session_state.transcribed_input = transcript

# =========================
# 📂 Upload File to DB
# =========================
with st.expander("📁 Upload CSV or Excel"):
    uploaded_file = st.file_uploader("Upload .csv or .xlsx", type=["csv", "xlsx"])
    if uploaded_file:
        table_name = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        conn = sqlite3.connect(DB_PATH)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        st.session_state.uploaded_files.append((table_name, datetime.now()))
        st.success(f"✅ Added `{table_name}` to the database.")

# =========================
# 🧠 Set up LangChain Agent
# =========================
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
schema_text = db.get_table_info()

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
db_chain = SQLDatabaseChain.from_llm(
    llm=llm, db=db, verbose=True, return_intermediate_steps=True
)

# =========================
# 🔍 Explore Schema
# =========================
with st.sidebar:
    st.subheader("📊 Table Viewer")
    try:
        tables = db.get_usable_table_names()
        selected = st.selectbox("Choose a table", sorted(tables))
        if selected:
            conn = sqlite3.connect(DB_PATH)
            st.write(f"Preview of `{selected}`:")
            st.dataframe(pd.read_sql_query(f"SELECT * FROM {selected} LIMIT 5", conn))
            conn.close()
    except Exception as e:
        st.error(f"Error loading tables: {e}")

    st.subheader("📋 Schema Explorer")
    st.code(schema_text, language="sql")

    st.subheader("📜 Uploaded Files History")
    for table, date in st.session_state.uploaded_files:
        st.write(f"• `{table}` at {date.strftime('%Y-%m-%d %H:%M:%S')}")

# =========================
# 💬 Ask a Question
# =========================
st.subheader("💬 Ask your question")
input_placeholder = "Type or speak your question..."
question = st.text_input("Question:", value=st.session_state.get("transcribed_input", ""), placeholder=input_placeholder)

if question:
    try:
        result = db_chain(question)
        sql_query = result["intermediate_steps"][0]

        # 🔒 Block risky SQL
        if any(cmd in sql_query.lower() for cmd in ["drop", "delete", "update", "insert", "alter", "truncate"]):
            st.error("❌ Destructive query blocked.")
        else:
            # Display query results
            conn = sqlite3.connect(DB_PATH)
            try:
                df = pd.read_sql_query(sql_query, conn)
                st.success("✅ Query executed!")
                st.dataframe(df)
            except:
                st.info(result["result"])

            # Save history
            st.session_state.query_history.append((question, sql_query))

            # Show SQL
            st.markdown("**Generated SQL:**")
            st.code(sql_query, language="sql")

            # Offer download
            if 'df' in locals() and not df.empty:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# =========================
# 🧾 Query History
# =========================
st.markdown("---")
with st.expander("📚 Query History"):
    if st.session_state.query_history:
        for q, sql in reversed(st.session_state.query_history[-10:]):
            st.markdown(f"**Q:** {q}\n\n```sql\n{sql}\n```")
        if st.button("🧹 Clear History"):
            st.session_state.query_history.clear()
    else:
        st.info("No queries yet.")
