import os
import sqlite3
import pandas as pd
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.utilities.sql_database import SQLDatabase
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import openai
import tempfile
import av
from datetime import datetime

st.set_page_config(page_title="Voice SQL Agent", layout="wide")
st.title("🧠 Text-to-SQL Agent with Voice Input")

openai.api_key = os.getenv("OPENAI_API_KEY")
DB_PATH = "my_database.db"
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

def connect_to_db():
    return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

def get_all_schemas(db) -> str:
    schema_str = ""
    for table in db.get_usable_table_names():
        try:
            columns = db.run(f"PRAGMA table_info({table})")
            col_lines = [f"{col['name']} ({col['type']})" for col in columns]
            schema_str += f"Table: {table}\\nColumns: {', '.join(col_lines)}\\n\\n"
        except:
            continue
    return schema_str.strip()

def format_schema_as_text(db) -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"]
        output = ""
        for table in tables:
            df = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
            output += f"📄 **{table}**\\n"
            for i, row in df.iterrows():
                output += f"- `{row['name']}` ({row['type']})\\n"
            output += "\\n"
        return output
    except:
        return ""

def add_file_to_db(uploaded_file):
    file_name = uploaded_file.name
    table_name = (
        os.path.splitext(file_name)[0]
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )

    try:
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
        st.success(f"✅ '{file_name}' added as table '{table_name}'")
    except Exception as e:
        st.error(f"❌ Failed to add file: {e}")

# Audio recorder setup
class AudioProcessor:
    def __init__(self):
        self.audio = b""
    def recv(self, frame):
        self.audio += frame.to_ndarray().tobytes()
        return av.AudioFrame.from_ndarray(frame.to_ndarray(), layout="mono")

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript.get("text", "")

with st.expander("📂 Upload CSV or Excel to add new tables"):
    uploaded_file = st.file_uploader("Upload .csv or .xlsx", type=["csv", "xlsx"])
    if uploaded_file:
        add_file_to_db(uploaded_file)

# Database + schema
db = connect_to_db()
schema_text = get_all_schemas(db)
schema_display = format_schema_as_text(db)

# Sidebar schema + history
with st.sidebar:
    st.header("📊 Table Preview")
    try:
        tables = db.get_usable_table_names()
        selected_table = st.selectbox("Preview table", sorted(tables))
        if selected_table:
            conn = sqlite3.connect(DB_PATH)
            preview_df = pd.read_sql_query(f"SELECT * FROM {selected_table} LIMIT 5;", conn)
            st.dataframe(preview_df)
    except Exception as e:
        st.warning(f"⚠️ Table preview failed: {e}")

    st.markdown("---")
    st.header("🕓 Query History")
    if st.session_state.query_history:
        for q in st.session_state.query_history:
            st.markdown(f"- {q}")
        if st.button("🧹 Clear History"):
            st.session_state.query_history = []
            st.experimental_rerun()
    else:
        st.info("No queries yet.")

# Schema hint
st.subheader("📚 Available Tables and Columns")
st.markdown(schema_display or "No tables yet. Upload one above ☝️")

# Voice input
st.subheader("🎤 Speak your question below")
webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    client_settings=ClientSettings(media_stream_constraints={"audio": True, "video": False}),
)

if webrtc_ctx.audio_receiver:
    audio_bytes = b"".join([frame.to_ndarray().tobytes() for frame in webrtc_ctx.audio_receiver.frames])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    try:
        transcription = transcribe_audio(tmp_path)
        st.session_state.transcribed_text = transcription
        st.success(f"🎧 Transcribed: {transcription}")
    except Exception as e:
        st.error(f"Transcription failed: {e}")

# Prompt setup
prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
You are a SQL expert. Based on the database schema and the user's question, write a correct SQLite SQL query.

Use only the tables and columns provided below.

Schema:
{schema}

User Question:
{question}

SQL Query:
"""
)

llm = ChatOpenAI(temperature=0, model_name="gpt-4")
chain = LLMChain(llm=llm, prompt=prompt)

# Question input (text or voice)
st.subheader("🔍 Ask your question")
text_input = st.text_input("Type your question or use the microphone above", value=st.session_state.transcribed_text)

if text_input:
    try:
        st.session_state.query_history.append(text_input)
        sql_query = chain.run({"schema": schema_text, "question": text_input})

        # Block destructive SQL
        forbidden = ["drop", "delete", "update", "insert", "alter", "truncate"]
        if any(f in sql_query.lower() for f in forbidden):
            st.error("❌ Unsafe SQL command detected.")
        else:
            try:
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql_query(sql_query, conn)
                st.success("✅ Query executed!")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download results as CSV", csv, "query_results.csv", "text/csv")
            except Exception:
                st.warning("⚠️ The SQL query was valid, but no data was returned.")
            st.markdown("**Generated SQL:**")
            st.code(sql_query, language="sql")
    except Exception as e:
        st.error(f"❌ Error from LLM:\n\n{e}")

