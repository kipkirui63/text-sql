import os
import sqlite3
import tempfile
import pandas as pd
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.utilities.sql_database import SQLDatabase
import openai
from datetime import datetime
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import queue
import subprocess

# MUST BE FIRST - Streamlit requirement
st.set_page_config(page_title="Voice SQL Agent", layout="wide", page_icon="🗣️")
st.title("🗣️ Voice-to-SQL Agent")

# Check for ffmpeg (without sudo)
try:
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    st.session_state.ffmpeg_available = True
except:
    st.warning("""
    FFmpeg is required for audio processing but isn't installed. 
    For Linux, run: `apt-get install ffmpeg`
    For Mac, run: `brew install ffmpeg`
    """)
    st.session_state.ffmpeg_available = False

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "recording" not in st.session_state:
    st.session_state.recording = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = queue.Queue()

# Environment setup
openai.api_key = os.getenv("OPENAI_API_KEY")
DB_PATH = "my_database.db"

# Custom Audio Processor
class AudioRecorder(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        if st.session_state.recording:
            st.session_state.audio_buffer.put(frame.to_ndarray())
        return frame

def save_audio_to_tempfile(audio_frames):
    """Save audio frames to a temporary WAV file"""
    try:
        # Convert frames to single numpy array
        audio_array = np.concatenate(audio_frames)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Simple WAV file header (44 bytes) + audio data
            tmp_file.write(b'RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00\x00\x77\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            tmp_file.write(audio_array.tobytes())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return None

def transcribe_audio(file_path):
    """Transcribe audio using Whisper API"""
    try:
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            return transcript.get("text", "")
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return ""

# Database functions
def connect_to_db():
    return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

def get_all_schemas(db) -> str:
    schema_str = ""
    for table in db.get_usable_table_names():
        try:
            columns = db.run(f"PRAGMA table_info({table})")
            col_lines = [f"{col['name']} ({col['type']})" for col in columns]
            schema_str += f"Table: {table}\nColumns: {', '.join(col_lines)}\n\n"
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
            output += f"📄 **{table}**\n"
            for i, row in df.iterrows():
                output += f"- `{row['name']}` ({row['type']})\n"
            output += "\n"
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
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to add file: {e}")

def process_question(question, db):
    """Process the user question and execute SQL query"""
    schema_text = get_all_schemas(db)
    
    prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template="""
        You are a SQL expert. Based on the database schema and the user's question, 
        write a correct SQLite SQL query. Use only the tables and columns provided.

        Schema:
        {schema}

        User Question:
        {question}

        SQL Query:
        """
    )
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        st.session_state.query_history.append(question)
        sql_query = chain.run({"schema": schema_text, "question": question})

        forbidden = ["drop", "delete", "update", "insert", "alter", "truncate"]
        if any(f in sql_query.lower() for f in forbidden):
            st.error("❌ Unsafe SQL command detected.")
        else:
            try:
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql_query(sql_query, conn)
                
                st.success("✅ Query executed successfully!")
                st.dataframe(df, use_container_width=True, height=400)
                
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download as CSV", 
                    csv, 
                    f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                    "text/csv"
                )
                
                with st.expander("🔍 View Generated SQL"):
                    st.code(sql_query, language="sql")
            except Exception as e:
                st.warning(f"⚠️ SQL ran but no data was returned: {e}")
    except Exception as e:
        st.error(f"❌ Error processing your question:\n\n{e}")

# Main UI Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Database Interaction")
    
    with st.expander("📤 Upload Data", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file:
            add_file_to_db(uploaded_file)
    
    db = connect_to_db()
    schema_display = format_schema_as_text(db)
    
    with st.expander("📊 Database Schema", expanded=True):
        st.markdown(schema_display or "No tables yet. Upload one above ☝️")
    
    if st.session_state.ffmpeg_available:
        with st.expander("🎤 Voice Input", expanded=True):
            st.markdown("""
            **How to use:**
            1. Click 'Start Recording' below
            2. Ask your question clearly (speak close to microphone)
            3. Click 'Stop Recording' when done
            4. Wait for transcription to appear
            """)
            
            # Recording controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎤 Start Recording", disabled=st.session_state.recording or st.session_state.processing):
                    st.session_state.recording = True
                    st.session_state.audio_buffer = queue.Queue()
                    st.session_state.transcribed_text = ""
                    st.success("Recording started... Speak now!")
            with col2:
                if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording or st.session_state.processing):
                    st.session_state.recording = False
                    st.session_state.processing = True
                    st.info("Processing your recording...")
                    
                    # Get all audio frames
                    audio_frames = []
                    while not st.session_state.audio_buffer.empty():
                        audio_frames.append(st.session_state.audio_buffer.get())
                    
                    if audio_frames:
                        # Save to temporary file
                        audio_file = save_audio_to_tempfile(audio_frames)
                        if audio_file:
                            # Transcribe
                            transcribed = transcribe_audio(audio_file)
                            os.unlink(audio_file)  # Clean up
                            
                            if transcribed:
                                st.session_state.transcribed_text = transcribed
                                st.success(f"🔊 Transcribed: {transcribed}")
                            else:
                                st.warning("No speech detected in recording")
                    else:
                        st.warning("No audio was recorded")
                    
                    st.session_state.processing = False
            
            # Audio recorder
            webrtc_ctx = webrtc_streamer(
                key="voice-input",
                mode=WebRtcMode.SENDONLY,
                audio_processor_factory=AudioRecorder,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"audio": True, "video": False},
            )
    else:
        st.warning("Voice input disabled - FFmpeg not installed")
    
    # Text input with voice transcription
    question = st.text_input(
        "Enter your question", 
        value=st.session_state.transcribed_text,
        placeholder="Type or speak your question about the data"
    )
    
    if st.button("🔍 Run Query", disabled=st.session_state.processing) and question:
        process_question(question, db)

with col2:
    st.header("Database Preview")
    try:
        tables = db.get_usable_table_names()
        selected_table = st.selectbox("Select table to preview", sorted(tables))
        if selected_table:
            conn = sqlite3.connect(DB_PATH)
            preview_df = pd.read_sql_query(f"SELECT * FROM {selected_table} LIMIT 10;", conn)
            st.dataframe(preview_df, height=300)
    except Exception as e:
        st.warning(f"⚠️ Table preview failed: {e}")
    
    st.markdown("---")
    st.header("🕒 Query History")
    if st.session_state.query_history:
        for i, q in enumerate(reversed(st.session_state.query_history[-5:])):
            st.markdown(f"{len(st.session_state.query_history)-i}. {q[:50]}...")
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.rerun()
    else:
        st.info("No queries yet.")