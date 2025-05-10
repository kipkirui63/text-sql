import os
import sqlite3
import pandas as pd
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.utilities.sql_database import SQLDatabase
from datetime import datetime
import hashlib
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="Natural SQL Query Assistant", layout="wide", page_icon="🔍")

# Initialize database for user auth
def init_db():
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            email TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, email=None):
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", (username, hash_password(password), email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0] == hash_password(password)
    return False

def auth_page():
    st.title("🔐 Authentication")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        with st.form("Login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if verify_user(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    with tab2:
        with st.form("Sign Up"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email (optional)")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Sign Up"):
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    if create_user(new_username, new_password, new_email):
                        st.success("Account created successfully! Please login.")
                    else:
                        st.error("Username already exists")

def main_app():
    DB_PATH = "my_database.db"

    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading environment variables: {e}")
        st.stop()

    @st.cache_resource
    def connect_to_db():
        try:
            return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            st.stop()

    def get_all_schemas(db):
        schema_str = ""
        for table in db.get_usable_table_names():
            try:
                columns = db.run(f"PRAGMA table_info({table})")
                col_lines = [f"{col['name']} ({col['type']})" for col in columns]
                schema_str += f"Table: {table}\nColumns: {', '.join(col_lines)}\n\n"
            except:
                continue
        return schema_str.strip()

    def format_schema_as_text(db):
        try:
            conn = sqlite3.connect(DB_PATH)
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"]
            output = ""
            for table in tables:
                df = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
                output += f"📄 **{table}**\n"
                for _, row in df.iterrows():
                    output += f"- `{row['name']}` ({row['type']})\n"
                output += "\n"
            return output
        except Exception as e:
            st.error(f"Error loading schema: {e}")
            return ""

    def add_file_to_db(uploaded_file):
        file_name = uploaded_file.name
        table_name = os.path.splitext(file_name)[0].replace(" ", "_").replace("-", "_").replace(".", "_").lower()
        try:
            df = pd.read_csv(uploaded_file) if file_name.endswith(".csv") else pd.read_excel(uploaded_file)
            conn = sqlite3.connect(DB_PATH)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.close()
            st.success(f"✅ '{file_name}' added as table '{table_name}'")
        except Exception as e:
            st.error(f"❌ Failed to add file: {e}")

    def process_question(question, db):
        schema_text = get_all_schemas(db)
        sql_template = """
        You are a SQL expert. Given the schema and user question, write a valid SQLite query.

        Schema:
        {schema}

        User Question:
        {question}

        SQL Query:
        """
        prompt = PromptTemplate(input_variables=["schema", "question"], template=sql_template)
        llm = ChatOpenAI(temperature=0, model_name=st.session_state.get("model", "gpt-4"), openai_api_key=openai_api_key)
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            st.session_state.query_history.append(question)
            sql_query = chain.run({"schema": schema_text, "question": question}).strip()
            sql_query = sql_query.split("```sql")[-1].split("```")[-1].strip()
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return df, sql_query
        except Exception as e:
            return None, f"Error: {e}"

    # --- Layout ---
    tab1, tab2 = st.tabs(["Dashboard", "⚙️ Settings"])

    with tab1:
        st.header("💬 Natural Language SQL Query")
        user_input = st.text_input("Ask a question about your data")
        if user_input:
            db = connect_to_db()
            with st.spinner("Analyzing your question..."):
                df, sql_or_error = process_question(user_input, db)
                if df is not None:
                    st.write(df)
                    st.download_button("Download CSV", df.to_csv(index=False), "query_results.csv")
                    with st.expander("Show SQL Query"):
                        st.code(sql_or_error, language="sql")
                else:
                    st.error(sql_or_error)

        st.divider()
        st.subheader("📊 Database Explorer")
        with st.expander("Upload CSV or Excel", expanded=True):
            uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])
            if uploaded_file:
                add_file_to_db(uploaded_file)

        db = connect_to_db()
        schema_display = format_schema_as_text(db)
        st.markdown(schema_display or "No tables found. Upload data to begin.")

    with tab2:
        st.header("Settings")
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.session_state.query_history = []
            st.success("Cleared!")
        selected_model = st.selectbox("Choose Model", ["gpt-4", "gpt-3.5-turbo"], index=0)
        st.session_state.model = selected_model

    st.sidebar.markdown(f"Logged in as: **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if st.session_state["authenticated"]:
    main_app()
else:
    auth_page()
