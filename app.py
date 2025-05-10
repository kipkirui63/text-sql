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

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="SQL Query Agent", layout="wide", page_icon="🔍")

# Initialize databases
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

# Initialize databases
init_db()

# Hash password function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Authentication functions
def create_user(username, password, email=None):
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
            (username, hash_password(password), email)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    c.execute(
        "SELECT password FROM users WHERE username = ?",
        (username,)
    )
    result = c.fetchone()
    conn.close()
    if result:
        return result[0] == hash_password(password)
    return False

# Authentication UI
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

# Main App
def main_app():
    st.title("🔍 SQL Query Agent")

    # Initialize session state
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    # Environment setup
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading environment variables: {e}")
        st.stop()

    DB_PATH = "my_database.db"

    # Database functions
    @st.cache_resource
    def connect_to_db():
        try:
            return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            st.stop()

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
        except Exception as e:
            st.error(f"Error loading schema: {e}")
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
        
        llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key)
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
        
        with st.expander("📝 Query Input", expanded=True):
            question = st.text_input(
                "Type your question here", 
                placeholder="Type your question about the data"
            )
            
            if st.button("🔍 Run Query"):
                if question and question.strip():
                    process_question(question.strip(), db)
                else:
                    st.warning("Please enter a question")

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
        else:
            st.info("No queries yet.")

    # Add logout button
    st.sidebar.markdown(f"Logged in as: **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

# Check authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main_app()
else:
    auth_page()