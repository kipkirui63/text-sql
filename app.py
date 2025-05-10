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
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="Natural SQL Query Assistant", layout="wide", page_icon="🔍")

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
    st.title("💬 Natural SQL Query Assistant")

    # Initialize session state
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
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

        # Create upload directory if not exists
        upload_dir = os.path.join(os.getcwd(), "uploaded_files")
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        # Save the original file
        file_path = os.path.join(upload_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Reset the file pointer to the beginning for processing
        uploaded_file.seek(0)
        
        try:
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.warning("⚠️ Only .csv or .xlsx files are supported.")
                return

            # Store file metadata in a special table
            conn = sqlite3.connect(DB_PATH)
            
            # Create file_metadata table if it doesn't exist
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_filename TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert file metadata
            file_size = os.path.getsize(file_path)
            c.execute(
                "INSERT INTO file_metadata (original_filename, table_name, file_path, file_size) VALUES (?, ?, ?, ?)",
                (file_name, table_name, file_path, file_size)
            )
            
            # Import the data into a table
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            
            conn.commit()
            conn.close()
            st.success(f"✅ '{file_name}' saved and added as table '{table_name}'")
        except Exception as e:
            st.error(f"❌ Failed to add file: {e}")

    def get_sample_data(table_name, limit=5):
        """Get sample data from a table"""
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit};", conn)
        conn.close()
        return df
    
    def get_table_info(db):
        """Get detailed information about all tables"""
        table_info = {}
        for table in db.get_usable_table_names():
            conn = sqlite3.connect(DB_PATH)
            # Get schema
            schema = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
            # Get row count
            row_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)["count"][0]
            # Get sample data
            sample = get_sample_data(table)
            
            table_info[table] = {
                "schema": schema,
                "row_count": row_count,
                "sample": sample
            }
            conn.close()
        return table_info

    def process_question(question, db, with_context=True):
        """Process the user question and execute SQL query with context awareness"""
        schema_text = get_all_schemas(db)
        
        # Get additional metadata about available files
        file_metadata = ""
        try:
            conn = sqlite3.connect(DB_PATH)
            # Check if file_metadata table exists
            check_table = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='file_metadata';", 
                conn
            )
            
            if not check_table.empty:  # Table exists
                files = pd.read_sql_query(
                    "SELECT original_filename, table_name FROM file_metadata", 
                    conn
                )
                
                if not files.empty:
                    file_metadata = "File Information:\n"
                    for _, row in files.iterrows():
                        file_metadata += f"- Original file '{row['original_filename']}' is available as table '{row['table_name']}'\n"
            conn.close()
        except Exception:
            pass  # Silently handle if there's an issue with file metadata
        
        # Template for SQL generation
        sql_template = """
        You are a SQL expert. Based on the database schema and the user's question, 
        write a correct SQLite SQL query. Use only the tables and columns provided.
        
        Schema:
        {schema}
        
        {file_metadata}
        
        {context}
        
        User Question:
        {question}
        
        SQL Query (ONLY return the SQL query without any explanation or markdown):
        """
        
        # Add context from conversation history if available
        context = ""
        if with_context and st.session_state.chat_history:
            context = "Previous conversation context:\n"
            # Include only the last 5 exchanges for context
            for i in range(max(0, len(st.session_state.chat_history) - 5), len(st.session_state.chat_history)):
                context += f"Q: {st.session_state.chat_history[i][0]}\n"
                if st.session_state.chat_history[i][1]:
                    context += f"Response: {st.session_state.chat_history[i][1]}\n"
        
        prompt = PromptTemplate(
            input_variables=["schema", "file_metadata", "context", "question"],
            template=sql_template
        )
        
        llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        try:
            st.session_state.query_history.append(question)
            sql_query = chain.run({"schema": schema_text, "file_metadata": file_metadata, "context": context, "question": question})
            
            # Clean up the SQL query (in case the LLM includes explanations or formatting)
            sql_query = sql_query.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query.split("```sql")[1]
            if sql_query.endswith("```"):
                sql_query = sql_query.split("```")[0]
            sql_query = sql_query.strip()
            
            forbidden = ["drop", "delete", "update", "insert", "alter", "truncate"]
            if any(f in sql_query.lower() for f in forbidden):
                st.error("❌ Unsafe SQL command detected.")
                return None, None
            else:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    df = pd.read_sql_query(sql_query, conn)
                    
                    # Generate a natural language explanation of the result
                    explanation_template = """
                    Based on the following:
                    
                    USER QUESTION: {question}
                    SQL QUERY: {sql_query}
                    QUERY RESULTS: {results}
                    
                    Provide a natural, conversational response explaining the results. 
                    Be concise but complete. Include key insights from the data.
                    If the result set is empty, explain why this might be the case.
                    Format numbers and dates in a human-readable way where appropriate.
                    """
                    
                    results_desc = df.to_markdown() if not df.empty else "No results found."
                    explanation_prompt = PromptTemplate(
                        input_variables=["question", "sql_query", "results"],
                        template=explanation_template
                    )
                    
                    explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)
                    explanation = explanation_chain.run({
                        "question": question,
                        "sql_query": sql_query,
                        "results": results_desc
                    })
                    
                    # Add to chat history
                    if len(st.session_state.chat_history) >= 20:
                        st.session_state.chat_history.pop(0)  # Remove oldest exchange
                    st.session_state.chat_history.append((question, explanation))
                    
                    return df, sql_query, explanation
                    
                except Exception as e:
                    st.warning(f"⚠️ SQL ran but encountered an error: {e}")
                    return None, sql_query, f"I couldn't execute that query successfully. Error: {e}"
        except Exception as e:
            st.error(f"❌ Error processing your question:\n\n{e}")
            return None, None, f"I couldn't process your question. Error: {e}"

    # New Integrated UI with Chat and DB Explorer
    def integrated_ui():
        # Two-column layout
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.header("💬 Ask Me About Your Data")
            
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for i, (user_msg, ai_response) in enumerate(st.session_state.chat_history):
                    st.chat_message("user").write(user_msg)
                    if ai_response:
                        st.chat_message("assistant").write(ai_response)
            
            # Get user input
            if user_input := st.chat_input("Ask a question about your data"):
                st.chat_message("user").write(user_input)
                
                db = connect_to_db()
                
                # Show a spinner while processing
                with st.spinner("Analyzing your data..."):
                    df, sql, explanation = process_question(user_input, db)
                    
                    # Display the response
                    response_container = st.chat_message("assistant")
                    response_container.write(explanation)
                    
                    # Show the dataframe result if available
                    if df is not None and not df.empty:
                        with response_container:
                            st.dataframe(df, use_container_width=True)
                            
                            # Add download button
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📥 Download as CSV", 
                                csv, 
                                f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                                "text/csv"
                            )
                    
                    # Show the SQL query in an expander
                    if sql:
                        with response_container:
                            with st.expander("🔍 View SQL Query"):
                                st.code(sql, language="sql")
        
        with col2:
            st.header("📊 Data Explorer")
            
            with st.expander("📤 Upload Data", expanded=True):
                uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
                if uploaded_file:
                    add_file_to_db(uploaded_file)
            
            db = connect_to_db()
            schema_display = format_schema_as_text(db)
            
            with st.expander("📊 Database Schema", expanded=True):
                st.markdown(schema_display or "No tables yet. Upload one above ☝️")
            
            # Table Explorer
            try:
                tables = db.get_usable_table_names()
                if tables:
                    selected_table = st.selectbox("Select table to preview", sorted(tables))
                    if selected_table:
                        # Get table info
                        conn = sqlite3.connect(DB_PATH)
                        row_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {selected_table}", conn)["count"][0]
                        st.write(f"Total rows: {row_count}")
                        
                        # Preview table
                        preview_rows = st.slider("Number of rows to preview", 5, 50, 10)
                        sort_options = ["None"] + list(pd.read_sql_query(f"PRAGMA table_info({selected_table});", conn)["name"])
                        sort_by = st.selectbox("Sort by", sort_options, index=0)
                        
                        # Build query with optional sorting
                        preview_query = f"SELECT * FROM {selected_table}"
                        if sort_by != "None":
                            preview_query += f" ORDER BY {sort_by}"
                        preview_query += f" LIMIT {preview_rows};"
                        
                        # Execute query
                        preview_df = pd.read_sql_query(preview_query, conn)
                        st.dataframe(preview_df, use_container_width=True)
                        
                        # Example questions section
                        with st.expander("🤔 Example Questions"):
                            cols = preview_df.columns.tolist()
                            if cols:
                                st.markdown("### Try asking:")
                                questions = [
                                    f"How many records are in the {selected_table} table?",
                                    f"What is the average {cols[0]} in {selected_table}?" if len(cols) > 0 else "",
                                    f"Show me the records with the highest {cols[0]} in {selected_table}" if len(cols) > 0 else "",
                                    f"What is the distribution of {cols[0]} in {selected_table}?" if len(cols) > 0 else "",
                                ]
                                
                                for q in questions:
                                    if q:  # Only add non-empty questions
                                        st.markdown(f"- *{q}*")
                else:
                    st.info("No tables available. Please upload data first.")
            except Exception as e:
                st.error(f"Error exploring database: {e}")

    # Main UI Layout with tabs for different modes
    tab1, tab2 = st.tabs(["📊 Data Chat & Explorer", "⚙️ Settings"])
    
    with tab1:
        integrated_ui()
    
    with tab2:
        st.header("Settings")
        
        # Clear conversation history
        if st.button("Clear Conversation History"):
            st.session_state.chat_history = []
            st.session_state.query_history = []
            st.success("Conversation history cleared!")
        
        # Model settings
        st.subheader("Model Settings")
        model_options = {
            "gpt-4": "GPT-4 (Most capable, slower)",
            "gpt-3.5-turbo": "GPT-3.5 Turbo (Faster, less capable)"
        }
        selected_model = st.selectbox(
            "Select AI Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        # Save settings to session state
        if "model" not in st.session_state or st.session_state.model != selected_model:
            st.session_state.model = selected_model
            st.success(f"Model updated to {model_options[selected_model]}")
        
        # Database management section
        st.subheader("Database Management")
        
        # File management
        st.markdown("#### Uploaded Files")
        try:
            conn = sqlite3.connect(DB_PATH)
            # Check if file_metadata table exists
            check_table = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='file_metadata';", 
                conn
            )
            
            if not check_table.empty:  # Table exists
                files = pd.read_sql_query(
                    "SELECT id, original_filename, table_name, file_path, file_size, uploaded_at FROM file_metadata ORDER BY uploaded_at DESC", 
                    conn
                )
                
                if not files.empty:
                    # Format the file size
                    files['file_size'] = files['file_size'].apply(
                        lambda x: f"{x/1024:.1f} KB" if x < 1024*1024 else f"{x/(1024*1024):.1f} MB"
                    )
                    
                    st.dataframe(
                        files[['original_filename', 'table_name', 'file_size', 'uploaded_at']], 
                        use_container_width=True
                    )
                    
                    # Download original files
                    selected_file_id = st.selectbox(
                        "Select file to download:",
                        files['id'].tolist(),
                        format_func=lambda x: files.loc[files['id'] == x, 'original_filename'].iloc[0]
                    )
                    
                    if selected_file_id:
                        file_path = files.loc[files['id'] == selected_file_id, 'file_path'].iloc[0]
                        original_filename = files.loc[files['id'] == selected_file_id, 'original_filename'].iloc[0]
                        
                        if os.path.exists(file_path):
                            with open(file_path, "rb") as file:
                                file_content = file.read()
                                st.download_button(
                                    "📥 Download Original File", 
                                    file_content, 
                                    original_filename, 
                                    mime="application/octet-stream"
                                )
                        else:
                            st.error("File not found at saved location.")
                else:
                    st.info("No files uploaded yet.")
            else:
                st.info("No files uploaded yet.")
            
            conn.close()
        except Exception as e:
            st.error(f"Error retrieving file list: {e}")
        
        # Clear database section
        st.markdown("---")
        with st.expander("⚠️ Danger Zone"):
            st.warning("The following actions cannot be undone!")
            
            delete_files = st.checkbox("Also delete uploaded files from disk")
            
            if st.button("Clear Database", type="secondary"):
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    
                    # Get file paths if needed for deletion
                    file_paths = []
                    if delete_files:
                        try:
                            # Check if file_metadata table exists
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_metadata';")
                            if cursor.fetchone():
                                cursor.execute("SELECT file_path FROM file_metadata")
                                file_paths = [row[0] for row in cursor.fetchall()]
                        except:
                            pass
                    
                    # Get all tables
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    # Drop all tables
                    for table in tables:
                        if table[0] != "sqlite_sequence":  # Skip internal SQLite table
                            cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
                    
                    conn.commit()
                    conn.close()
                    
                    # Delete physical files if requested
                    if delete_files:
                        deleted_count = 0
                        for file_path in file_paths:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                deleted_count += 1
                        st.success(f"Database cleared successfully! {deleted_count} files deleted from disk.")
                    else:
                        st.success("Database cleared successfully! (Original files preserved on disk)")
                except Exception as e:
                    st.error(f"Error clearing database: {e}")

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