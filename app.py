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
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="SQL Query Agent", 
    layout="wide", 
    page_icon="🔍",
    initial_sidebar_state="collapsed"
)

# =============================================
# CUSTOM CSS STYLING
# =============================================
st.markdown("""
<style>
    /* Main container styling */
    .auth-container {
        max-width: 500px;
        padding: 2rem;
        margin: 3rem auto;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    
    /* Title styling */
    .auth-title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 1.8rem;
        font-weight: 600;
        font-size: 1.8rem;
    }
    
    /* Form styling */
    .auth-form {
        display: flex;
        flex-direction: column;
        gap: 1.2rem;
    }
    
    /* Button styling */
    .auth-button {
        width: 100%;
        margin-top: 1.2rem;
        background-color: #4a6fa5 !important;
        color: white !important;
        border: none;
        padding: 0.6rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .auth-button:hover {
        background-color: #3a5a8c !important;
        transform: translateY(-1px);
    }
    
    /* Link styling */
    .auth-message {
        text-align: center;
        margin-top: 1.5rem;
        color: #7f8c8d;
        font-size: 0.95rem;
    }
    
    .auth-link {
        color: #4a6fa5;
        text-decoration: none;
        cursor: pointer;
        font-weight: 500;
    }
    
    .auth-link:hover {
        text-decoration: underline;
    }
    
    /* Error/Success messages */
    .auth-error {
        color: #e74c3c;
        text-align: center;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    
    .auth-success {
        color: #2ecc71;
        text-align: center;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    
    /* Password hint */
    .password-hint {
        font-size: 0.8rem;
        color: #7f8c8d;
        margin-top: -0.8rem;
        margin-bottom: 0.8rem;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        padding: 0.6rem !important;
        border-radius: 8px !important;
    }
    
    /* Main app header */
    .app-header {
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Logout button */
    .logout-btn {
        background-color: #e74c3c !important;
    }
    
    .logout-btn:hover {
        background-color: #c0392b !important;
    }

    /* Natural language query box */
    .query-box {
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }

    /* Query history item */
    .history-item {
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        border-radius: 8px;
        background-color: #f1f5f9;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .history-item:hover {
        background-color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# DATABASE FUNCTIONS
# =============================================
def init_db():
    """Initialize the user database"""
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            email TEXT,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create a table to track user-table associations
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            table_name TEXT,
            original_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username),
            UNIQUE(username, table_name)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize databases
init_db()

# =============================================
# SECURITY FUNCTIONS
# =============================================
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_email(email):
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None

def is_valid_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search("[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search("[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search("[0-9]", password):
        return False, "Password must contain at least one number"
    return True, ""

# =============================================
# AUTHENTICATION FUNCTIONS
# =============================================
def create_user(username, password, email=None, full_name=None):
    """Create a new user account"""
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password, email, full_name) VALUES (?, ?, ?, ?)",
            (username, hash_password(password), email, full_name)
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()

def verify_user(username, password):
    """Verify user credentials"""
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

# =============================================
# USER DATA MANAGEMENT
# =============================================
def get_user_tables(username):
    """Get all tables associated with the user"""
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    c.execute(
        "SELECT table_name, original_filename FROM user_tables WHERE username = ?",
        (username,)
    )
    results = c.fetchall()
    conn.close()
    return results

def add_user_table(username, table_name, original_filename):
    """Record a table as belonging to a specific user"""
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO user_tables (username, table_name, original_filename) VALUES (?, ?, ?)",
            (username, table_name, original_filename)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Update the entry if it already exists
        c.execute(
            "UPDATE user_tables SET original_filename = ? WHERE username = ? AND table_name = ?",
            (original_filename, username, table_name)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding user table: {e}")
        return False
    finally:
        conn.close()

def get_user_db_path(username):
    """Get the user-specific database path"""
    # Create directory if it doesn't exist
    os.makedirs("user_data", exist_ok=True)
    return f"user_data/{username}_database.db"

# =============================================
# AUTHENTICATION PAGE
# =============================================
def auth_page():
    """Render the authentication page with login/signup forms"""
    
    # Use columns to center the auth container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        # Tab selection
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown('<h2 class="auth-title">Welcome Back</h2>', unsafe_allow_html=True)
            
            with st.form("Login", clear_on_submit=True):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    remember_me = st.checkbox("Remember me")
                
                if st.form_submit_button("Login", type="primary", use_container_width=True):
                    if not username or not password:
                        st.error("Please enter both username and password")
                    elif verify_user(username, password):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            st.markdown(
                '<p class="auth-message">Don\'t have an account? <span class="auth-link" onclick="switchTab()">Sign up here</span></p>', 
                unsafe_allow_html=True
            )
        
        with tab2:
            st.markdown('<h2 class="auth-title">Create Account</h2>', unsafe_allow_html=True)
            
            with st.form("Sign Up", clear_on_submit=True):
                full_name = st.text_input("Full Name", placeholder="Enter your full name")
                email = st.text_input("Email", placeholder="Enter your email")
                username = st.text_input("Username", placeholder="Choose a username")
                password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                
                st.markdown(
                    '<p class="password-hint">Password must be at least 8 characters with uppercase, lowercase, and numbers</p>', 
                    unsafe_allow_html=True
                )
                
                if st.form_submit_button("Sign Up", type="primary", use_container_width=True):
                    if not all([full_name, email, username, password, confirm_password]):
                        st.error("Please fill in all fields")
                    elif not is_valid_email(email):
                        st.error("Please enter a valid email address")
                    else:
                        valid_pass, pass_msg = is_valid_password(password)
                        if not valid_pass:
                            st.error(pass_msg)
                        elif password != confirm_password:
                            st.error("Passwords don't match")
                        else:
                            success, message = create_user(username, password, email, full_name)
                            if success:
                                st.success(message)
                                st.session_state["authenticated"] = True
                                st.session_state["username"] = username
                                st.rerun()
                            else:
                                st.error(message)
            
            st.markdown(
                '<p class="auth-message">Already have an account? <span class="auth-link" onclick="switchTab()">Login here</span></p>', 
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # JavaScript for tab switching
        st.markdown("""
        <script>
            function switchTab() {
                const tabs = parent.document.querySelectorAll('.stTabs [role="tab"]');
                const activeTab = parent.document.querySelector('.stTabs [aria-selected="true"]');
                if (activeTab.textContent.trim().includes("Login")) {
                    tabs[1].click();
                } else {
                    tabs[0].click();
                }
            }
        </script>
        """, unsafe_allow_html=True)

# =============================================
# MAIN APPLICATION
# =============================================
def main_app():
    """Main application after authentication"""
    
    # Initialize session state
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    # Check OpenAI API key
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading environment variables: {e}")
        st.stop()

    # Get user-specific database path
    username = st.session_state["username"]
    DB_PATH = get_user_db_path(username)

    # Database connection
    @st.cache_resource
    def connect_to_db(db_path):
        try:
            return SQLDatabase.from_uri(f"sqlite:///{db_path}")
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            st.stop()

    def get_all_schemas(db) -> str:
        """Get all table schemas as formatted string"""
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
        """Format schema for display"""
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
        """Add uploaded file to database"""
        file_name = uploaded_file.name
        table_name = (
            f"{username}_{os.path.splitext(file_name)[0]}"
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
            
            # Record the table association in the user_tables
            if add_user_table(username, table_name, file_name):
                st.success(f"✅ '{file_name}' added successfully")
            else:
                st.warning("⚠️ File added but failed to record table association")
        except Exception as e:
            st.error(f"❌ Failed to add file: {e}")

    def process_question(question, db):
        """Process user question and execute SQL query"""
        schema_text = get_all_schemas(db)
        user_tables = get_user_tables(username)
        
        # Get the list of tables with their original names for better context
        table_context = "\n".join([f"Table '{original_file}' is stored as '{table_name}'" 
                                  for table_name, original_file in user_tables])
        
        # For single table scenarios, we can be extra helpful
        if len(user_tables) == 1:
            table_name, original_file = user_tables[0]
            hint = f"Note: The user has only one table from file '{original_file}'. You should query the table '{table_name}' even if the user doesn't specify it."
        else:
            hint = "Use the most relevant table(s) based on the question. If unclear which table to use, select the most appropriate one based on column names and context."
        
        prompt = PromptTemplate(
            input_variables=["schema", "question", "table_context", "hint"],
            template="""
            You are a SQL expert. Based on the database schema and the user's question, 
            write a correct SQLite SQL query. Use only the tables and columns provided.
            Your goal is to translate natural language questions to accurate SQL queries.

            Schema:
            {schema}
            
            Table Information:
            {table_context}
            
            {hint}

            User Question:
            {question}

            SQL Query:
            """
        )
        
        llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        try:
            st.session_state.query_history.append(question)
            sql_query = chain.run({
                "schema": schema_text, 
                "question": question,
                "table_context": table_context,
                "hint": hint
            })

            # Sanitize the SQL query
            forbidden = ["drop", "delete", "update", "insert", "alter", "truncate"]
            if any(f in sql_query.lower() for f in forbidden):
                st.error("❌ Unsafe SQL command detected.")
            else:
                try:
                    # Ensure query only accesses the user's tables
                    allowed_tables = [table for table, _ in user_tables]
                    
                    # Extract table names from the query (simplified approach)
                    tables_in_query = re.findall(r'from\s+([^\s,;]+)', sql_query.lower())
                    tables_in_query += re.findall(r'join\s+([^\s,;]+)', sql_query.lower())
                    
                    # Remove SQL aliases from table names
                    tables_in_query = [table.strip().split(' ')[0] for table in tables_in_query]
                    
                    # Check if query only accesses user's tables
                    if all(table.strip() in allowed_tables for table in tables_in_query):
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
                    else:
                        st.error("❌ The query attempts to access tables you don't have permission for.")
                except Exception as e:
                    st.warning(f"⚠️ SQL ran but no data was returned: {e}")
        except Exception as e:
            st.error(f"❌ Error processing your question:\n\n{e}")
            
    def get_table_description(db):
        """Get descriptive summary of user tables"""
        try:
            user_tables = get_user_tables(username)
            descriptions = []
            
            for table_name, original_file in user_tables:
                conn = sqlite3.connect(DB_PATH)
                
                # Get row count
                row_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table_name}", conn).iloc[0]['count']
                
                # Get column names and sample data
                df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn)
                columns = df.columns.tolist()
                
                descriptions.append({
                    "table_name": table_name,
                    "original_file": original_file,
                    "row_count": row_count,
                    "columns": columns,
                })
                
            return descriptions
        except Exception as e:
            st.error(f"Error getting table description: {e}")
            return []

    # =============================================
    # MAIN UI LAYOUT
    # =============================================
    
    # Sidebar with user info and logout
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['username']}")
        if st.button("🚪 Logout", type="primary", use_container_width=True, key="logout"):
            st.session_state["authenticated"] = False
            st.rerun()
    
    # Main content area
    st.markdown('<div class="app-header"><h1>🔍 Natural Language SQL Query Assistant</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("Ask Questions About Your Data")
        
        with st.expander("📤 Upload Data", expanded=True):
            uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
            if uploaded_file:
                add_file_to_db(uploaded_file)
        
        db = connect_to_db(DB_PATH)
        
        # Natural language query interface
        st.markdown('<div class="query-box">', unsafe_allow_html=True)
        st.markdown("### 💬 Ask ")
        st.markdown("")
        
        table_descriptions = get_table_description(db)
        if table_descriptions:
            if len(table_descriptions) == 1:
                table = table_descriptions[0]
                st.markdown(f"Your data from **{table['original_file']}** has columns: **{', '.join(table['columns'][:5])}**{' and more...' if len(table['columns']) > 5 else ''}")
            else:
                st.markdown(f"You have {len(table_descriptions)} datasets available.")
        
        
        
        question = st.text_area(
            "Type your question here", 
            placeholder="Type your question about the data",
            height=80
        )
        
        if st.button("🔍 Get Answer", use_container_width=True):
            if question and question.strip():
                # Check if user has any tables first
                user_tables = get_user_tables(username)
                if not user_tables:
                    st.warning("⚠️ You don't have any data uploaded yet. Please upload a CSV or Excel file first.")
                else:
                    process_question(question.strip(), db)
            else:
                st.warning("Please enter a question")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display user's tables
        table_descriptions = get_table_description(db)
        if table_descriptions:
            st.header("Your Available Data")
            for i, table in enumerate(table_descriptions):
                with st.expander(f"📄 {table['original_file']} ({table['row_count']} rows)"):
                    st.write(f"**Table name**: {table['table_name']}")
                    st.write(f"**Columns**: {', '.join(table['columns'])}")
                    
                    # Show sample data
                    conn = sqlite3.connect(DB_PATH)
                    sample_df = pd.read_sql_query(f"SELECT * FROM {table['table_name']} LIMIT 5", conn)
                    st.dataframe(sample_df, use_container_width=True)
        else:
            st.info("👆 You don't have any data yet. Please upload a file above.")

    with col2:
        st.header("Query History")
        if st.session_state.query_history:
            for i, q in enumerate(reversed(st.session_state.query_history[-10:])):
                st.markdown(f"<div class='history-item'>{len(st.session_state.query_history)-i}. {q[:50]}...</div>", unsafe_allow_html=True)
            if st.button("Clear History", use_container_width=True):
                st.session_state.query_history = []
        else:
            st.info("No queries yet.")
            
        st.markdown("---")
        
        # Help section
        st.header("Tips & Help")
        with st.expander("📝 Making Good Queries"):
            st.markdown("""
            - Be specific about what you want to know
            - Focus on the data fields you're interested in
            - For time-based queries, specify the period (e.g., "last month")
            - For comparisons, make clear what you're comparing (e.g., "compare sales by region")
            - You don't need to specify table names - the system will figure it out!
            """)
            
        with st.expander("🛠️ Supported Data Types"):
            st.markdown("""
            - CSV files (.csv)
            - Excel spreadsheets (.xlsx)
            
            More formats coming soon!
            """)

# =============================================
# AUTHENTICATION CHECK
# =============================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state.get("authenticated"):
    main_app()
else:
    auth_page()