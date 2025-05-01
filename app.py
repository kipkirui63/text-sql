import os
import sqlite3
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# App UI setup
st.set_page_config(page_title="Text-to-SQL Agent", layout="wide")
st.title("🧠 Text-to-SQL Agent")

DB_PATH = "my_database.db"

# ==========================
# 🧩 Connect / Update DB
# ==========================
def connect_to_db():
    return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

def add_file_to_db(uploaded_file):
    file_name = uploaded_file.name

    # ✅ Sanitize table name
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
        st.success(f"✅ File '{file_name}' added as table '{table_name}'")

    except Exception as e:
        st.error(f"❌ Failed to add file: {e}")

# ==========================
# 📂 File Upload Section
# ==========================
with st.expander("📂 Upload CSV or Excel to add new tables to the database"):
    uploaded_file = st.file_uploader("Upload .csv or .xlsx", type=["csv", "xlsx"])
    if uploaded_file:
        add_file_to_db(uploaded_file)

# ==========================
# 🔄 Connect to DB
# ==========================
db = connect_to_db()

# ==========================
# 📋 Table Dropdown + Preview
# ==========================
with st.sidebar:
    st.header("📊 Table Viewer")
    try:
        tables = db.get_usable_table_names()
        selected_table = st.selectbox("Select a table to preview", sorted(tables))
        if selected_table:
            conn = sqlite3.connect(DB_PATH)
            preview_df = pd.read_sql_query(f"SELECT * FROM {selected_table} LIMIT 5;", conn)
            st.markdown(f"**Preview of `{selected_table}` (first 5 rows):**")
            st.dataframe(preview_df)
    except Exception as e:
        st.warning(f"⚠️ Unable to load tables: {e}")

# ==========================
# 🤖 LLM + SQL Chain
# ==========================
llm = ChatOpenAI(temperature=0, model_name="gpt-4")
db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    verbose=True,
    return_intermediate_steps=True
)

# ==========================
# 💬 Ask a Question
# ==========================
st.subheader("🔍 Ask a question about your data")
question = st.text_input("Enter your natural language query:")

if question:
    try:
        result = db_chain(question)
        sql_query = result['intermediate_steps'][0]

        # Block dangerous SQL
        forbidden = ["drop", "delete", "update", "insert", "alter", "truncate"]
        if any(word in sql_query.lower() for word in forbidden):
            st.error("❌ Query blocked: Destructive SQL commands are not allowed.")
        else:
            try:
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql_query(sql_query, conn)
                st.success("✅ Query executed successfully!")
                st.dataframe(df)
            except Exception as query_error:
                st.markdown("**Raw Result:**")
                st.markdown(result["result"])

            st.markdown("---")
            st.markdown("**Generated SQL:**")
            st.code(sql_query, language="sql")

    except Exception as e:
        st.error(f"❌ Error while processing your query:\n\n{e}")
