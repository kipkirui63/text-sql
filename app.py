import os
import sqlite3
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# Setup
st.set_page_config(page_title="Text-to-SQL Agent", layout="wide")
st.title("🧠 Text-to-SQL Agent with Full Features")

DB_PATH = "my_database.db"
if "query_history" not in st.session_state:
    st.session_state.query_history = []

def connect_to_db():
    return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

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

# File upload
with st.expander("📂 Upload CSV or Excel to add new tables"):
    uploaded_file = st.file_uploader("Upload .csv or .xlsx", type=["csv", "xlsx"])
    if uploaded_file:
        add_file_to_db(uploaded_file)

# Connect to DB
db = connect_to_db()

# ========================
# Sidebar: Table Preview, Schema, History
# ========================
with st.sidebar:
    st.header("📊 Table Preview")
    try:
        tables = db.get_usable_table_names()
        selected_table = st.selectbox("Select a table", sorted(tables))
        if selected_table:
            conn = sqlite3.connect(DB_PATH)
            preview_df = pd.read_sql_query(f"SELECT * FROM {selected_table} LIMIT 5;", conn)
            st.markdown(f"**Preview of `{selected_table}` (first 5 rows):**")
            st.dataframe(preview_df)

            # Schema viewer
            st.markdown(f"**Schema of `{selected_table}`:**")
            cursor = conn.execute(f"PRAGMA table_info({selected_table});")
            schema = pd.DataFrame(cursor.fetchall(), columns=["cid", "name", "type", "notnull", "default", "pk"])
            st.dataframe(schema[["name", "type"]])
    except Exception as e:
        st.warning(f"⚠️ Error loading tables: {e}")

    # Query history
    st.markdown("---")
    st.header("🕓 Query History")
    if st.session_state.query_history:
        for q in st.session_state.query_history:
            st.markdown(f"- {q}")
        if st.button("🧹 Clear History"):
            st.session_state.query_history = []
            st.experimental_rerun()
    else:
        st.write("No queries yet.")

# ========================
# LLM + SQL Chain
# ========================
llm = ChatOpenAI(temperature=0, model_name="gpt-4")
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True, return_intermediate_steps=True)

# ========================
# Natural Language Input
# ========================
st.subheader("🔍 Ask a question across all tables (joins supported)")
question = st.text_input("Your question:")

if question:
    try:
        result = db_chain(question)
        sql_query = result["intermediate_steps"][0]
        st.session_state.query_history.append(question)

        # Safety filter
        forbidden = ["drop", "delete", "update", "insert", "alter", "truncate"]
        if any(word in sql_query.lower() for word in forbidden):
            st.error("❌ Destructive SQL command detected.")
        else:
            try:
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql_query(sql_query, conn)
                st.success("✅ Query executed!")
                st.dataframe(df, use_container_width=True)

                # Download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download results as CSV", csv, "query_results.csv", "text/csv")
            except:
                st.markdown("**Raw Response:**")
                st.markdown(result["result"])

            st.markdown("---")
            st.markdown("**Generated SQL:**")
            st.code(sql_query, language="sql")

    except Exception as e:
        st.error(f"❌ Error while processing your query:\n\n{e}")
