import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.chains import SQLDatabaseChain

# Load API key from .env file
load_dotenv()

# Step 1: Connect to SQLite database
db = SQLDatabase.from_uri("sqlite:///my_database.db")

# Step 2: Set up the LLM (GPT-4 or GPT-3.5)
llm = ChatOpenAI(temperature=0, model_name="gpt-4")  # or "gpt-3.5-turbo"

# Step 3: Build the chain
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

# Step 4: Ask a question
question = input("Ask a question about your data: ")
response = db_chain.run(question)

print("\n🔍 Answer:")
print(response)
