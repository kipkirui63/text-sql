import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv



load_dotenv()


# Step 1: Define your folder and database name
folder_path = "./data"  # Replace with your actual path if different
db_name = "my_database.db"

# Step 2: Create a SQLite engine
engine = create_engine(f"sqlite:///{db_name}")

# Step 3: Loop through each file in the folder
for file in os.listdir(folder_path):
    if file.endswith(".csv") or file.endswith(".xlsx"):
        file_path = os.path.join(folder_path, file)
        table_name = os.path.splitext(file)[0]  # e.g., customers.csv → customers

        if file.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        print(f"✅ Imported {file} into table '{table_name}'")

print(f"\n🎉 All files are now inside {db_name}")



