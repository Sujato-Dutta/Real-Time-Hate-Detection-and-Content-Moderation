import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_data(df: pd.DataFrame, table_name: str):
    """
    Load DataFrame into Supabase table.
    
    Args:
        df (pd.DataFrame): Transformed dataset
        table_name (str): Supabase table name
    """
    data = df.to_dict(orient="records")

    print(f"Inserting {len(data)} rows into Supabase table {table_name}...")
    response = supabase.table(table_name).insert(data).execute()
    print("Insert complete:", response)
