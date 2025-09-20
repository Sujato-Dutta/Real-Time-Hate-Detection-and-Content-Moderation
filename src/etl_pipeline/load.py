import pandas as pd

def load_data(df: pd.DataFrame, table_name: str):
    """
    Load DataFrame into Supabase table.
    
    Args:
        df (pd.DataFrame): Transformed dataset
        table_name (str): Supabase table name
    """
    # Build client from config.yaml (consistent with data_ingestion)
    from supabase import Client
    from src.data_ingestion import get_supabase_client

    supabase: Client = get_supabase_client()

    # Ensure JSON-safe payload (convert NaN -> None)
    df = df.astype(object).where(pd.notna(df), None)
    data = df.to_dict(orient="records")

    print(f"Inserting {len(data)} rows into Supabase table {table_name}...")
    # Use UPSERT to avoid duplicate key violations on Id
    response = supabase.table(table_name).upsert(data, on_conflict="Id").execute()
    print("Upsert complete:", response)
