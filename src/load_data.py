import os
import shutil
import pandas as pd
import sqlite3

def setup_directories():
    """Create the required directories if they don't exist."""
    directories = ['data', 'notebooks', 'src', 'models', 'dashboard']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def ingest_data():
    """Copy CSV to data/ and load it into a SQLite database."""
    print("Starting Phase 1: Data Ingestion & SQL Storage...")
    
    # Paths
    raw_csv = "dataset.csv"
    data_dir_csv = os.path.join("data", "students.csv")
    db_path = os.path.join("data", "students.db")
    
    # 1. Copy original dataset to data/
    if os.path.exists(raw_csv):
        shutil.copy(raw_csv, data_dir_csv)
        print(f"Copied raw dataset to {data_dir_csv}")
    else:
        print(f"Error: Original raw dataset not found at {raw_csv}!")
        return
        
    # 2. Ingest CSV into SQLite database table 'students'
    # The dataset uses standard comma separator as verified
    df = pd.read_csv(data_dir_csv)
    print(f"Loaded CSV with shape: {df.shape}")
    
    # Clean column names by replacing spaces and special characters if any
    # Keep them readable
    conn = sqlite3.connect(db_path)
    df.to_sql('students', conn, if_exists='replace', index=False)
    print("Ingested dataset into SQLite table 'students' successfully.")
    
    # 3. Create the 'predictions' table to log real-time API calls
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            probabilities TEXT NOT NULL,
            inputs TEXT NOT NULL
        )
    """)
    conn.commit()
    print("Created 'predictions' table in SQLite for API logging.")
    
    # 4. Verify ingestion by running a quick SQL query
    df_query = pd.read_sql("SELECT Target, COUNT(*) as count FROM students GROUP BY Target", conn)
    print("\nVerification SQL Query Result (Target distribution):")
    print(df_query.to_string(index=False))
    
    conn.close()
    print("\nPhase 1: Ingestion complete!")

if __name__ == "__main__":
    setup_directories()
    ingest_data()
