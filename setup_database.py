import sqlite3
import pandas as pd
import os

def create_app_database():
    db_file = 'hospital_app.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    print("🔧 Setting up Database...")

    # 1. The Users Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT,
        age INTEGER,
        gender TEXT,
        phone_number TEXT UNIQUE,
        password_hash TEXT
    )
    ''')

    # 2. The Consultations Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS consultations (
        case_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        ai_summary TEXT,
        predicted_category TEXT,
        urgency_score TEXT,
        doctor_assigned TEXT,
        doctor_response TEXT,
        status TEXT DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    )
    ''')

    # 3. The Doctors Table (NEW!) 👨‍⚕️
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS doctors (
        doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        specialty TEXT
    )
    ''')

    # 4. Populate Doctors from CSV (NEW!)
    # We check if the table is empty first to avoid duplicates
    cursor.execute("SELECT count(*) FROM doctors")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("📥 Importing doctors from CSV...")
        try:
            if os.path.exists('doctors.csv'):
                df = pd.read_csv('doctors.csv')
                # Insert rows
                for index, row in df.iterrows():
                    cursor.execute("INSERT INTO doctors (name, specialty) VALUES (?, ?)", 
                                   (row['name'], row['specialty']))
                conn.commit()
                print(f"✅ Added {len(df)} doctors to the database.")
            else:
                print("⚠️ Warning: 'doctors.csv' not found. Doctors table is empty.")
        except Exception as e:
            print(f"❌ Error importing doctors: {e}")
    else:
        print("✅ Doctors table already has data.")

    print("✅ Database System Ready!")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_app_database()