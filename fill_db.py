"""
Fill Database
=============
Populates the SQLite database with labeled articles.
Creates the database and table if they don't exist, then inserts articles.

Usage:
    python fill_db.py                    # Use BERT labeled data (default)
    python fill_db.py --model mdeberta   # Use mDeBERTa labeled data
    python fill_db.py --model umberto    # Use UmBERTo labeled data
    python fill_db.py --input custom.json --output custom.db
"""

import json
import sqlite3
import os
import argparse

from models.config import LABELS, get_model_config, get_available_models


def create_database(db_path: str):
    """Create the database and articles table.

    :param db_path: str: path to the database file

    """
    # Create directory if needed
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Build crime columns (value + prob for each)
    crime_columns = []
    for label in LABELS:
        crime_columns.append(f"{label} INTEGER DEFAULT 0")
        crime_columns.append(f"{label}_prob REAL DEFAULT 0.0")
    
    crime_columns_sql = ",\n        ".join(crime_columns)
    
    # Create table
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        link TEXT,
        quartiere TEXT,
        title TEXT,
        date TEXT,
        content TEXT,
        streets TEXT,
        sources TEXT,
        model TEXT,
        {crime_columns_sql},
        UNIQUE(link, model)
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database created: {db_path}")


def fill_database(json_path: str, db_path: str, model_name: str, clear_existing: bool = True):
    """Populate the database with data from JSON file.

    :param json_path: str: path to the JSON file
    :param db_path: str: path to the database file
    :param model_name: str: name of the model
    :param clear_existing: bool: whether to clear the existing data (Default value = True)

    """
    
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles from {json_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing table if requested
    if clear_existing:
        cursor.execute("DELETE FROM articles")
        print("Cleared existing data from table")
    
    # Build insert query
    columns = ['link', 'quartiere', 'title', 'date', 'content', 'streets', 'sources', 'model']
    for label in LABELS:
        columns.extend([label, f"{label}_prob"])
    
    placeholders = ', '.join(['?' for _ in columns])
    columns_sql = ', '.join(columns)
    
    insert_query = f"INSERT OR IGNORE INTO articles ({columns_sql}) VALUES ({placeholders})"
    
    # Insert articles
    inserted = 0
    errors = 0
    
    for item in articles:
        try:
            # Extract values
            values = [
                item.get('link', ''),
                item.get('python_id', item.get('quartiere', '')),
                item.get('title', ''),
                item.get('date', ''),
                item.get('content', ''),
                json.dumps(item.get('streets', []), ensure_ascii=False),
                json.dumps(item.get('sources', []), ensure_ascii=False),
                model_name,
            ]
            
            # Add crime values
            for label in LABELS:
                crime_data = item.get(label, {})
                if isinstance(crime_data, dict):
                    values.append(crime_data.get('value', 0))
                    values.append(crime_data.get('prob', 0.0))
                else:
                    # Old format (only 0/1)
                    values.append(int(crime_data) if crime_data else 0)
                    values.append(0.0)
            
            cursor.execute(insert_query, values)
            inserted += 1
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Error in article '{item.get('title', 'N/A')[:30]}': {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\nInserted {inserted} articles into database")
    if errors > 0:
        print(f"{errors} errors during insertion")


def main():
    """ Main function """
    parser = argparse.ArgumentParser(
        description="Populate SQLite database with labeled articles",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', '-m', type=str, default='bert',
                        choices=get_available_models(),
                        help='Model whose labeled data to use (default: bert)')
    parser.add_argument('--all', action='store_true',
                        help='Load data from all available models')
    parser.add_argument('--input', '-i', type=str,
                        help='Custom input JSON file (overrides --model)')
    parser.add_argument('--output', '-o', type=str, default='database.db',
                        help='Output database file (default: database.db)')
    parser.add_argument('--append', '-a', action='store_true',
                        help='Append to existing data instead of clearing')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FILL DATABASE")
    print("="*60)
    
    # Create database (if not exists)
    create_database(args.output)
    
    if args.all:
        # Load all models
        models = get_available_models()
        print(f"Loading data from all models: {models}")
        
        first = True
        total_inserted = 0
        
        for model_name in models:
            config = get_model_config(model_name)
            json_path = f"models/{config['output_dir']}/dataset.json"
            
            if not os.path.exists(json_path):
                print(f"\nSkipping {model_name}: File not found ({json_path})")
                continue
            
            print(f"\n--- Loading {config['name']} ---")
            # Only clear on first model
            fill_database(json_path, args.output, model_name=model_name, 
                         clear_existing=first and not args.append)
            first = False
        
    elif args.input:
        # Custom input file
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        
        if not os.path.exists(args.input):
            print(f"\nFile not found: {args.input}")
            return
        
        fill_database(args.input, args.output, model_name="custom", 
                     clear_existing=not args.append)
    else:
        # Single model
        config = get_model_config(args.model)
        json_path = f"models/{config['output_dir']}/dataset.json"
        
        print(f"Input:  {json_path}")
        print(f"Output: {args.output}")
        
        if not os.path.exists(json_path):
            print(f"\nFile not found: {json_path}")
            print("Run first: python label_quartieri.py --model " + args.model)
            return
        
        fill_database(json_path, args.output, model_name=args.model, 
                     clear_existing=not args.append)
    
    print("\n" + "="*60)
    print("DATABASE POPULATED!")
    print("="*60)
    print(f"\nFile: {args.output}")


if __name__ == "__main__":
    main()