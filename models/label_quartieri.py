"""
Label Quartieri
=================================
Single script to label all quartieri using any model (BERT, mDeBERTa, UmBERTo).

Usage:
    python label_quartieri.py --model bert
    python label_quartieri.py --model mdeberta --skip-labeling   # skips labeling phase and only merges + deduplicates
    python label_quartieri.py --model umberto
"""

import json
import os
import argparse

from config import QUARTIERI, LABELS, get_model_config, get_available_models
from inference import label_articles_from_file, load_model


def label_all_quartieri(model_name: str = 'bert'):
    """Label all quartieri with the specified model

    :param model_name: str: model to use (Default value = 'bert')

    """
    config = get_model_config(model_name)
    output_dir = config['output_dir']
    
    
    print(f"LABELING QUARTIERI ({config['name']})")
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading model {config['name']}...")
    model, tokenizer, device, _ = load_model(model_name)
    print(f"Model loaded on: {device}\n")
    
    total = len(QUARTIERI)
    for i, quartiere in enumerate(QUARTIERI, 1):
        input_file = f"data/{quartiere}.json"
        output_file = f"{output_dir}/labeled_{quartiere}.json"
        
        if not os.path.exists(input_file):
            print(f"[{i}/{total}] File not found: {input_file}")
            continue
        
        print(f"[{i}/{total}] Processing: {quartiere}")
        
        try:
            # Load articles
            with open(input_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            # Label each article
            from inference import label_article
            for article in articles:
                article["python_id"] = quartiere
                label_article(article, model, tokenizer, device, aggregation="max")
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
            
            print(f"Saved ({len(articles)} news)")
        except Exception as e:
            print(f"Error: {e}")
    
    return output_dir


def merge_all_files(output_dir: str):
    """Unifies all labeled_*.json files.

    :param output_dir: str: directory where the labeled files are stored

    """
    output_file = f"{output_dir}/merged_file.json"
    
    
    print("MERGING FILES")
    
    
    json_files = sorted([f for f in os.listdir(output_dir) 
                        if f.startswith("labeled_") and f.endswith('.json')])
    
    merged_data = []
    total_articles = 0
    
    for filename in json_files:
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data)
            total_articles += len(data)
        print(f"{filename}: {len(data)} news")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Merged: {total_articles} total news → {output_file}")
    return output_file


def remove_duplicates(input_file: str, output_dir: str):
    """Removes duplicates and combines labels.

    :param input_file: str: file to remove duplicates from
    :param output_dir: str: directory where the output file will be stored

    """
    output_file = f"{output_dir}/dataset.json"
    
    
    print("REMOVING DUPLICATES")
    
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    # Group by title, combine labels
    unique_items = {}
    
    for item in data:
        key = item.get("title", "")
        
        if key not in unique_items:
            unique_items[key] = item
        else:
            # Combine labels (OR logic)
            for category in LABELS:
                if category in item:
                    cat_data = item[category]
                    existing = unique_items[key].get(category, {})
                    
                    # If dict, then combine
                    if isinstance(cat_data, dict) and isinstance(existing, dict):
                        if cat_data.get("value", 0) == 1:
                            unique_items[key][category] = cat_data
                    # If int, then OR
                    elif cat_data == 1:
                        unique_items[key][category] = 1
    
    deduplicated = list(unique_items.values())
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)
    
    removed = original_count - len(deduplicated)
    print(f"Original news: {original_count}")
    print(f"Duplicates removed:  {removed}")
    print(f"Final news:    {len(deduplicated)}")
    print(f"\nFinal dataset: {output_file}")
    
    return output_file


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Unified pipeline to label all quartieri",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', '-m', type=str, default='bert',
                        choices=get_available_models(),
                        help='Model to use (default: bert)')
    parser.add_argument('--skip-labeling', action='store_true',
                        help='Skips labeling phase (only merge + deduplicate)')
    
    args = parser.parse_args()
    
    config = get_model_config(args.model)
    output_dir = config['output_dir']
    
    # Labeling
    if not args.skip_labeling:
        output_dir = label_all_quartieri(args.model)
    else:
        print("⏭️  Skipping labeling phase...")
        os.makedirs(output_dir, exist_ok=True)
    
    # Merge
    merged_file = merge_all_files(output_dir)
    
    # Deduplicate
    final_file = remove_duplicates(merged_file, output_dir)
    
    
    print(f"DONE! ({config['name']})")
    
    print(f"\nFinal output: {final_file}")


if __name__ == "__main__":
    main()
