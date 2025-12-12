# Crime Classification for Bari News Articles

A multi-label classification system for Italian news articles, specifically designed to detect and categorize crime-related content from the Bari metropolitan area.

## Project Structure

```
├── Dataset generation/          # Synthetic dataset generation using LLMs
│   ├── generate_dataset.py      # Generation with Gemini Flash
│   └── generate_dataset_gemma.py # Generation with Gemma
│
├── scraper/                     # News scraping from various sources
│   └── scraper_*.py             # Web scraper for website *
│
├── models/                      # ML models for crime classification
│   ├── config.py                # Shared configuration
│   ├── train.py                 # Unified training script
│   ├── evaluate.py              # Model evaluation
│   ├── inference.py             # Inference with text chunking
│   ├── label_quartieri.py       # Pipeline to label all neighborhoods
│   ├── compare_models.py        # Model comparison
│   └── extract_streets.py       # Street name extraction
│
├── datasets/                    # Generated and realistic datasets
│
├── BERTMAN/                     # Web application
│   ├── frontend/                # React frontend
│   └── backend/                 # FastAPI backend
│
├── fill_db.py                   # Database population
│
└── stradario_bari.xlsx          # Street database for Bari
```

## Crime Categories

The system classifies articles into 13 crime categories:

| Category                       | Description              |
| ------------------------------ | ------------------------ |
| `omicidio`                     | Voluntary homicide       |
| `omicidio_colposo`             | Involuntary manslaughter |
| `omicidio_stradale`            | Road homicide            |
| `tentato_omicidio`             | Attempted murder         |
| `furto`                        | Theft                    |
| `rapina`                       | Robbery                  |
| `violenza_sessuale`            | Sexual violence          |
| `aggressione`                  | Assault                  |
| `spaccio`                      | Drug dealing             |
| `truffa`                       | Fraud                    |
| `estorsione`                   | Extortion                |
| `contrabbando`                 | Smuggling                |
| `associazione_di_tipo_mafioso` | Mafia association        |

## Models

> **Language Note**: This project is designed for **Italian text**. All transformer models (BERT, mDeBERTa, UmBERTo) and the sentence-transformers model used for deduplication (`paraphrase-multilingual-MiniLM-L12-v2`) are either Italian-specific or multilingual with Italian support.

Three transformer models are supported:

| Model      | Base                                      | Description          |
| ---------- | ----------------------------------------- | -------------------- |
| `bert`     | `dbmdz/bert-base-italian-cased`           | Italian BERT         |
| `mdeberta` | `microsoft/mdeberta-v3-base`              | Multilingual DeBERTa |
| `umberto`  | `Musixmatch/umberto-commoncrawl-cased-v1` | Italian RoBERTa      |

## Usage

### Training

```bash
cd models

# Train BERT (default)
python train.py --model bert --epochs 5 --batch_size 24

# Train mDeBERTa
python train.py --model mdeberta --epochs 5 --batch_size 16

# Train UmBERTo
python train.py --model umberto --epochs 5
```

### Evaluation

```bash
# Evaluate model on test set
python evaluate.py --model bert --dataset dataset.json

# Results are saved to evaluation_results_<model>/
```

### Inference

```bash
# Test inference with sample text
python inference.py --model bert --test

# Label articles from a JSON file
python inference.py --model bert --input data/articles.json --output data/labeled.json
```

### Label All Neighborhoods

```bash
# Full pipeline: label + merge + deduplicate
python label_quartieri.py --model bert

# Skip labeling (only merge and deduplicate)
python label_quartieri.py --model bert --skip-labeling
```

### Compare Models

```bash
python compare_models.py
```

## Dataset Generation

Generate synthetic training data using LLMs:

```bash
cd "datasets_generation"

# Using Gemini 2.5 Flash (note that, as of 09/12/2025, Google slashed the free tier RPD to just 20 calls per day)
python generate_dataset.py --type crime
python generate_dataset.py --type non_crime
python generate_dataset.py --type ambiguous
python generate_dataset.py --type all

# Using Gemma (as of 09/12/2025 has higher limits)
python generate_dataset_gemma.py --type crime
python generate_dataset_gemma.py --type all --batches_crime 60 --batches_non_crime 700
```

### Multi-Label Support

The generators produce **~20% multi-label articles** with realistic crime combinations:

| Category                  | Combinations                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Violence + Aggression** | `rapina + aggressione`, `furto + aggressione`, `tentato_omicidio + aggressione`, `violenza_sessuale + aggressione` |
| **Mafia-related**         | `spaccio + mafia`, `estorsione + mafia`, `omicidio + mafia`, `contrabbando + mafia`, `rapina + mafia`              |
| **Drug-related**          | `spaccio + aggressione`, `spaccio + rapina`                                                                        |
| **Fraud combinations**    | `truffa + estorsione`, `truffa + furto`                                                                            |
| **Robbery escalations**   | `rapina + tentato_omicidio`, `rapina + omicidio_stradale`                                                          |

### Output Files

Output is saved to `datasets/<model_name>/` with separate files:

- `dataset_crime.json` - Crime articles
- `dataset_non_crime.json` - Non-crime articles
- `dataset_ambiguous.json` - Ambiguous cases (look like crimes but aren't)

## Web Scraping

> **Note**: The scraper scripts and scraped data are **not included** in this repository due to copyright and licensing considerations. Two templates are provided as reference:
>
> - `scraper/scraper_template.py` - generic scraper structure, customize it on your sources
> - `scraper/deduplicate_clustering_template.py` - deduplication script with masked source names

Scrape real news articles from a website:

```bash
cd scraper
python scraper_*.py
```

Output is saved to `scraper/news/news_*/`.

### Article Deduplication (Work In Progress)

> **Note**: This feature is still under development.

Since articles are scraped from multiple news sources, duplicates (same event from different sources as well as multiple articles of the same event) need to be identified and merged. The system uses **clustering on sentence embeddings** to find similar articles.

```bash
cd scraper

# Basic deduplication with default settings
python deduplicate_clustering.py

# Custom DBSCAN epsilon (lower = stricter matching)
python deduplicate_clustering.py --eps 0.5

# Custom date tolerance
python deduplicate_clustering.py --date-tolerance 14

# Custom output path
python deduplicate_clustering.py --output my_deduplicated.json

# Only merge cross-source duplicates
python deduplicate_clustering.py --cross-source-only
```

#### How It Works

1. **Embeddings**: Creates sentence embeddings for all articles using `sentence-transformers`
2. **Clustering**: Uses DBSCAN to find natural groupings of similar articles
3. **Date Filtering**: Rejects clusters where articles are too far apart in time (avoids merging recurring events)
4. **Cross-Source Detection**: Only merges clusters containing articles from different sources

#### Output Format

Deduplicated articles include source tracking:

```json
{
  "title": "Title...",
  "content": "Article content...",
  "date": "2025-01-15",
  "sources": [
    { "name": "Source1", "link": "https://...", "date": "2025-01-15" },
    { "name": "Source2", "link": "https://...", "date": "2025-01-16" }
  ],
  "is_merged": true,
  "merge_count": 2
}
```

#### Configuration

Parameters in `deduplicate_clustering.py`:

- `DBSCAN_EPS = 0.5`: Max embedding distance for same cluster (lower = stricter)
- `DATE_TOLERANCE_DAYS = 14`: Max days apart for valid duplicate
- `DBSCAN_MIN_SAMPLES = 2`: Minimum articles per cluster

#### Database Integration

The deduplicated output is ready for database insertion:

```bash
python fill_db.py --input scraper/deduplicated_clustered.json
```

## Web Application

The BERTMAN folder contains a full-stack web application:

- **Frontend**: React with TypeScript
- **Backend**: FastAPI with Python

```bash
cd BERTMAN

# Install dependencies
npm install

# Run development servers
npm run dev
```

## Requirements

### Python Dependencies

```
torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm
beautifulsoup4
requests
python-dotenv
google-generativeai
openpyxl
sentence-transformers
```

### Environment Variables

Create a `.env` file with your API keys:

```env
GOOGLE_API_KEY=your_gemini_api_key
```
