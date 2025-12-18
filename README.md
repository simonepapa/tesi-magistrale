# Crime Classification for Bari News Articles

A multi-label classification system for Italian news articles, specifically designed to detect and categorize crime-related content from the Bari metropolitan area.

## Project Structure

```
├── datasets_generation/         # Synthetic dataset generation using LLMs
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
│   └── xyz/                     # Dataset generated with xyz
│
├── results/                     # Training results (auto-generated)
│   ├── bert/
│   │   └── xyz/                 # Results for BERT on xyz dataset
│   │       └── e10_b32_v1/      # Versioned run (epochs, batch, version)
│   ├── mdeberta/
│   └── umberto/
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

# Train a single model (uses standard train/val/test split by default)
python train.py --model bert --dataset_dir ../datasets/gemma-3-27b-it

# Train all models sequentially (WARNING: long training time!)
python train.py --model all --dataset_dir ../datasets/gemma-3-27b-it

# Add more (e.g real) articles to improve generalization
python train.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --extra_train ../datasets/train_set_real.json

# Enable k-fold cross-validation (more robust evaluation, slower)
python train.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --kfold 5

# Override default parameters (optional)
python train.py --model mdeberta --dataset_dir ../datasets/gemma-3-27b-it --batch_size 8 --learning_rate 5e-6

# Custom regularization parameters
python train.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --weight_decay 0.05 --warmup_ratio 0.15
```

**Training defaults:**

- **K-Fold**: Disabled (use `--kfold N` to enable N-fold cross-validation)
- **Batch size**: 32 for BERT/UmBERTo, 16 for mDeBERTa
- **Learning rate**: 2e-5 for BERT/UmBERTo, 1e-5 for mDeBERTa
- **Weight decay**: 0.01
- **Warmup ratio**: 0.1
- **Epochs**: 10

> ⚠️ **Warning**: Using `--model all` with k-fold trains 3 models × N folds. This can take a very long time.

> **Note**: The training script supports both the old project format (manual labels as boolean columns) and the new format (labels as an array, e.g. `"labels": ["omicidio", "rapina"]`). The format is automatically detected.

### Evaluation

```bash
# Evaluate a single model
python evaluate.py --model bert --dataset_dir ../datasets/gemma-3-27b-it

# Evaluate all models at once
python evaluate.py --model all --dataset_dir ../datasets/gemma-3-27b-it

# Evaluate from a single file
python evaluate.py --model bert --dataset dataset.json

# Specify which dataset's models to use
python evaluate.py --model all --dataset_dir ../datasets/gemma-3-27b-it --dataset_models gemma-3-27b-it

# Results are saved to evaluation_results_{model}/
```

### Inference

```bash
# Test inference with sample text
python inference.py --model bert --test

# Test with a specific trained checkpoint
python inference.py --model bert --checkpoint results/bert/gemma-3-27b-it/model --test

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

To compare the performance of BERT, mDeBERTa, and UmBERTo:

```bash
# Full comparison on a specific dataset (uses latest available version)
python compare_models.py --mode full --dataset_models gemma-3-27b-it --test_file ../datasets/test_set.json

# Specify run folder for each model separately
python compare_models.py --mode full --dataset_models gemma-3-27b-it --test_file ../datasets/test_set.json \
    --bert_run e10_b32_v1 \
    --mdeberta_run e8_b16_v1 \
    --umberto_run e10_b32_v1

# Quick comparison with specific runs
python compare_models.py --mode quick --dataset_models gemma-3-27b-it --bert_run e10_b32_v1
```

Arguments:

- `--mode`: `quick` (sample text), `sample` (10 random articles), `evaluate` or `full` (full test set evaluation).
- `--dataset_models`: Name of the dataset folder in `results/` where trained models are located.
- `--test_file`: Path to the test set JSON file.
- `--bert_run`: Run folder for BERT model (e.g., `e10_b32_v1`).
- `--mdeberta_run`: Run folder for mDeBERTa model (e.g., `e10_b16_v1`).
- `--umberto_run`: Run folder for UmBERTo model (e.g., `e10_b32_v1`).
- `--llm_api`: Path to LLM API results JSON (from `evaluate_llm_api.py`).
- `--llm_local`: Path to LLM Local results JSON (from `evaluate_llm_local.py`).

### LLM Evaluation (API-based)

Evaluate Gemma 3 models via Google AI API for comparison with fine-tuned models:

```bash
# Zero-shot classification
python evaluate_llm_api.py --test_file ../datasets/test_set.json --model gemma-3-27b-it

# Few-shot classification
python evaluate_llm_api.py --test_file ../datasets/test_set.json --model gemma-3-27b-it --few_shot

# With article truncation (reduces token usage)
python evaluate_llm_api.py --test_file ../datasets/test_set.json --few_shot --max_chars 2000
```

Available API models: `gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-3-4b-it`, `gemma-3-2b-it`, `gemma-3-1b-it`

### LLM Evaluation (Local via Ollama)

Evaluate LLM locally without API limits or content filters:

```bash
# Prerequisites: Install Ollama and pull a model
ollama pull gemma3:12b

# Run evaluation
python evaluate_llm_local.py --test_file ../datasets/test_set.json --few_shot

# With different model
python evaluate_llm_local.py --test_file ../datasets/test_set.json --model gemma3:12b --few_shot

# List recommended models
python evaluate_llm_local.py --list_models
```

### Compare All Models (including LLMs)

```bash
# Compare fine-tuned models with both LLM API and Local
python compare_models.py --mode full --test_file ../datasets/test_set.json \
    --dataset_models gemma-3-27b-it \
    --llm_api evaluation_results/llm_gemma-3-27b-it_fewshot_*.json \
    --llm_local evaluation_results/llm_local_gemma3-12b_fewshot_*.json
```

### Hyperparameter Search

Find optimal hyperparameters using three strategies:

```bash
# Grid Search (exhaustive, tests all combinations)
python hyperparam_search.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --method grid
python hyperparam_search.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --method grid --quick

# Random Search (samples random configurations)
python hyperparam_search.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --method random --n_trials 10

# Bayesian Search with Optuna (intelligent search, recommended)
python hyperparam_search.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --method bayesian --n_trials 20

# With extra training data
python hyperparam_search.py --model bert --dataset_dir ../datasets/gemma-3-27b-it --method bayesian --extra_train ../datasets/train_set_real.json
```

**Parameters searched:**

- **Learning rate**: 1e-6 to 1e-4 (log scale)
- **Batch size**: 8, 16, 32
- **Weight decay**: 0.001 to 0.3
- **Warmup ratio**: 0.0 to 0.2
- **Label smoothing**: 0.0 to 0.2 (random/bayesian only)

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
python generate_dataset_gemma.py --type all --batches_crime 60 --batches_non_crime 1560
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
GEMINI_API_KEY=your_gemini_api_key
```
