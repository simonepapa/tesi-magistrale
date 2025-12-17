# Documentazione degli Script in `models/`

## Sommario

Questo documento fornisce una descrizione dettagliata di ogni script e funzione presente nella cartella `models/`. Questi script costituiscono il nucleo del sistema di classificazione multi-label di articoli di cronaca relativi a crimini.

---

## 1. Panoramica degli Script

| Script                 | Scopo                                            |
| ---------------------- | ------------------------------------------------ |
| `config.py`            | Configurazione centralizzata per tutti i modelli |
| `train.py`             | Training unificato per BERT, mDeBERTa, UmBERTo   |
| `evaluate.py`          | Valutazione dettagliata con metriche e grafici   |
| `inference.py`         | Inferenza con chunking per articoli lunghi       |
| `compare_models.py`    | Confronto delle performance tra modelli          |
| `evaluate_llm_api.py`  | Valutazione LLM via API (Gemma 3)                |
| `hyperparam_search.py` | Grid/Random/Bayesian search iperparametri        |
| `label_quartieri.py`   | Pipeline di labeling per quartieri di Bari       |
| `extract_streets.py`   | Estrazione di indirizzi stradali dal testo       |

---

## 2. `config.py` - Configurazione Centralizzata

### 2.1 Descrizione

File di configurazione centrale che contiene tutte le costanti, mappature e funzioni di utilità condivise tra gli script.

### 2.2 Costanti

#### `LABELS`

**Tipo**: `List[str]`

**Scopo**: Lista delle 13 categorie di crimine supportate dal sistema.

```python
LABELS = [
    'omicidio', 'omicidio_colposo', 'omicidio_stradale', 'tentato_omicidio',
    'furto', 'rapina', 'violenza_sessuale', 'aggressione', 'spaccio',
    'truffa', 'estorsione', 'contrabbando', 'associazione_di_tipo_mafioso'
]
```

---

#### `id2label` e `label2id`

**Tipo**: `Dict[int, str]` e `Dict[str, int]`

**Scopo**: Mappature bidirezionali tra indici numerici e nomi delle etichette. Necessarie per la compatibilità con i modelli HuggingFace.

```python
id2label = {0: 'omicidio', 1: 'omicidio_colposo', ...}
label2id = {'omicidio': 0, 'omicidio_colposo': 1, ...}
```

---

#### `MAX_LENGTH`, `CHUNK_OVERLAP`, `THRESHOLD`

**Tipo**: `int`, `int`, `float`

**Scopo**: Parametri per l'inferenza.

| Parametro       | Valore | Descrizione                                        |
| --------------- | ------ | -------------------------------------------------- |
| `MAX_LENGTH`    | 512    | Lunghezza massima in token per input               |
| `CHUNK_OVERLAP` | 50     | Sovrapposizione tra chunk per articoli lunghi      |
| `THRESHOLD`     | 0.75   | Soglia di probabilità per classificazione positiva |

---

#### `MODELS`

**Tipo**: `Dict[str, Dict]`

**Scopo**: Configurazioni specifiche per ogni modello supportato.

```python
MODELS = {
    'bert': {
        'name': 'BERT',
        'checkpoint': 'bert_fine_tuned_model',
        'base_model': 'dbmdz/bert-base-italian-xxl-cased',
        'use_fast_tokenizer': True,
        'output_dir': 'data/labeled_bert'
    },
    'mdeberta': {...},
    'umberto': {...}
}
```

**Motivazione tecnica**:

- **BERT italiano**: Modello pre-addestrato specificamente su testi italiani
- **mDeBERTa**: Richiede tokenizer lento per compatibilità
- **UmBERTo**: Addestrato su CommonCrawl italiano

---

#### `QUARTIERI` e `QUARTIERI_ALIASES`

**Tipo**: `List[str]` e `Dict[str, List[str]]`

**Scopo**: Lista dei 18 quartieri di Bari e mappatura degli alias per normalizzazione.

**Motivazione tecnica**: Diverse fonti giornalistiche usano nomenclature diverse (es. "Libertà" vs "Libertà*-\_Marconi*-_Fesca_-\_San_Girolamo"). Il sistema di alias permette normalizzazione automatica.

---

### 2.3 Funzioni

#### `normalizza_quartiere(nome: str) → list[str]`

**Scopo**: Normalizza un nome di quartiere agli standard definiti.

**Input**:

- `nome` (str): Nome del quartiere come trovato nella fonte

**Output**: Lista di nomi standard corrispondenti (può essere multipla per zone aggregate)

**Logica**:

1. Converte il nome in minuscolo
2. Sostituisce underscore e spazi con trattini
3. Cerca nella mappa degli alias

```python
>>> normalizza_quartiere("Carbonara_-_Ceglie_-_Loseto")
['carbonara', 'ceglie-del-campo', 'loseto']
```

---

#### `get_model_config(model_name: str) → dict`

**Scopo**: Recupera la configurazione completa per un modello specifico.

**Input**:

- `model_name` (str): Nome del modello ('bert', 'mdeberta', 'umberto')

**Output**: Dizionario con tutte le configurazioni del modello

**Comportamento errore**: Solleva `ValueError` se il modello non è riconosciuto

---

#### `get_available_models() → list`

**Scopo**: Restituisce la lista dei nomi di modelli disponibili.

**Output**: `['bert', 'mdeberta', 'umberto']`

---

## 3. `train.py` - Script di Training

### 3.1 Descrizione

Script unificato per il training di tutti i modelli supportati. Gestisce il caricamento dei dati, preprocessing, training con early stopping, e salvataggio dei risultati.

### 3.2 Funzioni

#### `check_gpu() → torch.device`

**Scopo**: Verifica la disponibilità della GPU e restituisce il device appropriato.

**Output**: `torch.device("cuda")` se disponibile, altrimenti `torch.device("cpu")`

**Comportamento**:

- Stampa informazioni sulla GPU (nome, memoria)
- Avvisa se si usa CPU (training lento)

---

#### `load_json_files_from_folder(folder_path: str) → list`

**Scopo**: Carica e combina tutti i file JSON da una cartella.

**Input**:

- `folder_path` (str): Percorso alla cartella contenente i file JSON

**Output**: Lista combinata di tutti gli articoli

**Motivazione tecnica**: Permette di avere dataset suddivisi in più file (es. per categoria di crimine) e caricarli tutti insieme per il training.

---

#### `convert_labels_format(df: pd.DataFrame) → pd.DataFrame`

**Scopo**: Converte le etichette dal formato array a colonne one-hot encoded.

**Input**:

- `df` (pd.DataFrame): DataFrame con articoli

**Output**: DataFrame con colonne separate per ogni etichetta (0/1)

**Formati supportati**:

1. **Nuovo formato**: Colonna `labels` con array di stringhe → Conversione a colonne
2. **Vecchio formato**: Etichette già come colonne → Nessuna conversione

```python
# Input (nuovo formato):
{"content": "...", "labels": ["rapina", "aggressione"]}

# Output:
{"content": "...", "rapina": 1, "aggressione": 1, "furto": 0, ...}
```

---

#### `load_dataset(path, folder, test_file, extra_train, test_size, val_size) → Tuple[DatasetDict, list]`

**Scopo**: Carica e prepara il dataset con split train/val/test.

**Input**:

- `path` (str): Percorso a un singolo file JSON (opzionale)
- `folder` (str): Percorso a cartella di file JSON (alternativa a path)
- `test_file` (str): File di test separato (opzionale)
- `extra_train` (str): File con dati di training aggiuntivi (es. articoli reali)
- `test_size` (float): Proporzione per test set (default: 0.1)
- `val_size` (float): Proporzione per validation set (default: 0.1)

**Output**: Tupla di (DatasetDict HuggingFace, lista etichette disponibili)

**Logica di split**:

```
Se test_file fornito:
    Test = test_file
    Train, Val = split(main_data, val_size)
Altrimenti:
    Train+Val, Test = split(main_data, test_size)
    Train, Val = split(Train+Val, val_size_adjusted)
```

---

#### `preprocess_data(examples, tokenizer) → dict`

**Scopo**: Tokenizza e prepara i dati per il training.

**Input**:

- `examples`: Batch di esempi dal dataset HuggingFace
- `tokenizer`: Tokenizer del modello

**Output**: Dizionario con `input_ids`, `attention_mask`, `labels`

**Parametri di tokenizzazione**:

- `padding="max_length"`: Padding fisso a 512 token
- `truncation=True`: Tronca testi più lunghi
- `max_length=512`: Limite massimo

---

#### `multi_label_metrics(predictions, labels, threshold) → dict`

**Scopo**: Calcola metriche per classificazione multi-label.

**Input**:

- `predictions`: Array di logit dal modello
- `labels`: Array di etichette vere
- `threshold` (float): Soglia per binarizzazione (default: 0.5)

**Output**: Dizionario con metriche

| Metrica    | Descrizione          |
| ---------- | -------------------- |
| `f1_micro` | F1 micro-averaged    |
| `f1_macro` | F1 macro-averaged    |
| `roc_auc`  | Area Under ROC Curve |
| `accuracy` | Subset accuracy      |

**Motivazione tecnica**: F1 macro è preferito per dataset sbilanciati, mentre micro considera il contributo di ogni singola predizione.

---

#### `compute_metrics(p: EvalPrediction) → dict`

**Scopo**: Wrapper per `multi_label_metrics` compatibile con Trainer HuggingFace.

**Input**:

- `p` (EvalPrediction): Oggetto HuggingFace con predizioni e labels

**Output**: Dizionario metriche

---

#### `train(args) → None`

**Scopo**: Funzione principale di training per un singolo modello.

**Input**:

- `args` (argparse.Namespace): Argomenti da linea di comando

**Flusso di esecuzione**:

```
1. Recupera configurazione modello
2. Crea directory di output organizzate: results/{model}/{dataset}/
3. Verifica GPU
4. Carica tokenizer e modello
5. Carica e processa dataset
6. Configura TrainingArguments
7. Crea Trainer con EarlyStoppingCallback
8. Esegue training
9. Valuta su test set
10. Salva modello, tokenizer, info training
```

**Parametri di training**:

- `learning_rate`: 2e-5 (default)
- `batch_size`: 32 (default, 16 per mDeBERTa)
- `epochs`: 10
- `early_stopping_patience`: 2
- `warmup_ratio`: 0.1
- `weight_decay`: 0.01
- `fp16`: Abilitato se GPU disponibile

---

#### `train_all_models(args) → None`

**Scopo**: Addestra tutti i modelli in sequenza con parametri ottimizzati.

**Input**:

- `args` (argparse.Namespace): Argomenti da linea di comando

**Parametri ottimali per modello**:

```python
MODEL_OPTIMAL_PARAMS = {
    'bert': {'batch_size': 32, 'learning_rate': 2e-5},
    'mdeberta': {'batch_size': 16, 'learning_rate': 1e-5},
    'umberto': {'batch_size': 32, 'learning_rate': 2e-5}
}
```

**Motivazione tecnica**: mDeBERTa richiede batch size minore e learning rate inferiore per stabilità del training.

---

#### `main() → None`

**Scopo**: Entry point con parsing argomenti CLI.

**Argomenti CLI**:
| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--model` | bert | Modello da addestrare o 'all' |
| `--dataset` | - | Path a singolo file JSON |
| `--dataset_dir` | - | Path a cartella JSON |
| `--test_file` | - | File test set separato |
| `--epochs` | 10 | Numero di epoche |
| `--batch_size` | auto* | Dimensione batch (32 BERT/UmBERTo, 16 mDeBERTa) |
| `--learning_rate` | auto* | Learning rate (2e-5 BERT/UmBERTo, 1e-5 mDeBERTa) |
| `--weight_decay` | 0.01 | Weight decay per AdamW optimizer |
| `--warmup_ratio` | 0.1 | Warmup ratio per learning rate scheduler |
| `--patience` | 2 | Patience early stopping |
| `--extra_train` | - | File con dati training aggiuntivi (es. articoli reali) |
| `--kfold` | 0 | Numero di fold per K-Fold Cross-Validation. Default 0 usa split standard train/val/test |

\*I parametri `batch_size` e `learning_rate` usano valori ottimali per ogni modello (`MODEL_OPTIMAL_PARAMS`) a meno che non vengano specificati esplicitamente.

> ⚠️ **Attenzione**: Usando `--model all` con k-fold si eseguono 3 modelli × N fold. Questo può richiedere molto tempo.

---

#### `train_kfold(args) → None`

**Scopo**: Addestra un modello usando K-Fold Cross-Validation per una stima più robusta delle performance.

**Input**:

- `args` (argparse.Namespace): Argomenti da linea di comando con `kfold` specificato

**Flusso di esecuzione**:

```
1. Carica e combina dataset + extra_train
2. Per ogni fold:
   a. Divide dati in train/val
   b. Carica modello fresco da base_model
   c. Addestra e valuta
   d. Salva modello del fold
3. Calcola metriche medie e deviazione standard
4. Salva summary in kfold_summary.json
```

**Output directory**: `results/{model}/{dataset}_kfold{n}/`

**Contenuto summary**:

```python
{
    'model': 'bert',
    'n_folds': 5,
    'fold_metrics': [{...}, {...}, ...],
    'average_metrics': {
        'f1_macro': 0.85,
        'f1_macro_std': 0.02,
        'f1_micro': 0.87,
        'accuracy': 0.82
    }
}
```

---

## 4. `evaluate.py` - Script di Valutazione

### 4.1 Descrizione

Script per valutazione dettagliata dei modelli con generazione di confusion matrix, classificazione report e grafici F1.

### 4.2 Funzioni

#### `load_model(model_name, checkpoint) → Tuple`

**Scopo**: Carica un modello per valutazione.

**Input**:

- `model_name` (str): Nome del modello
- `checkpoint` (str): Path checkpoint custom (opzionale)

**Output**: Tupla (model, tokenizer, device, config)

---

#### `load_json_files_from_folder(folder_path: str) → list`

**Scopo**: Identica alla funzione in `train.py`. Carica JSON da cartella.

---

#### `convert_labels_format(df: pd.DataFrame) → pd.DataFrame`

**Scopo**: Identica alla funzione in `train.py`. Converte formato etichette.

---

#### `load_dataset(dataset_path, dataset_dir) → Tuple[pd.DataFrame, list]`

**Scopo**: Carica il dataset per valutazione.

**Input**:

- `dataset_path` (str): Path a singolo file JSON
- `dataset_dir` (str): Path a cartella con file JSON

**Output**: Tupla (DataFrame, lista etichette disponibili)

**Differenza da train.py**: Non esegue split, carica tutto il dataset per valutazione.

---

#### `predict_batch(model, tokenizer, texts, device, batch_size) → np.ndarray`

**Scopo**: Esegue predizioni batch-wise su una lista di testi.

**Input**:

- `model`: Modello PyTorch
- `tokenizer`: Tokenizer
- `texts` (list): Lista di testi
- `device`: Device PyTorch
- `batch_size` (int): Dimensione batch (default: 16)

**Output**: Array numpy di probabilità (N × num_labels)

**Motivazione tecnica**: Il processing a batch evita out-of-memory per dataset grandi e sfrutta parallelismo GPU.

---

#### `evaluate_model(df, model, tokenizer, device, available_labels, model_name) → Tuple`

**Scopo**: Esegue valutazione completa del modello.

**Input**:

- `df` (pd.DataFrame): Dataset di test
- `model`: Modello
- `tokenizer`: Tokenizer
- `device`: Device
- `available_labels` (list): Etichette da valutare
- `model_name` (str): Nome modello

**Output**: Tupla (results_dict, y_true, y_pred, y_probs, labels)

**Metriche calcolate**:

- Per ogni classe: Precision, Recall, F1, Support
- Globali: Micro/Macro averages

---

#### `plot_confusion_matrices(y_true, y_pred, labels, output_dir, model_name) → None`

**Scopo**: Genera e salva matrice di confusione per ogni etichetta.

**Input**:

- `y_true`: Etichette vere
- `y_pred`: Predizioni
- `labels`: Lista nomi etichette
- `output_dir`: Directory output
- `model_name`: Nome modello

**Output**: Salva `confusion_matrices.png`

**Layout**: Griglia 4 colonne × N righe, una heatmap per etichetta.

---

#### `plot_class_distribution(df, labels, output_dir) → None`

**Scopo**: Genera grafico distribuzione classi nel dataset.

**Output**: Salva `class_distribution.png`

**Visualizzazione**: Barre orizzontali, rosso per classi con <50 campioni.

---

#### `plot_f1_scores(results, output_dir, model_name) → None`

**Scopo**: Genera grafico F1 per ogni classe.

**Output**: Salva `f1_scores.png`

**Visualizzazione**: Barre orizzontali colorate:

- Rosso: F1 < 0.5
- Giallo: 0.5 ≤ F1 < 0.7
- Verde: F1 ≥ 0.7

---

#### `main() → None`

**Scopo**: Entry point con parsing argomenti.

**Argomenti CLI**:
| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--model` | bert | Modello da valutare o 'all' per tutti |
| `--checkpoint` | auto | Path checkpoint custom |
| `--dataset` | - | Path file JSON |
| `--dataset_dir` | - | Path cartella JSON |
| `--dataset_models` | - | Nome dataset per localizzare modelli |
| `--output` | auto | Directory output |

**Nuova funzionalità `--model all`**:

Quando si usa `--model all`, lo script valuta sequenzialmente tutti i modelli disponibili e genera un report comparativo finale:

```
EVALUATION SUMMARY
============================================================
bert:
  F1 Macro: 0.8532
  F1 Micro: 0.8721
  Accuracy: 0.8156

mdeberta:
  F1 Macro: 0.8298
  ...
```

---

## 5. `inference.py` - Inferenza con Chunking

### 5.1 Descrizione

Modulo per inferenza su articoli di qualsiasi lunghezza. Implementa chunking con sovrapposizione per gestire testi che eccedono il limite di 512 token.

### 5.2 Funzioni

#### `load_model(model_name, checkpoint) → Tuple`

**Scopo**: Carica modello fine-tuned per inferenza.

**Input**:

- `model_name` (str): Nome modello ('bert', 'mdeberta', 'umberto')
- `checkpoint` (str): Path checkpoint custom (opzionale)

**Output**: Tupla (model, tokenizer, device, config)

**Comportamento**:

- Imposta modello in eval mode
- Stampa numero parametri

---

#### `split_into_chunks(text, tokenizer, max_length, overlap) → List[str]`

**Scopo**: Divide un testo lungo in chunk sovrapposti.

**Input**:

- `text` (str): Testo da dividere
- `tokenizer`: Tokenizer del modello
- `max_length` (int): Lunghezza massima chunk (default: 512)
- `overlap` (int): Token di sovrapposizione (default: 50)

**Output**: Lista di chunk testuali

**Algoritmo**:

```
1. Tokenizza intero testo
2. Se token ≤ max_length - 2: ritorna [text]
3. Altrimenti:
   - Calcola step = max_length - 2 - overlap
   - Per ogni finestra scorrevole:
     - Estrae token[start:end]
     - Decodifica in testo
     - Aggiunge a lista chunk
```

**Motivazione tecnica**: L'overlap di 50 token garantisce continuità semantica tra chunk adiacenti, evitando perdita di contesto ai bordi.

---

#### `predict_with_chunking(model, tokenizer, text, device, aggregation) → Tuple[Dict, int]`

**Scopo**: Esegue predizione su testo con chunking.

**Input**:

- `model`: Modello
- `tokenizer`: Tokenizer
- `text` (str): Testo input
- `device`: Device PyTorch
- `aggregation` (str): Metodo aggregazione ('max', 'mean', 'any')

**Output**: Tupla (dict predizioni, numero chunk usati)

**Formato predizioni**:

```python
{
    "rapina": {"value": 1, "prob": 0.92},
    "aggressione": {"value": 0, "prob": 0.45},
    ...
}
```

**Metodi di aggregazione**:
| Metodo | Logica |
|--------|--------|
| `max` | Prende probabilità massima tra chunk |
| `mean` | Media delle probabilità tra chunk |
| `any` | Come max (per compatibilità) |

**Motivazione tecnica**: `max` è preferito per crimini perché un crimine menzionato in qualsiasi parte dell'articolo è rilevante.

---

#### `label_article(article, model, tokenizer, device, aggregation) → Dict`

**Scopo**: Etichetta un singolo articolo con crimini e indirizzi stradali.

**Input**:

- `article` (Dict): Dizionario articolo con chiave `content`
- `model`, `tokenizer`, `device`: Come sopra
- `aggregation` (str): Metodo aggregazione

**Output**: Articolo arricchito con:

- Predizioni per ogni categoria crimine
- `_chunks_used`: Numero chunk usati
- `streets`: Lista indirizzi estratti

---

#### `label_articles_from_file(input_file, output_file, model_name, quartiere, aggregation, verbose, checkpoint) → None`

**Scopo**: Etichetta tutti gli articoli in un file JSON.

**Input**:

- `input_file` (str): Path file JSON input
- `output_file` (str): Path file JSON output
- `model_name` (str): Nome modello
- `quartiere` (str): Nome quartiere da aggiungere (opzionale)
- `aggregation` (str): Metodo aggregazione
- `verbose` (bool): Stampa progresso
- `checkpoint` (str): Path checkpoint custom

**Output**: Salva file JSON con articoli etichettati

**Statistiche stampate**:

- Totale articoli elaborati
- Articoli che hanno richiesto chunking
- Media chunk per articolo

---

#### `run_test(model_name, checkpoint) → None`

**Scopo**: Esegue test con testo di esempio.

**Input**:

- `model_name` (str): Nome modello
- `checkpoint` (str): Path checkpoint custom

**Comportamento**: Classifica un articolo di esempio e stampa risultati.

---

#### `main() → None`

**Scopo**: Entry point CLI.

**Argomenti CLI**:
| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--model` | bert | Modello da usare |
| `--checkpoint` | auto | Path checkpoint custom |
| `--input` | - | File JSON input |
| `--output` | auto | File JSON output |
| `--quartiere` | - | Nome quartiere |
| `--aggregation` | max | Metodo aggregazione |
| `--test` | false | Esegui test con esempio |

---

## 6. `compare_models.py` - Confronto Modelli

### 6.1 Descrizione

Script per confrontare le performance di più modelli sullo stesso dataset, analizzando accordi/disaccordi e generando report comparativi.

### 6.2 Funzioni

#### `find_checkpoint(model_name, dataset_name) → str`

**Scopo**: Trova il path del checkpoint per un modello.

**Input**:

- `model_name` (str): Nome modello
- `dataset_name` (str): Nome dataset usato per training (opzionale)

**Output**: Path al checkpoint o None

**Logica di ricerca**:

```
1. Se dataset_name: cerca results/{model}/{dataset}/model
2. Altrimenti: cerca prima sottodirectory in results/{model}/
3. Fallback: ritorna None (usa default config)
```

---

#### `convert_labels_format(df) → pd.DataFrame`

**Scopo**: Identica alle altre. Converte formato etichette.

---

#### `load_test_data(dataset_path, test_file, test_size) → pd.DataFrame`

**Scopo**: Carica porzione test del dataset.

**Input**:

- `dataset_path` (str): Path dataset completo
- `test_file` (str): File test separato
- `test_size` (float): Proporzione test (default: 0.1)

**Output**: DataFrame del test set

**Logica**: Usa stesso split di training per garantire test set identico.

---

#### `evaluate_single_model(model_name, test_df, dataset_name, run_folder) → Tuple[Dict, np.ndarray, np.ndarray]`

**Scopo**: Valuta un singolo modello sul test set.

**Input**:

- `model_name` (str): Nome modello
- `test_df` (pd.DataFrame): Test set
- `dataset_name` (str): Nome dataset training
- `run_folder` (str): Cartella run specifica (es. `e10_b32_v1`) (opzionale)

**Output**: Tupla (metriche_dict, predizioni, probabilità)

**Metriche calcolate**:

- F1 micro/macro
- Precision micro/macro
- Recall micro/macro
- Accuracy
- ROC AUC micro/macro
- F1 per ogni classe

---

#### `compare_predictions(predictions, content, title) → Dict`

**Scopo**: Confronta predizioni di più modelli su un articolo.

**Input**:

- `predictions` (Dict[str, Dict]): Predizioni per modello
- `content` (str): Testo articolo
- `title` (str): Titolo articolo

**Output**: Dizionario con accordi e disaccordi

```python
{
    "title": "...",
    "agreements": [{"label": "rapina", "predictions": {...}}],
    "disagreements": [{"label": "furto", "predictions": {...}}]
}
```

---

#### `run_quick_comparison(models_to_compare, dataset_name, run_folders) → None`

**Scopo**: Confronto veloce su testo di esempio.

**Input**:

- `models_to_compare` (List[str]): Lista modelli
- `dataset_name` (str): Nome dataset training
- `run_folders` (Dict[str, str]): Mapping modello → cartella run (opzionale)

**Output**: Stampa tabella comparativa side-by-side.

---

#### `run_sample_comparison(models_to_compare, num_samples, dataset_path, test_file, dataset_name, run_folders) → None`

**Scopo**: Confronta modelli su campione di articoli.

**Input**:

- `models_to_compare` (List[str]): Lista modelli
- `num_samples` (int): Numero campioni (default: 10)
- `dataset_path`, `test_file`, `dataset_name`: Path dati
- `run_folders` (Dict[str, str]): Mapping modello → cartella run (opzionale)

**Output**: Stampa risultati per ogni campione.

---

#### `run_full_evaluation(models_to_compare, dataset_path, test_file, dataset_name, run_folders) → dict`

**Scopo**: Valutazione completa su intero test set.

**Input**:

- `models_to_compare` (List[str]): Lista modelli
- `dataset_path`, `test_file`, `dataset_name`: Path dati
- `run_folders` (Dict[str, str]): Mapping modello → cartella run (opzionale)

**Output**: Dizionario risultati completo

**Analisi generate**:

1. Tabella metriche principali
2. F1 per classe con indicazione vincitore
3. Conteggio vittorie per categoria
4. Analisi accordo pairwise tra modelli
5. Determinazione miglior modello (F1 Macro)

**File output**: `evaluation_results/model_comparison.json`

---

#### `main() → None`

**Scopo**: Entry point CLI.

**Argomenti CLI**:
| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--mode` | quick | Modalità: quick/sample/evaluate/full |
| `--models` | all | Lista modelli (comma-separated) |
| `--samples` | 10 | Numero campioni per mode sample |
| `--dataset` | - | Path dataset |
| `--test_file` | - | Path test file |
| `--dataset_models` | - | Nome dataset per trovare checkpoint |
| `--bert_run` | - | Cartella run per BERT (es. `e10_b32_v1`) |
| `--mdeberta_run` | - | Cartella run per mDeBERTa (es. `e10_b16_v1`) |
| `--umberto_run` | - | Cartella run per UmBERTo (es. `e10_b32_v1`) |
| `--llm_results` | - | Path file JSON risultati LLM (da `evaluate_llm_api.py`) |

---

## 7. `evaluate_llm_api.py` - Valutazione LLM via API

### 9.1 Descrizione

Script per valutare modelli Gemma 3 via Google AI API per classificazione multi-label. Permette di confrontare le performance di LLM zero/few-shot con i modelli fine-tuned locali.

### 7.2 Modelli Supportati

- `gemma-3-27b-it` (default, migliore qualità)
- `gemma-3-12b-it`
- `gemma-3-4b-it`
- `gemma-3-2b-it`
- `gemma-3-1b-it`

### 8.3 Funzioni Principali

#### `setup_api(api_key) → None`

**Scopo**: Configura l'API Google Generative AI.

**Input**: `api_key` (str, opzionale): Chiave API (o usa env var `GEMINI_API_KEY`)

---

#### `create_model(model_name) → GenerativeModel`

**Scopo**: Crea istanza del modello con safety settings per contenuti di cronaca nera.

---

#### `create_zero_shot_prompt(article_text) → str`

**Scopo**: Genera prompt zero-shot per classificazione.

---

#### `create_few_shot_prompt(article_text) → str`

**Scopo**: Genera prompt few-shot con 5 esempi per migliorare l'accuratezza.

---

#### `classify_article(model, article_text, few_shot) → Tuple[List[str], str]`

**Scopo**: Classifica un singolo articolo via API.

**Output**: Tupla (labels predette, risposta raw)

---

#### `evaluate_llm(model, test_df, few_shot, rate_limit_delay) → Dict`

**Scopo**: Valuta LLM su intero test set.

**Output**: Dizionario con metriche, predizioni binarie, errori.

---

#### `main() → None`

**Scopo**: Entry point CLI.

**Argomenti CLI**:
| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--test_file` | (required) | Path test set JSON |
| `--model` | gemma-3-27b-it | Modello da usare |
| `--few_shot` | False | Usa few-shot prompting |
| `--api_key` | - | API key (o usa env var) |
| `--delay` | 0.5 | Delay tra chiamate API (secondi) |
| `--limit` | - | Limita numero articoli (per test) |
| `--output` | auto | File output risultati |

### 7.4 Output

File JSON con:

- Metriche (F1, precision, recall, accuracy)
- Predizioni binarie (per agreement analysis)
- Esempi di errori

---

## 8. `hyperparam_search.py` - Ottimizzazione Iperparametri

### 9.1 Descrizione

Script per ottimizzazione iperparametri con tre strategie di ricerca:

- **Grid Search**: Ricerca esaustiva su tutte le combinazioni
- **Random Search**: Campionamento casuale dello spazio di ricerca
- **Bayesian Search**: Ottimizzazione intelligente con Optuna (TPE sampler)

### 8.2 Costanti

#### `GRID_PARAMS` / `QUICK_GRID_PARAMS`

Spazi di ricerca per grid search (esaustivo):

```python
GRID_PARAMS = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'batch_size': [16, 32],
    'weight_decay': [0.01, 0.1],
    'warmup_ratio': [0.0, 0.1],
}  # 24 combinazioni
```

#### `SEARCH_SPACE`

Spazio di ricerca per random/bayesian search:

```python
SEARCH_SPACE = {
    'learning_rate': {'type': 'loguniform', 'low': 1e-6, 'high': 1e-4},
    'batch_size': {'type': 'categorical', 'choices': [8, 16, 32]},
    'weight_decay': {'type': 'loguniform', 'low': 0.001, 'high': 0.3},
    'warmup_ratio': {'type': 'uniform', 'low': 0.0, 'high': 0.2},
    'label_smoothing': {'type': 'uniform', 'low': 0.0, 'high': 0.2},
}
```

### 8.3 Funzioni

#### `run_grid_search(...)` → List

**Scopo**: Ricerca esaustiva su tutte le combinazioni di iperparametri.

---

#### `run_random_search(...)` → List

**Scopo**: Campiona configurazioni casuali dallo spazio di ricerca.

**Vantaggi**: Più veloce della grid search, copre meglio spazi ad alta dimensionalità.

---

#### `run_bayesian_search(...)` → List

**Scopo**: Utilizza Optuna con TPE (Tree-structured Parzen Estimator) per trovare intelligentemente le configurazioni migliori.

**Vantaggi**: Converge più velocemente verso l'ottimo, usa informazioni dalle prove precedenti.

**Requisiti**: `pip install optuna`

---

#### `main()` → None

**Argomenti CLI**:
| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--model` | bert | Modello da ottimizzare o 'all' |
| `--method` | grid | Metodo: grid, random, bayesian |
| `--dataset_dir` | - | Path cartella JSON |
| `--extra_train` | - | File training aggiuntivo |
| `--epochs` | 5 | Epoche per configurazione |
| `--n_trials` | 10 | Numero prove (random/bayesian) |
| `--quick` | false | Usa griglia ridotta (solo grid) |
| `--output` | auto | File output risultati |

### 8.4 Confronto Metodi

| Metodo   | Velocità | Qualità          | Uso consigliato            |
| -------- | -------- | ---------------- | -------------------------- |
| Grid     | Lenta    | Ottimo garantito | Spazi piccoli              |
| Random   | Veloce   | Buono            | Esplorazione rapida        |
| Bayesian | Media    | Migliore         | Uso generale (consigliato) |

---

## 9. `label_quartieri.py` - Pipeline Labeling Quartieri

### 9.1 Descrizione

Script per processare tutti i quartieri di Bari in pipeline: labeling → merge → deduplicazione.

### 9.2 Funzioni

#### `label_all_quartieri(model_name) → str`

**Scopo**: Etichetta tutti i file quartiere con il modello specificato.

**Input**:

- `model_name` (str): Nome modello

**Output**: Path directory output

**Pipeline**:

```
Per ogni quartiere in QUARTIERI:
    1. Carica data/{quartiere}.json
    2. Etichetta ogni articolo
    3. Salva in {output_dir}/labeled_{quartiere}.json
```

---

#### `merge_all_files(output_dir) → str`

**Scopo**: Unifica tutti i file labeled\_\*.json in un unico file.

**Input**:

- `output_dir` (str): Directory contenente i file labeled

**Output**: Path al file merged (`merged_file.json`)

**Logica**: Concatena tutti gli articoli da tutti i file.

---

#### `remove_duplicates(input_file, output_dir) → str`

**Scopo**: Rimuove duplicati combinando le etichette.

**Input**:

- `input_file` (str): File merged
- `output_dir` (str): Directory output

**Output**: Path al file finale (`dataset.json`)

**Logica deduplicazione**:

```
Raggruppa per titolo:
    Per ogni duplicato:
        Combina etichette con logica OR
        (Se uno ha value=1, il risultato ha value=1)
```

**Motivazione tecnica**: Lo stesso articolo può apparire in più quartieri con etichette potenzialmente diverse; la combinazione OR preserva tutte le classificazioni positive.

---

#### `main() → None`

**Scopo**: Entry point CLI.

**Argomenti CLI**:
| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--model` | bert | Modello da usare |
| `--skip-labeling` | false | Salta fase labeling |

**Flusso completo**:

```
1. [Se non skip] label_all_quartieri()
2. merge_all_files()
3. remove_duplicates()
4. Stampa riepilogo
```

---

## 8. `extract_streets.py` - Estrattore Indirizzi

### 9.1 Descrizione

Modulo per estrarre indirizzi stradali italiani dal testo degli articoli usando pattern regex.

### 8.2 Costanti

#### `STREET_PREFIXES`

**Tipo**: `List[str]`

**Scopo**: Lista prefissi stradali italiani riconosciuti.

```python
STREET_PREFIXES = [
    "Via", "Viale", "Corso", "Piazza", "Piazzale", "Piazzetta",
    "Largo", "Vicolo", "Traversa", "Strada", "Lungomare",
    "Contrada", "Borgata", "Rione", "Quartiere"
]
```

---

#### `STREET_PATTERN`

**Tipo**: `re.Pattern`

**Scopo**: Regex compilata per matching indirizzi.

**Struttura pattern**:

```
(Prefisso) + (articolo opzionale) + (Nome proprio)

Esempi match:
- "Via Roma"
- "Piazza della Libertà"
- "Corso Vittorio Emanuele"
```

---

### 8.3 Funzioni

#### `extract_streets(text: str) → List[str]`

**Scopo**: Estrae tutti gli indirizzi stradali da un testo.

**Input**:

- `text` (str): Testo dell'articolo

**Output**: Lista di indirizzi unici trovati

**Esempio**:

```python
>>> extract_streets("L'incidente è avvenuto in Via Roma, vicino a Piazza Garibaldi.")
['Via Roma', 'Piazza Garibaldi']
```

**Caratteristiche**:

- Evita duplicati (usa set interno)
- Gestisce articoli italiani (del, della, dei, etc.)
- Supporta nomi composti (Via della Libertà)
- Non include numeri civici

---

## 9. Flusso di Lavoro Tipico

### 9.1 Training

```bash
# Addestra BERT su dataset
python train.py --model bert --dataset_dir datasets/gemma-3-27b-it

# Addestra tutti i modelli
python train.py --model all --dataset_dir datasets/gemma-3-27b-it
```

### 9.2 Valutazione

```bash
# Valuta modello specifico
python evaluate.py --model bert --dataset_dir datasets/test

# Confronta tutti i modelli
python compare_models.py --mode evaluate --dataset_dir datasets/test
```

### 9.3 Inferenza

```bash
# Etichetta file di articoli
python inference.py --model bert --input articles.json --output labeled.json

# Test rapido
python inference.py --model bert --test
```

### 9.4 Pipeline Completa Quartieri

```bash
# Esegue labeling, merge e deduplicazione
python label_quartieri.py --model bert
```

---

## 10. Riferimenti

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."
2. He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention."
3. Parisi, L., et al. (2020). "UmBERTo: An Italian Language Model trained with Whole Word Masking."
4. HuggingFace Transformers Documentation.
