"""
Dataset Generator (Gemma)
==========================
Generates synthetic newspaper articles using Google Gemma 3 API.
Uses Gemma model which currently has higher limits compared to Gemini.

Usage:
    python generate_dataset_gemma.py --type crime
    python generate_dataset_gemma.py --type non_crime --batches_non_crime 100
    python generate_dataset_gemma.py --type ambiguous
    python generate_dataset_gemma.py --type all
"""

import google.generativeai as genai
import pandas as pd
import json
import time
import random
import re
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Config
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")
genai.configure(api_key=API_KEY)

# Gemma model as it currently has high limits (09-12-2025)
MODEL_NAME = "gemma-3-12b-it"

# Gemma config: the greater the temperature, the more variety in the news
generation_config_crimine = {
    "temperature": 0.7,
}
generation_config_non_crimine = {
    "temperature": 0.85,
}

model_crimine = genai.GenerativeModel(model_name=MODEL_NAME, generation_config=generation_config_crimine)
model_non_crimine = genai.GenerativeModel(model_name=MODEL_NAME, generation_config=generation_config_non_crimine)

# --- CRIME DEFINITIONS ---
categorie_crimine = {
    "omicidio": "Omicidio volontario, cadavere ritrovato, esecuzione.",
    "omicidio_colposo": "Incidenti sul lavoro mortali, errori fatali, crolli.",
    "omicidio_stradale": "Incidente mortale auto/moto, pirata della strada.",
    "tentato_omicidio": "Aggressione quasi letale, accoltellamento grave, sparatoria senza morti.",
    "associazione_di_tipo_mafioso": "Clan, boss, 416bis, blitz antimafia, controllo territorio.",
    "furto": "Furto in appartamento, auto rubata, borseggio (no violenza).",
    "rapina": "Rapina a mano armata, farmacia, tabaccaio (con minaccia).",
    "violenza_sessuale": "Molestie, abusi, palpeggiamenti.",
    "aggressione": "Rissa, pestaggio, lite violenta (no armi da fuoco).",
    "spaccio": "Pusher, cocaina, hashish, arresto per droga.",
    "truffa": "Truffa anziani, truffe online, falsi dipendenti INPS.",
    "estorsione": "Pizzo, minacce a commercianti.",
    "contrabbando": "Sigarette, fuochi d'artificio illegali."
}

# --- MULTI-LABEL COMBINATIONS ---
combinazioni_multilabel = [
    # Violent crimes with aggression
    (["rapina", "aggressione"], "Rapina con violenza fisica sulla vittima, pestaggio durante il furto."),
    (["furto", "aggressione"], "Furto degenerato in rissa, ladro che aggredisce il proprietario."),
    (["tentato_omicidio", "aggressione"], "Aggressione brutale con intento omicida, pestaggio quasi letale."),
    (["violenza_sessuale", "aggressione"], "Violenza sessuale con percosse, aggressione a sfondo sessuale."),
    
    # Mafia-related combinations
    (["spaccio", "associazione_di_tipo_mafioso"], "Spaccio gestito da clan mafiosi, piazza di spaccio controllata dalla criminalità organizzata."),
    (["estorsione", "associazione_di_tipo_mafioso"], "Pizzo imposto dal clan, estorsione mafiosa a commercianti."),
    (["omicidio", "associazione_di_tipo_mafioso"], "Omicidio di mafia, esecuzione ordinata dal boss, faida tra clan."),
    (["contrabbando", "associazione_di_tipo_mafioso"], "Contrabbando gestito dalla criminalità organizzata."),
    (["rapina", "associazione_di_tipo_mafioso"], "Rapina organizzata da clan, assalto a portavalori mafioso."),
    
    # Drug-related combinations
    (["spaccio", "aggressione"], "Rissa tra spacciatori, aggressione per debiti di droga."),
    (["spaccio", "rapina"], "Rapina di droga tra pusher, furto di stupefacenti."),
    
    # Fraud combinations
    (["truffa", "estorsione"], "Truffa seguita da minacce per ottenere altri soldi."),
    (["truffa", "furto"], "Truffa per entrare in casa e poi furto."),
    
    # Robbery escalations
    (["rapina", "tentato_omicidio"], "Rapina finita in sparatoria, tentato omicidio durante rapina."),
    (["rapina", "omicidio_stradale"], "Fuga dopo rapina con investimento mortale."),
]

# --- Writing styles ---
stili_giornalistici = [
    "Scrivi in tono asciutto e fattuale, come un comunicato stampa.",
    "Scrivi in tono sensazionalistico, enfatizzando i dettagli drammatici.",
    "Scrivi in tono sobrio e istituzionale, citando fonti ufficiali.",
    "Scrivi in tono narrativo, raccontando la vicenda come una storia.",
    "Scrivi in modo conciso, solo i fatti essenziali in poche righe.",
]

# --- ENHANCE DATASET WITH AMBIGUITY ---
casi_ambigui = [
    "Auto in fiamme per cortocircuito (NON incendio doloso)",
    "Tensioni verbali tra vicini sfociate in insulti (NON rissa)",
    "Controlli di routine dei carabinieri senza arresti",
    "Smarrimento oggetti/persone poi ritrovati (NON furto/sequestro)",
    "Incidente stradale senza feriti gravi (NON omicidio stradale)",
    "Protesta accesa con momenti di tensione (NON aggressione)",
    "Trattativa commerciale finita male (NON estorsione)",
    "Persona molesta allontanata da locale (NON violenza)",
]

# --- WEIGHT FOR NON-CRIME NEWS ---
argomenti_pesati = {
    # High
    "Eventi e Cultura (Fiera del Levante, Teatro Petruzzelli, mostre, concerti)": 20,
    "Curiosità e Cronaca Bianca (Storie di quartiere, volontariato, iniziative solidali)": 20,
    "Politica locale (Consiglio comunale, Sindaco, decisioni giunta, manutenzione)": 15,
    
    # Medium
    "Economia locale (Saldi, nuove aperture negozi, turismo, caro prezzi)": 10,
    "Scuola e Università (Scioperi, ricerca, edilizia scolastica, progetti studenti)": 10,
    "Meteo e Ambiente (Allerte meteo, caldo record, verde pubblico, rifiuti)": 10,
    
    # Low
    "Calcio Bari (SSC Bari, risultati, stadio San Nicola)": 5,
    "Sanità (Ospedale Policlinico, liste d'attesa, ASL)": 5,
    "Traffico e Viabilità (Lavori in corso, deviazioni temporanee)": 5
}

lista_topics = list(argomenti_pesati.keys())
lista_pesi = list(argomenti_pesati.values())

# --- DEFAULTS ---
ROWS_PER_BATCH = 5
DEFAULT_BATCHES_CRIME = 60       # Per category (13 categories) = ~3900 articles
DEFAULT_BATCHES_NON_CRIME = 700  # = ~3500 articles
DEFAULT_BATCHES_AMBIGUOUS = 35   # = ~175 articles
MAX_CONSECUTIVE_ERRORS = 10


def extract_json_from_response(text):
    """Extracts a JSON array from the text (Gemma doesn't support JSON mode).

    :param text: str: Raw response text from Gemma model
    :returns: list[dict] | None: Parsed JSON array or None if parsing fails

    """
    # Remove markdown
    text = text.replace("```json", "").replace("```", "").strip()
    
    # Look for JSON array in text
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: parse whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_streets():
    """Load street names from Excel file."""
    print("Loading streets file...")
    try:
        df_vie = pd.read_excel("stradario_bari.xlsx")
        df_vie['NOME_COMPLETO'] = df_vie.iloc[:, 2].astype(str).str.strip() + " " + df_vie.iloc[:, 3].astype(str).str.strip()
        streets = [v for v in df_vie['NOME_COMPLETO'].tolist() if len(v) > 4 and "nan" not in v.lower()]
        print(f"  Loaded {len(streets)} streets")
        return streets
    except Exception as e:
        print(f"  Warning: Could not load stradario_bari.xlsx ({e})")
        return ["Via Sparano", "Corso Cavour", "Via Manzoni", "Lungomare Nazario Sauro", "Piazza Umberto", "Via Napoli"]


def genera_data_recente():
    """Generates a random date within the last year in ISO format.
    
    :returns: str: Date string in YYYY-MM-DD format
    
    """
    return (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")


def genera_fake_link(titolo):
    """Generates a fake news URL from an article title.

    :param titolo: str: Article title to convert to URL slug
    :returns: str: Fake news URL

    """
    slug = re.sub(r'[^a-zA-Z0-9\s]', '', titolo).lower().replace(" ", "-")[:50]
    return f"https://bari-news-24.it/{slug}"


def get_versioned_filename(output_dir, base_name):
    """Returns file name with version if it exists.

    :param output_dir: Path: Directory where the file will be saved
    :param base_name: str: Base filename (e.g., 'dataset_crime.json')
    :returns: Path: Versioned filepath (e.g., dataset_crime_v2.json if v1 exists)

    """
    filepath = output_dir / base_name
    if not filepath.exists():
        return filepath
    name = filepath.stem
    ext = filepath.suffix
    version = 2
    while True:
        versioned_name = output_dir / f"{name}_v{version}{ext}"
        if not versioned_name.exists():
            return versioned_name
        version += 1


def save_dataset(data, output_dir, base_name):
    """Saves dataset with versioning.

    :param data: list[dict]: List of article dictionaries to save
    :param output_dir: Path: Directory where the file will be saved
    :param base_name: str: Base filename for the output file

    """
    if not data:
        print(f"No data to save for {base_name}")
        return
    filename = get_versioned_filename(output_dir, base_name)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} articles to: {filename}")


def generate_crime_news(batches, streets, output_dir, categories=None, skip_multilabel=False):
    """Generates crime news articles (single-label and multi-label).

    :param batches: int: Number of batches to generate per crime category
    :param streets: list[str]: List of street names for location generation
    :param output_dir: Path: Directory where the output file will be saved
    :param categories: list[str] | None: Specific categories to generate (None = all)
    :param skip_multilabel: bool: If True, skip multi-label generation
    :returns: list[dict]: List of generated crime article dictionaries

    Generates ~80% single-label and ~20% multi-label crime articles.
    """
    dataset = []
    consecutive_errors = 0
    
    # Filter categories if specified
    if categories:
        selected_categories = {k: v for k, v in categorie_crimine.items() if k in categories}
        if not selected_categories:
            print(f"⚠️ No valid categories found in: {categories}")
            print(f"Valid categories: {list(categorie_crimine.keys())}")
            return dataset
    else:
        selected_categories = categorie_crimine
    
    # Calculate batches: 80% single-label, 20% multi-label
    single_label_batches = int(batches * 0.8) if not categories else batches
    multilabel_batches = batches - single_label_batches if not categories else 0
    
    # Skip multilabel if only generating specific categories or flag is set
    if skip_multilabel or categories:
        multilabel_batches = 0
    
    print("\n--- GENERATING CRIME NEWS ---")
    if categories:
        print(f"Selected categories: {list(selected_categories.keys())}")
    print(f"Batches per category (single-label): {single_label_batches if not categories else batches}")
    if multilabel_batches > 0:
        print(f"Batches for multi-label: {multilabel_batches * len(combinazioni_multilabel)}")
    print(f"Categories: {len(selected_categories)}")
    expected_single = (single_label_batches if not categories else batches) * len(selected_categories) * ROWS_PER_BATCH
    expected_multi = multilabel_batches * len(combinazioni_multilabel) * ROWS_PER_BATCH if multilabel_batches > 0 else 0
    print(f"Expected articles: ~{expected_single} single-label" + (f" + ~{expected_multi} multi-label" if expected_multi > 0 else "") + f" = ~{expected_single + expected_multi} total")
    
    # --- Single-label articles ---
    print("\n[PHASE 1] Generating single-label articles...")
    for cat, desc in selected_categories.items():
        print(f"\nGenerating: {cat}...")
        batch_count = batches if categories else single_label_batches
        for i in range(batch_count):
            if i % 10 == 0:
                print(f"  Batch {i}/{single_label_batches} ({len(dataset)} articles so far)")
            
            if random.random() < 0.2:
                via_scelta = random.choice(streets)
                istruzioni_luogo = f"Ambientazione: Cita esplicitamente '{via_scelta}'."
            else:
                istruzioni_luogo = "Ambientazione: Generica (es. 'in centro', 'quartiere periferico', 'zona industriale'). NON citare vie specifiche."

            stile = random.choice(stili_giornalistici)
            
            prompt = f"""
            Scrivi {ROWS_PER_BATCH} articoli di giornale locale di Bari in stile REALISTICO.
            L'articolo deve riportare un fatto che rientra in questa tipologia: {desc}.
            
            REGOLE IMPORTANTI PER REALISMO:
            - {stile}
            - NON tutti gli articoli devono menzionare esplicitamente il crimine nel titolo
            - Alcuni articoli possono riportare il fatto in modo indiretto o accennato
            - Usa terminologia giornalistica italiana autentica
            - Varia la lunghezza: alcuni brevi (2-3 frasi), altri più dettagliati
            {istruzioni_luogo}
            
            IMPORTANTE: Rispondi SOLO con un array JSON valido, senza altro testo.
            Schema JSON: [{{"title": "...", "content": "...", "labels": ["{cat}"]}}]
            """
            
            try:
                res = model_crimine.generate_content(prompt)
                data = extract_json_from_response(res.text)
                if data:
                    for x in data:
                        x['date'] = genera_data_recente()
                        x['link'] = genera_fake_link(x.get('title', ''))
                        x['labels'] = [cat]  # Force single label
                        dataset.append(x)
                    consecutive_errors = 0
                else:
                    print(f"    ⚠️ Invalid JSON for {cat} batch {i}")
                    consecutive_errors += 1
                time.sleep(5)
            except Exception as e:
                print(f"    Err {cat} batch {i}: {e}")
                consecutive_errors += 1
                time.sleep(10)
            
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"❌ Too many consecutive errors ({MAX_CONSECUTIVE_ERRORS}), saving and stopping...")
                break
        
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            break
    
    # --- Multi-label articles ---
    if consecutive_errors < MAX_CONSECUTIVE_ERRORS:
        print("\n[PHASE 2] Generating multi-label articles...")
        for labels, desc in combinazioni_multilabel:
            labels_str = ', '.join(labels)
            print(f"\nGenerating: {labels_str}...")
            for i in range(multilabel_batches):
                if i % 5 == 0:
                    print(f"  Batch {i}/{multilabel_batches} ({len(dataset)} articles so far)")
                
                if random.random() < 0.2:
                    via_scelta = random.choice(streets)
                    istruzioni_luogo = f"Ambientazione: Cita esplicitamente '{via_scelta}'."
                else:
                    istruzioni_luogo = "Ambientazione: Generica (es. 'in centro', 'quartiere periferico', 'zona industriale'). NON citare vie specifiche."

                stile = random.choice(stili_giornalistici)
                labels_json = json.dumps(labels)
                
                prompt = f"""
                Scrivi {ROWS_PER_BATCH} articoli di giornale locale di Bari in stile REALISTICO.
                L'articolo deve riportare un fatto che coinvolge PIÙ TIPOLOGIE DI REATO contemporaneamente:
                {desc}
                
                REGOLE IMPORTANTI PER REALISMO:
                - {stile}
                - L'articolo deve descrivere una situazione dove sono presenti ENTRAMBI i reati
                - NON tutti gli articoli devono menzionare esplicitamente i crimini nel titolo
                - Usa terminologia giornalistica italiana autentica
                - Varia la lunghezza: alcuni brevi (2-3 frasi), altri più dettagliati
                {istruzioni_luogo}
                
                IMPORTANTE: Rispondi SOLO con un array JSON valido, senza altro testo.
                Schema JSON: [{{"title": "...", "content": "...", "labels": {labels_json}}}]
                """
                
                try:
                    res = model_crimine.generate_content(prompt)
                    data = extract_json_from_response(res.text)
                    if data:
                        for x in data:
                            x['date'] = genera_data_recente()
                            x['link'] = genera_fake_link(x.get('title', ''))
                            x['labels'] = labels  # Use the multi-label combination
                            dataset.append(x)
                        consecutive_errors = 0
                    else:
                        print(f"    ⚠️ Invalid JSON for {labels_str} batch {i}")
                        consecutive_errors += 1
                    time.sleep(5)
                except Exception as e:
                    print(f"    Err {labels_str} batch {i}: {e}")
                    consecutive_errors += 1
                    time.sleep(10)
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"❌ Too many consecutive errors ({MAX_CONSECUTIVE_ERRORS}), saving and stopping...")
                    break
            
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                break
    
    save_dataset(dataset, output_dir, "dataset_crime.json")
    return dataset


def generate_non_crime_news(batches, streets, output_dir):
    """Generates non-crime news articles.

    :param batches: int: Number of batches to generate
    :param streets: list[str]: List of street names for location generation
    :param output_dir: Path: Directory where the output file will be saved
    :returns: list[dict]: List of generated non-crime article dictionaries

    """
    dataset = []
    consecutive_errors = 0
    print("\n--- GENERATING NON-CRIME NEWS ---")
    print(f"Batches: {batches}")
    print(f"Expected articles: ~{batches * ROWS_PER_BATCH}")
    
    for i in range(batches):
        topic = random.choices(lista_topics, weights=lista_pesi, k=1)[0]
        
        if random.random() < 0.2:
            via_scelta = random.choice(streets)
            istruzioni_luogo = f"Cita la via/zona specifica: {via_scelta}."
        else:
            istruzioni_luogo = "Non citare vie specifiche. Usa riferimenti generici (es. 'sul lungomare', 'in ateneo', 'negli uffici comunali', 'in piazza')."

        if i % 50 == 0: 
            print(f"  Batch {i}/{batches} (Topic: {topic[:30]}...) - {len(dataset)} articles")

        stile = random.choice(stili_giornalistici)
        
        prompt = f"""
        Scrivi {ROWS_PER_BATCH} articoli di giornale locale di Bari in stile REALISTICO.
        Argomento principale: {topic}.
        
        REGOLE IMPORTANTI PER REALISMO:
        - {stile}
        - Gli articoli devono sembrare autentici articoli di cronaca locale
        - Possono menzionare fatti di attualità, ma NON reati o crimini
        - Varia la lunghezza e il tono
        {istruzioni_luogo}
        
        IMPORTANTE: Rispondi SOLO con un array JSON valido, senza altro testo prima o dopo.
        Schema JSON: [{{"title": "...", "content": "...", "labels": []}}]
        """
        
        try:
            res = model_non_crimine.generate_content(prompt)
            data = extract_json_from_response(res.text)
            if data:
                for x in data:
                    x['date'] = genera_data_recente()
                    x['link'] = genera_fake_link(x.get('title', ''))
                    x['labels'] = []
                    dataset.append(x)
                consecutive_errors = 0
            else:
                print(f"    ⚠️ Invalid JSON for batch {i}")
                consecutive_errors += 1
            time.sleep(5) 
        except Exception as e:
            print(f"    Err batch {i}: {e}")
            consecutive_errors += 1
            time.sleep(10)
        
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(f"❌ Too many consecutive errors ({MAX_CONSECUTIVE_ERRORS}), saving and stopping...")
            break

    save_dataset(dataset, output_dir, "dataset_non_crime.json")
    return dataset


def generate_ambiguous_news(batches, output_dir):
    """Generates ambiguous news articles.

    :param batches: int: Number of batches to generate
    :param output_dir: Path: Directory where the output file will be saved
    :returns: list[dict]: List of generated ambiguous article dictionaries

    """
    dataset = []
    consecutive_errors = 0
    print("\n--- GENERATING AMBIGUOUS NEWS ---")
    print(f"Batches: {batches}")
    print(f"Expected articles: ~{batches * ROWS_PER_BATCH}")
    
    for i in range(batches):
        caso = random.choice(casi_ambigui)
        stile = random.choice(stili_giornalistici)
        
        if i % 10 == 0:
            print(f"  Batch {i}/{batches} - {len(dataset)} articles")
        
        prompt = f"""
        Scrivi {ROWS_PER_BATCH} articoli di giornale locale di Bari.
        L'articolo deve descrivere una situazione che SEMBRA un crimine ma NON lo è:
        Esempio: {caso}
        
        IMPORTANTE: L'articolo deve essere ambiguo - un lettore potrebbe inizialmente
        pensare che sia un crimine, ma leggendo si capisce che non lo è.
        {stile}
        
        IMPORTANTE: Rispondi SOLO con un array JSON valido, senza altro testo prima o dopo.
        Schema JSON: [{{"title": "...", "content": "...", "labels": []}}]
        """
        
        try:
            res = model_non_crimine.generate_content(prompt)
            data = extract_json_from_response(res.text)
            if data:
                for x in data:
                    x['date'] = genera_data_recente()
                    x['link'] = genera_fake_link(x.get('title', ''))
                    x['labels'] = []
                    dataset.append(x)
                consecutive_errors = 0
            else:
                print(f"    ⚠️ Invalid JSON for batch {i}")
                consecutive_errors += 1
            time.sleep(5)
        except Exception as e:
            print(f"    Err batch {i}: {e}")
            consecutive_errors += 1
            time.sleep(10)
        
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(f"❌ Too many consecutive errors ({MAX_CONSECUTIVE_ERRORS}), saving and stopping...")
            break

    save_dataset(dataset, output_dir, "dataset_ambiguous.json")
    return dataset


def generate(args):
    """Main generation function.

    :param args: argparse.Namespace: Parsed command line arguments with type, batches_crime, batches_non_crime, batches_ambiguous

    """
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "datasets" / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load streets
    streets = load_streets()
    
    # Print configuration
    print("\n" + "="*60)
    print("DATASET GENERATOR - GEMMA 3 12B IT")
    print("="*60)
    print(f"Type: {args.type}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {MODEL_NAME}")
    print("="*60)
    
    # Track generated datasets
    crime_data = []
    non_crime_data = []
    ambiguous_data = []
    
    # Generate based on type
    if args.type in ['crime', 'all']:
        crime_data = generate_crime_news(
            args.batches_crime, 
            streets, 
            output_dir,
            categories=args.categories,
            skip_multilabel=args.skip_multilabel
        )
    
    if args.type in ['non_crime', 'all']:
        non_crime_data = generate_non_crime_news(args.batches_non_crime, streets, output_dir)
    
    if args.type in ['ambiguous', 'all']:
        ambiguous_data = generate_ambiguous_news(args.batches_ambiguous, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    total = len(crime_data) + len(non_crime_data) + len(ambiguous_data)
    print(f"Total articles generated: {total}")
    if crime_data:
        print(f"  - Crime: {len(crime_data)}")
    if non_crime_data:
        print(f"  - Non-crime: {len(non_crime_data)}")
    if ambiguous_data:
        print(f"  - Ambiguous: {len(ambiguous_data)}")
    print(f"\nOutput saved to: {output_dir}")


def main():
    """Main function to parse command line arguments and run dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic newspaper articles dataset using Google Gemma 3 API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_dataset_gemma.py --type crime
    python generate_dataset_gemma.py --type crime --categories omicidio_colposo
    python generate_dataset_gemma.py --type crime --categories rapina aggressione --batches_crime 10
    python generate_dataset_gemma.py --type non_crime --batches_non_crime 100
    python generate_dataset_gemma.py --type all
        """
    )
    parser.add_argument('--type', '-t', type=str, required=True,
                        choices=['crime', 'non_crime', 'ambiguous', 'all'],
                        help='Type of articles to generate')
    parser.add_argument('--categories', '-c', type=str, nargs='+', default=None,
                        help='Specific crime categories to generate (e.g., omicidio_colposo rapina)')
    parser.add_argument('--skip_multilabel', '-sm', action='store_true',
                        help='Skip multi-label article generation')
    parser.add_argument('--batches_crime', '-bc', type=int, default=DEFAULT_BATCHES_CRIME,
                        help=f'Number of batches per crime category (default: {DEFAULT_BATCHES_CRIME})')
    parser.add_argument('--batches_non_crime', '-bn', type=int, default=DEFAULT_BATCHES_NON_CRIME,
                        help=f'Number of batches for non-crime news (default: {DEFAULT_BATCHES_NON_CRIME})')
    parser.add_argument('--batches_ambiguous', '-ba', type=int, default=DEFAULT_BATCHES_AMBIGUOUS,
                        help=f'Number of batches for ambiguous news (default: {DEFAULT_BATCHES_AMBIGUOUS})')
    
    # Print available categories if requested
    print(f"\nAvailable crime categories: {list(categorie_crimine.keys())}\n")
    
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
