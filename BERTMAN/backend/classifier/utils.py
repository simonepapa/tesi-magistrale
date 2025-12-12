import pandas as pd
import math
import torch
from sklearn.preprocessing import MinMaxScaler

def do_label(jsonFile, quartiere):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load JSON, make it a dataframe
    df = pd.DataFrame(jsonFile)
    df = df.drop(columns=['link', 'title', 'date'])
    df["omicidio"] = 0	
    df["omicidio_colposo"] = 0	
    df["omicidio_stradale"] = 0	
    df["tentato_omicidio"] = 0	
    df["furto"] = 0	
    df["rapina"] = 0	
    df["violenza_sessuale"] = 0	
    df["aggressione"] = 0	
    df["spaccio"] = 0	
    df["truffa"] = 0	
    df["estorsione"] = 0	
    df["contrabbando"] = 0	
    df["associazione_di_tipo_mafioso"] = 0

    # Needed to classify
    cols = df.columns.tolist()
    labels = cols[1:]
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    # Load model
    checkpoint = 'data/bert_fine_tuned_model'
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                            problem_type="multi_label_classification",
                                                            num_labels=len(labels),
                                                            id2label=id2label,
                                                            label2id=label2id)
    model.to(device)

    # Apply model to label articles
    sigmoid = torch.nn.Sigmoid()
    
    for article in jsonFile:
        content = article.get("content", "")
        python_id = {"python_id": quartiere}
        article.update(python_id)
        if content:
            # Tokenize content
            encoding = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
            encoding = {k: v.to(model.device) for k, v in encoding.items()}
    
            # Calculate probability
            with torch.no_grad():
                outputs = model(**encoding)
            logits = outputs.logits
            probs = sigmoid(logits.squeeze().cpu())
    
            # 1 if prob >75% else 0
            label_scores = {
                labels[idx]: {"value": int(prob > 0.75), "prob": round(prob.item(), 2)}
                for idx, prob in enumerate(probs.numpy())
            }
    
            #  Add labels
            article.update(label_scores)
        else:
            article.update({label: 0 for label in labels})  # No labels if no content

    return jsonFile

