from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

# Load sentiment model & tokenizer once (not inside the loop)
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# Load semantic embedding model once
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(query):
    """
    query: dict where keys are names and values are lists of strings
    Returns a dict: name -> {semantic: np.array, sentiment: np.array}
    """
    trace = {}
    for q, texts in query.items():
        if texts:
            semantic = get_single_embedding_semantic(texts)
            sentiment = get_single_embedding_sentiment(texts)
            trace[q] = {
                'semantic': semantic,
                'sentiment': sentiment
            }

        print(f"Processed {q} with {len(texts)} texts")
    return trace

def get_single_embedding_semantic(texts):
    combined_text = " ".join(texts)
    emb = semantic_model.encode(combined_text)
    return np.array(emb)

def get_single_embedding_sentiment(texts):
    combined_text = " ".join(texts)

    # Tokenize with truncation so it fits model's 512-token limit
    inputs = sentiment_tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    return np.array(scores)  # [neg, neu, pos] or similar depending on model
