from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import re
from time import sleep
from sentence_transformers import SentenceTransformer

def get_urls(query, max_results=5, delay=1):
    urls = []
    with DDGS() as ddgs:
        results = ddgs.text(f'{query}+name', max_results=max_results)
        for r in results:
            if r.get("href") and 'sully' in r["href"].lower():
                urls.append(r["href"])

            sleep(delay)
    return urls

def extract_clean_article_texts(url_list, remove_numbers=False):
    clean_texts = []
    
    for url in url_list:
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            
            # NLP cleaning
            text = text.lower()
            text = re.sub(r"[^\w\s]", " ", text)
            if remove_numbers:
                text = re.sub(r"\d+", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            
            if text != "":
                clean_texts.append(text)
        
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    return clean_texts

def get_single_embedding_per_name(texts, model_name='all-MiniLM-L6-v2'):

    model = SentenceTransformer(model_name)

    combined_text = " ".join(texts)  # concatenate all texts for that name
    # Optionally truncate combined_text here if too long
    emb = model.encode(combined_text)

    return emb
