from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import re
from time import sleep
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def get_urls(query, max_results=5, delay=1):
    urls = []
    with DDGS() as ddgs:
        results = ddgs.text(f'{query}+name', max_results=max_results)
        for r in results:
            if r.get("href") and query.lower() in r["href"].lower():
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

# Processing pipeline to handle multiple queries and return embeddings
def extract_texts(query, max_results=5, delay=1, remove_numbers=False, model_name='all-MiniLM-L6-v2'):
    # Read txt file of queries and run the pipeline for each query
    with open(query, 'r') as file:
        queries = file.readlines()

    trace = {}

    for q in queries:
        q = q.strip()
        if q:  # Skip empty lines
            texts = processing_pipeline_helper(q, max_results=max_results, delay=delay, remove_numbers=remove_numbers, model_name=model_name)
            trace[q] = texts

        print(f"Processed {q}")

    return trace

# Helper function to run the processing pipeline for a single query
def processing_pipeline_helper(query, max_results=5, delay=1, remove_numbers=False, model_name='all-MiniLM-L6-v2'):

    urls = get_urls(query, max_results=max_results, delay=delay)
    clean_texts = extract_clean_article_texts(urls, remove_numbers=remove_numbers)
    
    return clean_texts
