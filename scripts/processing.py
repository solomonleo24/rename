from scripts.scraping import get_urls, extract_clean_article_texts, get_single_embedding_per_name

# Processing pipeline to handle multiple queries and return embeddings
def pipeline(query, max_results=5, delay=1, remove_numbers=False, model_name='all-MiniLM-L6-v2'):
    # Read txt file of queries and run the pipeline for each query
    with open(query, 'r') as file:
        queries = file.readlines()

    results = {}
    for q in queries:
        q = q.strip()
        if q:  # Skip empty lines
            embeddings = processing_pipeline_helper(q, max_results=max_results, delay=delay, remove_numbers=remove_numbers, model_name=model_name)
            results[q] = embeddings

        print(f"Processed {q}")

    return results

# Helper function to run the processing pipeline for a single query
def processing_pipeline_helper(query, max_results=5, delay=1, remove_numbers=False, model_name='all-MiniLM-L6-v2'):

    urls = get_urls(query, max_results=max_results, delay=delay)
    clean_texts = extract_clean_article_texts(urls, remove_numbers=remove_numbers)
    
    embeddings = get_single_embedding_per_name(clean_texts, model_name=model_name)
    
    return embeddings