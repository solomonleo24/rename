import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_names(target_name, embeddings_dict, top_k=5):

    if target_name not in embeddings_dict:
        raise ValueError(f"{target_name} not found in embeddings.")
    
    target_emb = embeddings_dict[target_name].reshape(1, -1)
    
    names = []
    vectors = []
    for name, emb in embeddings_dict.items():
        if name != target_name:
            names.append(name)
            vectors.append(emb)
    
    vectors = np.vstack(vectors)  # shape (N, embedding_dim)
    
    similarities = cosine_similarity(target_emb, vectors)[0]  # shape (N,)
    
    # Pair names and similarities, sort by similarity descending
    name_sim_pairs = sorted(zip(names, similarities), key=lambda x: x[1], reverse=True)
    
    return name_sim_pairs[:top_k]