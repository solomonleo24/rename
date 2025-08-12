import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_names(target_name, embeddings_dict, top_k=5, w=0.5):

    if target_name not in embeddings_dict:
        raise ValueError(f"{target_name} not found in embeddings.")
    
    target_emb_sem = np.array(embeddings_dict[target_name]['semantic']).reshape(1, -1)
    target_emb_sen = np.array(embeddings_dict[target_name]['sentiment']).reshape(1, -1)
    
    names = []
    vectors_sem, vectors_sen = [], []
    for name, emb in embeddings_dict.items():
        if name != target_name:
            names.append(name)
            vectors_sem.append(emb['semantic'])
            vectors_sen.append(emb['sentiment'])
    
    vectors_sem = np.vstack(vectors_sem)  # shape (N, embedding_dim)
    vectors_sen = np.vstack(vectors_sen)  # shape (N, embedding_dim)

    similarities_sem = cosine_similarity(target_emb_sem, vectors_sem)[0]  # shape (N,)
    similarities_sen = cosine_similarity(target_emb_sen, vectors_sen)[0]  # shape (N,)
    
    # Combine similarities (you can adjust the weighting if needed)
    similarities = w*similarities_sem + (1-w)*similarities_sen
    
    # Pair names and similarities, sort by similarity descending
    name_sim_pairs = sorted(zip(names, similarities), key=lambda x: x[1], reverse=True)
    
    return name_sim_pairs[:top_k]