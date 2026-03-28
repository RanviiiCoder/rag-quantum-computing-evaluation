import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        self.chunks = None
        self.embeddings = None
    
    def set_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        self.embeddings = embeddings
        self.chunks = chunks
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.embeddings is None or self.chunks is None:
            raise ValueError("Index not set. Call set_index first.")
        
        query_embedding = self.embedding_manager.compute_query_embedding(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[idx],
                'similarity_score': float(similarities[idx])
            })
        
        return results
    
    def retrieve_with_scores(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        results = self.retrieve(query, top_k)
        return [(r['chunk'], r['similarity_score']) for r in results]