from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import pickle

class EmbeddingManager:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.chunks = None
    
    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        texts = [chunk['content'] for chunk in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self.chunks = chunks
        return self.embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict], 
                       embeddings_path: str, chunks_path: str):
        np.save(embeddings_path, embeddings)
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
    
    def load_embeddings(self, embeddings_path: str, chunks_path: str) -> tuple[np.ndarray, List[Dict]]:
        self.embeddings = np.load(embeddings_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        return self.embeddings, self.chunks
    
    def compute_query_embedding(self, query: str) -> np.ndarray:
        return self.model.encode([query])[0]