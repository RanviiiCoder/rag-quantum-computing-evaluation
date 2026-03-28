import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import re
from collections import Counter

class EvaluationMetrics:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
    
    def semantic_similarity(self, generated_answer: str, expected_answer: str) -> float:
        emb1 = self.embedding_model.encode([generated_answer])
        emb2 = self.embedding_model.encode([expected_answer])
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    def keyword_overlap_score(self, generated_answer: str, expected_answer: str) -> float:
        def extract_keywords(text: str) -> set:
            text = text.lower()
            words = re.findall(r'\b[a-z]{3,}\b', text)
            stopwords = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they'}
            return set(w for w in words if w not in stopwords)
        
        gen_keywords = extract_keywords(generated_answer)
        exp_keywords = extract_keywords(expected_answer)
        
        if not exp_keywords:
            return 0.0
        
        overlap = len(gen_keywords & exp_keywords)
        return overlap / len(exp_keywords)
    
    def factual_accuracy_score(self, generated_answer: str, expected_answer: str) -> int:
        gen_lower = generated_answer.lower()
        exp_lower = expected_answer.lower()
        
        facts = re.split(r'[.,;]', exp_lower)
        facts = [f.strip() for f in facts if len(f.strip()) > 10]
        
        if not facts:
            return 1
        
        found_facts = 0
        for fact in facts:
            if fact in gen_lower:
                found_facts += 1
        
        return found_facts / len(facts)
    
    def retrieval_precision(self, retrieved_chunks: List[Dict], question: str, 
                           relevant_chunks: List[str]) -> float:
        if not retrieved_chunks:
            return 0.0
        
        relevant_retrieved = 0
        for chunk_data in retrieved_chunks:
            chunk_content = chunk_data['chunk']['content']
            for relevant in relevant_chunks:
                if relevant in chunk_content or chunk_content in relevant:
                    relevant_retrieved += 1
                    break
        
        return relevant_retrieved / len(retrieved_chunks)
    
    def retrieval_recall(self, retrieved_chunks: List[Dict], question: str,
                        relevant_chunks: List[str]) -> float:
        if not relevant_chunks:
            return 1.0
        
        retrieved_contents = [c['chunk']['content'] for c in retrieved_chunks]
        relevant_retrieved = 0
        
        for relevant in relevant_chunks:
            for retrieved in retrieved_contents:
                if relevant in retrieved or retrieved in relevant:
                    relevant_retrieved += 1
                    break
        
        return relevant_retrieved / len(relevant_chunks)
    
    def compute_all_metrics(self, generated_answer: str, expected_answer: str,
                           retrieved_chunks: List[Dict], relevant_chunks: List[str]) -> Dict:
        semantic_sim = self.semantic_similarity(generated_answer, expected_answer)
        keyword_overlap = self.keyword_overlap_score(generated_answer, expected_answer)
        factual_accuracy = self.factual_accuracy_score(generated_answer, expected_answer)
        
        retrieval_precision = self.retrieval_precision(retrieved_chunks, generated_answer, relevant_chunks)
        retrieval_recall = self.retrieval_recall(retrieved_chunks, generated_answer, relevant_chunks)
        
        f1_score = 2 * (retrieval_precision * retrieval_recall) / (retrieval_precision + retrieval_recall + 1e-8)
        
        return {
            'semantic_similarity': semantic_sim,
            'keyword_overlap': keyword_overlap,
            'factual_accuracy': factual_accuracy,
            'retrieval_precision': retrieval_precision,
            'retrieval_recall': retrieval_recall,
            'retrieval_f1': f1_score
        }