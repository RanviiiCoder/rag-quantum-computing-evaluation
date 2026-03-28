from transformers import pipeline
from typing import List, Dict
import torch

class RAGPipeline:
    def __init__(self, retriever, model_name: str = "google/flan-t5-base"):
        self.retriever = retriever
        
        self.device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text2text-generation",
            model=model_name,
            device=self.device,
            max_length=200
        )
    
    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        context_parts = []
        for i, chunk_data in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Excerpt {i}]: {chunk_data['chunk']['content']}")
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""Answer the following question based on the provided context. If the answer cannot be found in the context, say "I cannot find this information in the available documents."

Context:
{context}

Question: {query}

Answer:"""
        
        result = self.generator(prompt, max_length=200, do_sample=True, temperature=0.1)
        answer = result[0]['generated_text'].strip()
        
        return answer
    
    def answer_question(self, query: str, top_k: int = 3) -> Dict:
        retrieved = self.retriever.retrieve(query, top_k)
        
        if not retrieved:
            return {
                'query': query,
                'answer': "No relevant information found in the documents.",
                'retrieved_chunks': [],
                'confidence': 0.0
            }
        
        context = self.format_context(retrieved)
        answer = self.generate_answer(query, context)
        
        avg_similarity = sum(r['similarity_score'] for r in retrieved) / len(retrieved)
        
        return {
            'query': query,
            'answer': answer,
            'retrieved_chunks': retrieved,
            'confidence': avg_similarity
        }