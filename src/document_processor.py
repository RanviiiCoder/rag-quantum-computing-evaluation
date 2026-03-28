import os
import json
from typing import List, Dict, Tuple
import hashlib

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, directory_path: str) -> List[Dict]:
        documents = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]
                    documents.append({
                        'id': doc_id,
                        'filename': filename,
                        'content': content
                    })
        return documents
    
    def chunk_document(self, content: str, doc_id: str) -> List[Dict]:
        chunks = []
        words = content.split()
        
        start = 0
        chunk_index = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_{chunk_index}",
                'content': chunk_text,
                'start_pos': start,
                'end_pos': min(end, len(words))
            })
            
            start = end - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def process_all_documents(self, directory_path: str) -> Tuple[List[Dict], List[Dict]]:
        documents = self.load_documents(directory_path)
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc['content'], doc['id'])
            all_chunks.extend(chunks)
        
        return documents, all_chunks
    
    def save_chunks(self, chunks: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    def load_chunks(self, filepath: str) -> List[Dict]:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)