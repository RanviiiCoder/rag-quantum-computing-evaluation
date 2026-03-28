import os
import argparse
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline
from src.evaluation_metrics import EvaluationMetrics
from src.evaluator import Evaluator

def build_rag_system(documents_dir, chunks_file=None, embeddings_file=None, rebuild=False):
    print("Initializing RAG System...")
    
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    embedding_manager = EmbeddingManager()
    retriever = Retriever(embedding_manager)
    
    if rebuild or not os.path.exists('data/processed_chunks.json'):
        print("Processing documents...")
        documents, chunks = processor.process_all_documents(documents_dir)
        processor.save_chunks(chunks, 'data/processed_chunks.json')
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        print("Generating embeddings...")
        embeddings = embedding_manager.generate_embeddings(chunks)
        embedding_manager.save_embeddings(embeddings, chunks, 
                                         'data/embeddings.npy', 
                                         'data/chunks.pkl')
        print("Embeddings generated and saved")
    else:
        print("Loading existing chunks and embeddings...")
        chunks = processor.load_chunks('data/processed_chunks.json')
        embeddings, chunks = embedding_manager.load_embeddings('data/embeddings.npy', 
                                                                'data/chunks.pkl')
        print(f"Loaded {len(chunks)} chunks")
    
    retriever.set_index(embeddings, chunks)
    rag_pipeline = RAGPipeline(retriever)
    
    return rag_pipeline

def interactive_mode(rag_pipeline):
    print("\nRAG System Interactive Mode")
    print("Type 'quit' to exit, 'eval' to run evaluation")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'quit':
            break
        elif query.lower() == 'eval':
            return
        
        result = rag_pipeline.answer_question(query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nConfidence: {result['confidence']:.3f}")
        print("\nRetrieved from:")
        for i, chunk_data in enumerate(result['retrieved_chunks'], 1):
            print(f"  {i}. Score: {chunk_data['similarity_score']:.3f}")
            print(f"     Preview: {chunk_data['chunk']['content'][:100]}...")

def run_evaluation(rag_pipeline, qa_dataset_path, relevant_chunks_file=None):
    print("\nRunning Evaluation...")
    
    metrics = EvaluationMetrics()
    evaluator = Evaluator(rag_pipeline, metrics)
    
    relevant_chunks_map = None
    if relevant_chunks_file and os.path.exists(relevant_chunks_file):
        import json
        with open(relevant_chunks_file, 'r') as f:
            relevant_chunks_map = json.load(f)
    
    results_df = evaluator.evaluate_qa_dataset(qa_dataset_path, relevant_chunks_map)
    
    evaluator.print_summary()
    evaluator.generate_qualitative_report('qualitative_report.txt')
    
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nResults saved to evaluation_results.csv")
    
    return evaluator

def main():
    parser = argparse.ArgumentParser(description='RAG System with Evaluation')
    parser.add_argument('--mode', choices=['interactive', 'evaluate', 'build'], 
                       default='interactive', help='Run mode')
    parser.add_argument('--rebuild', action='store_true', 
                       help='Rebuild chunks and embeddings')
    
    args = parser.parse_args()
    
    rag_pipeline = build_rag_system('data/documents', rebuild=args.rebuild)
    
    if args.mode == 'interactive':
        interactive_mode(rag_pipeline)
    elif args.mode == 'evaluate':
        run_evaluation(rag_pipeline, 'data/qa_dataset.json')
    elif args.mode == 'build':
        print("RAG system built successfully")

if __name__ == "__main__":
    main()