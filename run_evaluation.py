from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline
from src.evaluation_metrics import EvaluationMetrics
from src.evaluator import Evaluator
import os

def main():
    
    print("RAG SYSTEM EVALUATION")
    
    
    print("\nLoading RAG system...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    embedding_manager = EmbeddingManager()
    retriever = Retriever(embedding_manager)
    
    chunks = processor.load_chunks('data/processed_chunks.json')
    embeddings, chunks = embedding_manager.load_embeddings('data/embeddings.npy', 
                                                            'data/chunks.pkl')
    retriever.set_index(embeddings, chunks)
    rag_pipeline = RAGPipeline(retriever)
    
    print("\nInitializing evaluation framework...")
    metrics = EvaluationMetrics()
    evaluator = Evaluator(rag_pipeline, metrics)
    
    print("\nRunning evaluation on QA dataset...")
    results_df = evaluator.evaluate_qa_dataset('data/qa_dataset.json')
    
    evaluator.print_summary()
    evaluator.generate_qualitative_report('qualitative_report.txt')
    
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nDetailed results saved to evaluation_results.csv")
    print("Qualitative assessment template saved to qualitative_report.txt")
    
    print("\nSample of generated answers:")
    print("-"*60)
    for idx, row in results_df.head(3).iterrows():
        print(f"\nQ: {row['question'][:80]}...")
        print(f"A: {row['generated_answer'][:150]}...")
        print(f"Semantic Similarity: {row['semantic_similarity']:.3f}")
        print(f"Factual Accuracy: {row['factual_accuracy']:.3f}")
        print("-"*40)

if __name__ == "__main__":
    main()