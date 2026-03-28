import json
from typing import List, Dict
import pandas as pd
from datetime import datetime

class Evaluator:
    def __init__(self, rag_pipeline, evaluation_metrics):
        self.rag_pipeline = rag_pipeline
        self.metrics = evaluation_metrics
        self.results = []
    
    def evaluate_qa_dataset(self, qa_dataset_path: str, relevant_chunks_map: Dict = None) -> pd.DataFrame:
        with open(qa_dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        results_list = []
        
        for item in dataset['questions']:
            q_id = item['id']
            question = item['question']
            expected_answer = item['expected_answer']
            
            print(f"Evaluating question {q_id}: {question[:50]}...")
            
            rag_result = self.rag_pipeline.answer_question(question)
            generated_answer = rag_result['answer']
            retrieved_chunks = rag_result['retrieved_chunks']
            
            relevant_chunks = []
            if relevant_chunks_map and str(q_id) in relevant_chunks_map:
                relevant_chunks = relevant_chunks_map[str(q_id)]
            
            metrics = self.metrics.compute_all_metrics(
                generated_answer, expected_answer, retrieved_chunks, relevant_chunks
            )
            
            results_list.append({
                'question_id': q_id,
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'confidence': rag_result['confidence'],
                **metrics
            })
        
        self.results = results_list
        return pd.DataFrame(results_list)
    
    def generate_qualitative_report(self, output_file: str = None):
        if not self.results:
            print("No evaluation results available. Run evaluate_qa_dataset first.")
            return
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("QUALITATIVE ASSESSMENT REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Questions Evaluated: {len(self.results)}")
        report_lines.append("")
        
        for idx, result in enumerate(self.results, 1):
            report_lines.append(f"\n{'='*60}")
            report_lines.append(f"QUESTION {idx}: {result['question']}")
            report_lines.append(f"\nEXPECTED ANSWER:")
            report_lines.append(f"{result['expected_answer'][:200]}...")
            report_lines.append(f"\nGENERATED ANSWER:")
            report_lines.append(f"{result['generated_answer']}")
            report_lines.append(f"\nMETRICS SUMMARY:")
            report_lines.append(f"  - Semantic Similarity: {result['semantic_similarity']:.3f}")
            report_lines.append(f"  - Keyword Overlap: {result['keyword_overlap']:.3f}")
            report_lines.append(f"  - Factual Accuracy: {result['factual_accuracy']:.3f}")
            report_lines.append(f"  - Confidence: {result['confidence']:.3f}")
            
            report_lines.append("\n" + "-" * 40)
            report_lines.append("QUALITATIVE SCORING RUBRIC (1-5):")
            report_lines.append("  - Coherence (1-5): ___")
            report_lines.append("  - Completeness (1-5): ___")
            report_lines.append("  - Factual Correctness (1-5): ___")
            report_lines.append("  - Notes: _________________")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Qualitative assessment report saved to {output_file}")
        
        print(report)
        return report
    
    def generate_quantitative_summary(self) -> Dict:
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            'total_questions': len(df),
            'avg_semantic_similarity': df['semantic_similarity'].mean(),
            'avg_keyword_overlap': df['keyword_overlap'].mean(),
            'avg_factual_accuracy': df['factual_accuracy'].mean(),
            'avg_retrieval_precision': df['retrieval_precision'].mean(),
            'avg_retrieval_recall': df['retrieval_recall'].mean(),
            'avg_retrieval_f1': df['retrieval_f1'].mean(),
            'avg_confidence': df['confidence'].mean(),
            'std_semantic_similarity': df['semantic_similarity'].std(),
            'std_keyword_overlap': df['keyword_overlap'].std()
        }
        
        return summary
    
    def print_summary(self):
        summary = self.generate_quantitative_summary()
        
        print("\n" + "="*60)
        print("QUANTITATIVE EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions: {summary['total_questions']}")
        print(f"\nGeneration Metrics:")
        print(f"  - Average Semantic Similarity: {summary['avg_semantic_similarity']:.3f}")
        print(f"  - Average Keyword Overlap: {summary['avg_keyword_overlap']:.3f}")
        print(f"  - Average Factual Accuracy: {summary['avg_factual_accuracy']:.3f}")
        print(f"\nRetrieval Metrics:")
        print(f"  - Average Precision: {summary['avg_retrieval_precision']:.3f}")
        print(f"  - Average Recall: {summary['avg_retrieval_recall']:.3f}")
        print(f"  - Average F1 Score: {summary['avg_retrieval_f1']:.3f}")
        print(f"\nSystem Confidence:")
        print(f"  - Average Confidence: {summary['avg_confidence']:.3f}")
        print("="*60)