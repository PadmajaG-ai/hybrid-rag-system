"""
COMPLETE RAG Evaluation with Method Comparison, Hallucination Detection & Error Analysis
========================================================================================

Features:
- Dense vs Sparse vs Hybrid comparison
- Hallucination detection
- Error categorization (6 categories)
- Comprehensive metrics

Usage:
    python evaluate_rag_methods.py qa_pairs.json --output evaluation_results.json
"""

import json
import argparse
import sys
from typing import Dict, List, Tuple
from datetime import datetime
import time
import re

try:
    from hybrid_rag_system import HybridRAGSystem
    from sentence_transformers import SentenceTransformer
    from rouge_score import rouge_scorer
    import numpy as np
except ImportError as e:
    print(f" Missing dependency: {e}")
    sys.exit(1)


class ComprehensiveRAGEvaluator:
    """
    Complete evaluator with:
    - Method comparison (Dense/Sparse/Hybrid)
    - Hallucination detection
    - Error analysis
    """
    
    def __init__(self, rag_system: HybridRAGSystem):
        self.rag_system = rag_system
        
        # Load semantic similarity model
        print("\n Loading semantic similarity model...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ROUGE scorer
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
        print(" Evaluator ready with full feature set!")
    
    def detect_hallucination(self, answer: str, context_chunks: List[str]) -> Dict:
        """
        Detect if answer contains hallucinations
        
        Returns:
            Dict with hallucination score, grounding score, and faithful flag
        """
        
        if not answer or not context_chunks:
            return {
                'is_hallucination': True,
                'grounding_score': 0.0,
                'is_faithful': False,
                'explanation': 'Empty answer or no context'
            }
        
        # Combine all context
        full_context = " ".join(context_chunks)
        
        # Calculate semantic overlap
        answer_embedding = self.semantic_model.encode([answer])[0]
        context_embedding = self.semantic_model.encode([full_context])[0]
        
        grounding_score = float(np.dot(answer_embedding, context_embedding) / 
                               (np.linalg.norm(answer_embedding) * np.linalg.norm(context_embedding)))
        
        # Check for specific hallucination patterns
        hallucination_indicators = [
            len(answer) > 5 * len(full_context),  # Answer much longer than context
            grounding_score < 0.3,  # Very low semantic overlap
            self._contains_unsupported_claims(answer, full_context)
        ]
        
        is_hallucination = sum(hallucination_indicators) >= 2
        is_faithful = grounding_score >= 0.5 and not is_hallucination
        
        return {
            'is_hallucination': is_hallucination,
            'grounding_score': float(grounding_score),
            'is_faithful': is_faithful,
            'explanation': self._get_hallucination_explanation(grounding_score, is_hallucination)
        }
    
    def _contains_unsupported_claims(self, answer: str, context: str) -> bool:
        """Check for claims not supported by context"""
        # Simple heuristic: check for specific numerical claims
        answer_numbers = set(re.findall(r'\b\d+\.?\d*\b', answer))
        context_numbers = set(re.findall(r'\b\d+\.?\d*\b', context))
        
        # If answer has numbers not in context, might be hallucination
        unsupported = answer_numbers - context_numbers
        return len(unsupported) > 3
    
    def _get_hallucination_explanation(self, grounding_score: float, is_hallucination: bool) -> str:
        """Get explanation for hallucination detection"""
        if is_hallucination:
            if grounding_score < 0.3:
                return "Very low grounding - answer not supported by context"
            else:
                return "Contains unsupported claims or fabricated details"
        else:
            return "Well-grounded in retrieved context"
    
    def categorize_error(self, result: Dict) -> Dict:
        """
        Categorize errors into 6 types:
        1. Retrieval Failure
        2. Incomplete Answer
        3. Hallucination
        4. Over-generalization
        5. Factual Error
        6. Successful
        """
        
        mrr = result['mrr_score']
        f1 = result['f1_score']
        grounding = result['grounding_score']
        is_hallucination = result.get('is_hallucination', False)
        answer_length = len(result.get('generated_answer', '').split())
        reference_length = len(result.get('ground_truth', '').split())
        
        # Error categorization logic
        if mrr == 0.0:
            category = 'retrieval_failure'
            explanation = 'Failed to retrieve relevant documents'
        elif is_hallucination:
            category = 'hallucination'
            explanation = 'Generated content not grounded in retrieved context'
        elif f1 < 0.3 and answer_length < reference_length * 0.5:
            category = 'incomplete'
            explanation = 'Answer is incomplete, missing key information'
        elif grounding < 0.4:
            category = 'over_generalization'
            explanation = 'Answer is too general, lacks specific details from context'
        elif f1 < 0.5:
            category = 'factual_error'
            explanation = 'Contains factual errors or inaccuracies'
        else:
            category = 'successful'
            explanation = 'High quality answer with good metrics'
        
        # Determine severity
        if category == 'successful':
            severity = 'none'
        elif category in ['retrieval_failure', 'hallucination']:
            severity = 'critical'
        elif category in ['factual_error', 'incomplete']:
            severity = 'major'
        else:
            severity = 'minor'
        
        return {
            'category': category,
            'severity': severity,
            'explanation': explanation,
            'metrics': {
                'mrr': float(mrr),
                'f1': float(f1),
                'grounding': float(grounding)
            }
        }
    
    def evaluate_all_methods(self, qa_pairs: List[Dict], top_k: int = 20, top_n: int = 10):
        """
        Complete evaluation with all features
        """
        
        print("\n" + "="*70)
        print("🔬 COMPREHENSIVE RAG EVALUATION")
        print("="*70)
        print("Features: Method Comparison | Hallucination Detection | Error Analysis")
        print("="*70 + "\n")
        
        per_question_results = []
        
        # Track metrics per method
        method_metrics = {
            'dense': {'mrr_scores': [], 'f1_scores': [], 'rouge_scores': [], 
                     'semantic_scores': [], 'grounding_scores': []},
            'sparse': {'mrr_scores': [], 'f1_scores': [], 'rouge_scores': [], 
                      'semantic_scores': [], 'grounding_scores': []},
            'hybrid': {'mrr_scores': [], 'f1_scores': [], 'rouge_scores': [], 
                      'semantic_scores': [], 'grounding_scores': []}
        }
        
        # Track error categories
        error_categories = {
            'retrieval_failure': 0,
            'incomplete': 0,
            'hallucination': 0,
            'over_generalization': 0,
            'factual_error': 0,
            'successful': 0
        }
        
        # Track hallucinations
        hallucination_count = 0
        faithful_count = 0
        
        for i, qa in enumerate(qa_pairs, 1):
            question = qa['question']
            ground_truth = qa['answer']
            source_url = qa.get('source_url', '')
            
            print(f"\n[{i}/{len(qa_pairs)}] Evaluating: {question[:60]}...")
            
            # === RETRIEVE WITH ALL THREE METHODS ===
            
            # 1. Dense-only retrieval
            dense_results = self.rag_system.dense_retrieval(question, top_k=top_k)
            
            # 2. Sparse-only retrieval
            sparse_results = self.rag_system.sparse_retrieval(question, top_k=top_k)
            
            # 3. Hybrid (RRF) retrieval
            hybrid_search_results = self.rag_system.hybrid_search(
                question, 
                top_k_per_method=top_k, 
                top_n_final=top_n
            )
            hybrid_results = hybrid_search_results['hybrid_results']
            
            # === GENERATE ANSWER ===
            context = self.rag_system.get_context_for_llm(hybrid_results, max_tokens=2000)
            generated_answer = self.rag_system.generate_answer(question, context)
            
            # === HALLUCINATION DETECTION ===
            context_chunks = [r['text'] for r in hybrid_results]
            hallucination_result = self.detect_hallucination(generated_answer, context_chunks)
            
            if hallucination_result['is_hallucination']:
                hallucination_count += 1
            if hallucination_result['is_faithful']:
                faithful_count += 1
            
            # === CALCULATE METRICS FOR EACH METHOD ===
            
            # Dense metrics
            dense_mrr = self._calculate_mrr(dense_results, source_url)
            dense_f1 = self._calculate_f1(generated_answer, ground_truth)
            dense_rouge = self._calculate_rouge(generated_answer, ground_truth)
            dense_semantic = self._calculate_semantic_similarity(generated_answer, ground_truth)
            dense_grounding = hallucination_result['grounding_score']
            
            # Sparse metrics
            sparse_mrr = self._calculate_mrr(sparse_results, source_url)
            sparse_f1 = self._calculate_f1(generated_answer, ground_truth)
            sparse_rouge = self._calculate_rouge(generated_answer, ground_truth)
            sparse_semantic = self._calculate_semantic_similarity(generated_answer, ground_truth)
            sparse_grounding = hallucination_result['grounding_score']
            
            # Hybrid metrics
            hybrid_mrr = self._calculate_mrr(hybrid_results, source_url)
            hybrid_f1 = self._calculate_f1(generated_answer, ground_truth)
            hybrid_rouge = self._calculate_rouge(generated_answer, ground_truth)
            hybrid_semantic = self._calculate_semantic_similarity(generated_answer, ground_truth)
            hybrid_grounding = hallucination_result['grounding_score']
            
            # === ERROR CATEGORIZATION ===
            error_cat = self.categorize_error({
                'mrr_score': hybrid_mrr,
                'f1_score': hybrid_f1,
                'grounding_score': hybrid_grounding,
                'is_hallucination': hallucination_result['is_hallucination'],
                'generated_answer': generated_answer,
                'ground_truth': ground_truth
            })
            
            error_categories[error_cat['category']] += 1
            
            # Store aggregated metrics
            method_metrics['dense']['mrr_scores'].append(dense_mrr)
            method_metrics['dense']['f1_scores'].append(dense_f1)
            method_metrics['dense']['rouge_scores'].append(dense_rouge)
            method_metrics['dense']['semantic_scores'].append(dense_semantic)
            method_metrics['dense']['grounding_scores'].append(dense_grounding)
            
            method_metrics['sparse']['mrr_scores'].append(sparse_mrr)
            method_metrics['sparse']['f1_scores'].append(sparse_f1)
            method_metrics['sparse']['rouge_scores'].append(sparse_rouge)
            method_metrics['sparse']['semantic_scores'].append(sparse_semantic)
            method_metrics['sparse']['grounding_scores'].append(sparse_grounding)
            
            method_metrics['hybrid']['mrr_scores'].append(hybrid_mrr)
            method_metrics['hybrid']['f1_scores'].append(hybrid_f1)
            method_metrics['hybrid']['rouge_scores'].append(hybrid_rouge)
            method_metrics['hybrid']['semantic_scores'].append(hybrid_semantic)
            method_metrics['hybrid']['grounding_scores'].append(hybrid_grounding)
            
            # Store per-question result
            question_result = {
                'question': question,
                'ground_truth_answer': ground_truth,
                'generated_answer': generated_answer,
                'source_url': source_url,
                
                # Dense-only scores
                'dense': {
                    'mrr_score': float(dense_mrr),
                    'f1_score': float(dense_f1),
                    'rouge_l': float(dense_rouge),
                    'semantic_similarity': float(dense_semantic),
                    'grounding_score': float(dense_grounding),
                    'rank': self._find_url_rank(dense_results, source_url),
                    'top_3_urls': [r['metadata']['url'] for r in dense_results[:3]]
                },
                
                # Sparse-only scores
                'sparse': {
                    'mrr_score': float(sparse_mrr),
                    'f1_score': float(sparse_f1),
                    'rouge_l': float(sparse_rouge),
                    'semantic_similarity': float(sparse_semantic),
                    'grounding_score': float(sparse_grounding),
                    'rank': self._find_url_rank(sparse_results, source_url),
                    'top_3_urls': [r['metadata']['url'] for r in sparse_results[:3]]
                },
                
                # Hybrid (RRF) scores
                'hybrid': {
                    'mrr_score': float(hybrid_mrr),
                    'f1_score': float(hybrid_f1),
                    'rouge_l': float(hybrid_rouge),
                    'semantic_similarity': float(hybrid_semantic),
                    'grounding_score': float(hybrid_grounding),
                    'rank': self._find_url_rank(hybrid_results, source_url),
                    'top_3_urls': [r['metadata']['url'] for r in hybrid_results[:3]]
                },
                
                # Best method for each metric
                'best_method': {
                    'mrr': self._find_best_method(dense_mrr, sparse_mrr, hybrid_mrr),
                    'f1': self._find_best_method(dense_f1, sparse_f1, hybrid_f1),
                    'rouge_l': self._find_best_method(dense_rouge, sparse_rouge, hybrid_rouge),
                    'semantic': self._find_best_method(dense_semantic, sparse_semantic, hybrid_semantic),
                    'grounding': self._find_best_method(dense_grounding, sparse_grounding, hybrid_grounding),
                },
                
                # Hallucination detection
                'hallucination_detection': hallucination_result,
                
                # Error categorization
                'error_analysis': error_cat,
                
                # Retrieved chunks
                'retrieved_chunks': hybrid_results[:3]
            }
            
            per_question_results.append(question_result)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(qa_pairs)} complete ({i/len(qa_pairs)*100:.1f}%)")
        
        # Calculate method comparison
        method_comparison = self._calculate_method_comparison(method_metrics)
        
        # Calculate error analysis summary
        total_questions = len(qa_pairs)
        error_analysis_summary = {
            'total_questions': total_questions,
            'error_distribution': {
                cat: {
                    'count': count,
                    'percentage': float(count / total_questions * 100)
                }
                for cat, count in error_categories.items()
            },
            'hallucination_stats': {
                'total_hallucinations': hallucination_count,
                'hallucination_rate': float(hallucination_count / total_questions * 100),
                'faithful_answers': faithful_count,
                'faithful_rate': float(faithful_count / total_questions * 100)
            },
            'success_rate': float(error_categories['successful'] / total_questions * 100)
        }
        
        # Print summary
        self._print_method_comparison(method_comparison)
        self._print_error_analysis(error_analysis_summary)
        
        return per_question_results, method_comparison, error_analysis_summary
    
    def _calculate_mrr(self, results: List[Dict], target_url: str) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, result in enumerate(results, 1):
            if result['metadata']['url'] == target_url:
                return 1.0 / i
        return 0.0
    
    def _find_url_rank(self, results: List[Dict], target_url: str) -> int:
        """Find rank of target URL"""
        for i, result in enumerate(results, 1):
            if result['metadata']['url'] == target_url:
                return i
        return -1
    
    def _calculate_f1(self, generated: str, reference: str) -> float:
        """Calculate F1 score"""
        gen_tokens = set(generated.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        common = gen_tokens & ref_tokens
        if not common:
            return 0.0
        
        precision = len(common) / len(gen_tokens)
        recall = len(common) / len(ref_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_rouge(self, generated: str, reference: str) -> float:
        """Calculate ROUGE-L score"""
        scores = self.rouge.score(reference, generated)
        return scores['rougeL'].fmeasure
    
    def _calculate_semantic_similarity(self, generated: str, reference: str) -> float:
        """Calculate semantic similarity"""
        gen_emb = self.semantic_model.encode([generated])[0]
        ref_emb = self.semantic_model.encode([reference])[0]
        
        similarity = np.dot(gen_emb, ref_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(ref_emb))
        return float(similarity)
    
    def _find_best_method(self, dense_score: float, sparse_score: float, hybrid_score: float) -> str:
        """Find which method performed best"""
        scores = {'dense': dense_score, 'sparse': sparse_score, 'hybrid': hybrid_score}
        return max(scores, key=scores.get)
    
    def _calculate_method_comparison(self, method_metrics: Dict) -> Dict:
        """Calculate aggregate comparison between methods"""
        
        comparison = {}
        
        for method in ['dense', 'sparse', 'hybrid']:
            comparison[method] = {
                'mrr': {
                    'mean': float(np.mean(method_metrics[method]['mrr_scores'])),
                    'std': float(np.std(method_metrics[method]['mrr_scores'])),
                    'median': float(np.median(method_metrics[method]['mrr_scores']))
                },
                'f1': {
                    'mean': float(np.mean(method_metrics[method]['f1_scores'])),
                    'std': float(np.std(method_metrics[method]['f1_scores'])),
                    'median': float(np.median(method_metrics[method]['f1_scores']))
                },
                'rouge_l': {
                    'mean': float(np.mean(method_metrics[method]['rouge_scores'])),
                    'std': float(np.std(method_metrics[method]['rouge_scores'])),
                    'median': float(np.median(method_metrics[method]['rouge_scores']))
                },
                'semantic_similarity': {
                    'mean': float(np.mean(method_metrics[method]['semantic_scores'])),
                    'std': float(np.std(method_metrics[method]['semantic_scores'])),
                    'median': float(np.median(method_metrics[method]['semantic_scores']))
                },
                'grounding': {
                    'mean': float(np.mean(method_metrics[method]['grounding_scores'])),
                    'std': float(np.std(method_metrics[method]['grounding_scores'])),
                    'median': float(np.median(method_metrics[method]['grounding_scores']))
                }
            }
        
        # Calculate improvement percentages
        comparison['improvements'] = {
            'hybrid_vs_dense': {
                'mrr': ((comparison['hybrid']['mrr']['mean'] - comparison['dense']['mrr']['mean']) 
                       / comparison['dense']['mrr']['mean'] * 100) if comparison['dense']['mrr']['mean'] > 0 else 0,
                'f1': ((comparison['hybrid']['f1']['mean'] - comparison['dense']['f1']['mean']) 
                      / comparison['dense']['f1']['mean'] * 100) if comparison['dense']['f1']['mean'] > 0 else 0,
            },
            'hybrid_vs_sparse': {
                'mrr': ((comparison['hybrid']['mrr']['mean'] - comparison['sparse']['mrr']['mean']) 
                       / comparison['sparse']['mrr']['mean'] * 100) if comparison['sparse']['mrr']['mean'] > 0 else 0,
                'f1': ((comparison['hybrid']['f1']['mean'] - comparison['sparse']['f1']['mean']) 
                      / comparison['sparse']['f1']['mean'] * 100) if comparison['sparse']['f1']['mean'] > 0 else 0,
            }
        }
        
        return comparison
    
    def _print_method_comparison(self, comparison: Dict):
        """Print method comparison summary"""
        
        print("\n" + "="*70)
        print("📊 METHOD COMPARISON SUMMARY")
        print("="*70)
        
        print("\n1️⃣  DENSE-ONLY RETRIEVAL")
        print(f"   MRR:      {comparison['dense']['mrr']['mean']:.4f} (±{comparison['dense']['mrr']['std']:.4f})")
        print(f"   F1:       {comparison['dense']['f1']['mean']:.4f} (±{comparison['dense']['f1']['std']:.4f})")
        print(f"   ROUGE-L:  {comparison['dense']['rouge_l']['mean']:.4f} (±{comparison['dense']['rouge_l']['std']:.4f})")
        
        print("\n2️⃣  SPARSE-ONLY RETRIEVAL (BM25)")
        print(f"   MRR:      {comparison['sparse']['mrr']['mean']:.4f} (±{comparison['sparse']['mrr']['std']:.4f})")
        print(f"   F1:       {comparison['sparse']['f1']['mean']:.4f} (±{comparison['sparse']['f1']['std']:.4f})")
        print(f"   ROUGE-L:  {comparison['sparse']['rouge_l']['mean']:.4f} (±{comparison['sparse']['rouge_l']['std']:.4f})")
        
        print("\n3️⃣  HYBRID (RRF FUSION)")
        print(f"   MRR:      {comparison['hybrid']['mrr']['mean']:.4f} (±{comparison['hybrid']['mrr']['std']:.4f})")
        print(f"   F1:       {comparison['hybrid']['f1']['mean']:.4f} (±{comparison['hybrid']['f1']['std']:.4f})")
        print(f"   ROUGE-L:  {comparison['hybrid']['rouge_l']['mean']:.4f} (±{comparison['hybrid']['rouge_l']['std']:.4f})")
        
        print("\n📈 IMPROVEMENT ANALYSIS")
        print(f"   Hybrid vs Dense:")
        print(f"      MRR: {comparison['improvements']['hybrid_vs_dense']['mrr']:+.2f}%")
        print(f"      F1:  {comparison['improvements']['hybrid_vs_dense']['f1']:+.2f}%")
        print(f"   Hybrid vs Sparse:")
        print(f"      MRR: {comparison['improvements']['hybrid_vs_sparse']['mrr']:+.2f}%")
        print(f"      F1:  {comparison['improvements']['hybrid_vs_sparse']['f1']:+.2f}%")
        
        print("\n" + "="*70)
    
    def _print_error_analysis(self, error_analysis: Dict):
        """Print error analysis summary"""
        
        print("\n" + "="*70)
        print(" ERROR ANALYSIS SUMMARY")
        print("="*70)
        
        dist = error_analysis['error_distribution']
        
        print("\nError Distribution:")
        print(f"   Successful:           {dist['successful']['count']:3d} ({dist['successful']['percentage']:5.1f}%)")
        print(f"   Retrieval Failure:    {dist['retrieval_failure']['count']:3d} ({dist['retrieval_failure']['percentage']:5.1f}%)")
        print(f"   Incomplete Answer:    {dist['incomplete']['count']:3d} ({dist['incomplete']['percentage']:5.1f}%)")
        print(f"   Hallucination:        {dist['hallucination']['count']:3d} ({dist['hallucination']['percentage']:5.1f}%)")
        print(f"   Over-generalization:  {dist['over_generalization']['count']:3d} ({dist['over_generalization']['percentage']:5.1f}%)")
        print(f"   Factual Error:        {dist['factual_error']['count']:3d} ({dist['factual_error']['percentage']:5.1f}%)")
        
        hall_stats = error_analysis['hallucination_stats']
        
        print(f"\nHallucination Detection:")
        print(f"   Total Hallucinations: {hall_stats['total_hallucinations']} ({hall_stats['hallucination_rate']:.1f}%)")
        print(f"   Faithful Answers: {hall_stats['faithful_answers']} ({hall_stats['faithful_rate']:.1f}%)")
        
        print(f"\nOverall Success Rate: {error_analysis['success_rate']:.1f}%")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Complete RAG Evaluation")
    parser.add_argument('qa_file', help="Q&A pairs JSON file")
    parser.add_argument('--output', default='evaluation_results.json', help="Output file")
    parser.add_argument('--top-k', type=int, default=20, help="Top-K per method")
    parser.add_argument('--top-n', type=int, default=10, help="Final top-N")
    
    args = parser.parse_args()
    
    # Load Q&A pairs
    print(f"\n Loading Q&A pairs from {args.qa_file}...")
    with open(args.qa_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both formats
    if isinstance(data, dict):
        if 'qa_pairs' in data:
            qa_pairs = data['qa_pairs']
            print(f" Extracted qa_pairs from dictionary structure")
        else:
            print(f" Error: Dictionary found but no 'qa_pairs' key")
            return
    else:
        qa_pairs = data

    print(f" Loaded {len(qa_pairs)} Q&A pairs")
    
    # Initialize RAG system
    print("\n Initializing Hybrid RAG System...")
    rag_system = HybridRAGSystem(
        chroma_collection_name="wikipedia_corpus",
        bm25_index_file="bm25_index.pkl"
    )
    
    # Initialize evaluator
    evaluator = ComprehensiveRAGEvaluator(rag_system)
    
    # Run evaluation
    start_time = time.time()
    per_question_results, method_comparison, error_analysis = evaluator.evaluate_all_methods(
        qa_pairs,
        top_k=args.top_k,
        top_n=args.top_n
    )
    elapsed_time = time.time() - start_time
    
    # Build results structure
    results = {
        'summary': {
            'total_questions': len(qa_pairs),
            'evaluation_date': datetime.now().isoformat(),
            'top_k': args.top_k,
            'top_n': args.top_n,
            'elapsed_time_seconds': elapsed_time
        },
        'method_comparison': method_comparison,
        'error_analysis': error_analysis,
        'per_question_results': per_question_results
    }
    
    # Save results
    print(f"\n Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f" Saved!")
    
    print(f"\n Evaluation complete!")
    print(f"   Time: {elapsed_time/60:.1f} minutes")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user.")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
