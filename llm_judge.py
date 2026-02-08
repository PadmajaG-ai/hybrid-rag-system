"""
LLM-as-Judge Evaluation System
================================

Uses Flan-T5 as a judge to evaluate:
1. Factual Accuracy
2. Completeness
3. Relevance
4. Coherence

Each with automated explanations.

Usage:
    python llm_judge.py evaluation_results.json --output judge_results.json
"""

import json
import argparse
import sys
from typing import Dict, List, Tuple
import time

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    import numpy as np
except ImportError as e:
    print(f" Missing dependency: {e}")
    sys.exit(1)


class LLMJudge:
    """LLM-based evaluation of answer quality"""
    
    EVALUATION_ASPECTS = {
        'factual_accuracy': {
            'prompt_template': """Evaluate the factual accuracy of this answer.

Question: {question}
Reference Answer: {reference}
Generated Answer: {generated}
Retrieved Context: {context}

Does the generated answer contain factually correct information compared to the reference?
Rate from 1-5 where:
1 = Completely incorrect/contradicts reference
2 = Mostly incorrect with some correct facts
3 = Partially correct, missing key facts
4 = Mostly correct with minor errors
5 = Completely accurate

Score and brief explanation:""",
            'weight': 0.3
        },
        'completeness': {
            'prompt_template': """Evaluate the completeness of this answer.

Question: {question}
Reference Answer: {reference}
Generated Answer: {generated}

Does the generated answer cover all important points from the reference?
Rate from 1-5 where:
1 = Misses all main points
2 = Covers few points, many gaps
3 = Covers some points, notable gaps
4 = Covers most points, minor gaps
5 = Covers all important points

Score and brief explanation:""",
            'weight': 0.3
        },
        'relevance': {
            'prompt_template': """Evaluate the relevance of this answer.

Question: {question}
Generated Answer: {generated}

Does the generated answer directly address the question asked?
Rate from 1-5 where:
1 = Completely off-topic
2 = Mostly irrelevant, tangential
3 = Partially relevant, some drift
4 = Mostly relevant, minor drift
5 = Directly addresses question

Score and brief explanation:""",
            'weight': 0.2
        },
        'coherence': {
            'prompt_template': """Evaluate the coherence and readability of this answer.

Generated Answer: {generated}

Is the answer well-structured, logical, and easy to understand?
Rate from 1-5 where:
1 = Incoherent, confusing
2 = Poorly structured, hard to follow
3 = Acceptable structure, some clarity issues
4 = Well-structured, mostly clear
5 = Excellent structure, very clear

Score and brief explanation:""",
            'weight': 0.2
        }
    }
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize LLM judge"""
        print(f"\n🤖 Loading {model_name} as judge...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        print("   ✅ Judge model loaded!")
    
    def judge_answer(self, question: str, reference: str, generated: str, 
                     context: str = "") -> Dict:
        """
        Evaluate answer on all aspects
        
        Returns dict with scores and explanations for each aspect
        """
        
        results = {}
        
        for aspect, config in self.EVALUATION_ASPECTS.items():
            # Create prompt
            prompt = config['prompt_template'].format(
                question=question[:500],
                reference=reference[:500],
                generated=generated[:500],
                context=context[:800]
            )
            
            # Get evaluation from LLM
            score, explanation = self._get_judgment(prompt)
            
            results[aspect] = {
                'score': score,
                'explanation': explanation,
                'weight': config['weight']
            }
        
        # Calculate weighted overall score
        overall_score = sum(
            r['score'] * r['weight'] 
            for r in results.values()
        )
        
        results['overall_score'] = overall_score
        
        return results
    
    def _get_judgment(self, prompt: str) -> Tuple[int, str]:
        """Get judgment from LLM"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=4,
                temperature=0.7,
                do_sample=False
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Parse score and explanation
        score, explanation = self._parse_response(response)
        
        return score, explanation
    
    def _parse_response(self, response: str) -> Tuple[int, str]:
        """Parse LLM response to extract score and explanation"""
        
        # Try to extract score (1-5)
        import re
        
        # Look for patterns like "Score: 4" or "4/5" or just "4"
        score_patterns = [
            r'score[:\s]+(\d)',
            r'(\d)\s*/\s*5',
            r'^(\d)',
            r'rating[:\s]+(\d)'
        ]
        
        score = 3  # Default middle score
        
        for pattern in score_patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 5:
                        break
                except:
                    pass
        
        # Ensure score is in valid range
        score = max(1, min(5, score))
        
        # Use rest of response as explanation
        explanation = response[:200]  # Limit length
        
        return score, explanation
    
    def evaluate_all(self, evaluation_results: Dict) -> Dict:
        """
        Evaluate all answers using LLM-as-Judge
        
        Args:
            evaluation_results: Results from evaluate_rag.py
            
        Returns:
            Enhanced results with LLM judgments
        """
        
        print("\n" + "="*70)
        print("👨‍⚖️ LLM-AS-JUDGE EVALUATION")
        print("="*70)
        print("Evaluating: Factual Accuracy, Completeness, Relevance, Coherence")
        print("="*70)
        
        per_question = evaluation_results['per_question_results']
        
        judgments = []
        
        for i, result in enumerate(per_question, 1):
            # Get context from retrieved chunks
            context = " ".join([
                chunk.get('text', '')[:200] 
                for chunk in result.get('retrieved_chunks', [])[:3]
            ])
            
            # Judge answer
            judgment = self.judge_answer(
                question=result['question'],
                reference=result['ground_truth_answer'],
                generated=result['generated_answer'],
                context=context
            )
            
            judgment['question_id'] = f"qa_{i:03d}"
            judgment['question'] = result['question']
            judgments.append(judgment)
            
            if i % 10 == 0:
                print(f"   Evaluated {i}/{len(per_question)} answers...")
        
        print(f"   ✅ All answers evaluated!")
        
        # Calculate average scores
        avg_scores = self._calculate_averages(judgments)
        
        # Print summary
        self._print_summary(avg_scores)
        
        return {
            'per_question_judgments': judgments,
            'average_scores': avg_scores,
            'methodology': self._get_methodology(),
            'interpretation': self._interpret_scores(avg_scores)
        }
    
    def _calculate_averages(self, judgments: List[Dict]) -> Dict:
        """Calculate average scores across all questions"""
        
        aspects = ['factual_accuracy', 'completeness', 'relevance', 'coherence', 'overall_score']
        
        averages = {}
        
        for aspect in aspects:
            if aspect == 'overall_score':
                scores = [j['overall_score'] for j in judgments]
            else:
                scores = [j[aspect]['score'] for j in judgments]
            
            # Use Python native operations (JSON-compatible)
            # This fixes the "int64/float64 not JSON serializable" error
            n = len(scores)
            mean_val = sum(scores) / n
            
            # Calculate standard deviation manually
            variance = sum((x - mean_val) ** 2 for x in scores) / n
            std_val = variance ** 0.5
            
            averages[aspect] = {
                'mean': mean_val,      # Python float (JSON-compatible)
                'std': std_val,        # Python float (JSON-compatible)
                'min': min(scores),    # Python int/float (JSON-compatible)
                'max': max(scores)     # Python int/float (JSON-compatible)
            }
        
        return averages
    
    def _print_summary(self, avg_scores: Dict):
        """Print evaluation summary"""
        
        print(f"\n{'='*70}")
        print(" LLM-AS-JUDGE SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n   Factual Accuracy: {avg_scores['factual_accuracy']['mean']:.2f}/5.0")
        print(f"\n   Completeness: {avg_scores['completeness']['mean']:.2f}/5.0")
        print(f"     Relevance: {avg_scores['relevance']['mean']:.2f}/5.0")
        print(f"     Coherence: {avg_scores['coherence']['mean']:.2f}/5.0")
        print(f"\n   Overall Score: {avg_scores['overall_score']['mean']:.2f}/5.0")
    
    def _get_methodology(self) -> str:
        """Return methodology description"""
        return """
LLM-as-Judge Methodology:

Uses Flan-T5-base as an automated judge to evaluate answer quality across 4 dimensions:

1. Factual Accuracy (30% weight): Correctness compared to reference answer
2. Completeness (30% weight): Coverage of key points from reference
3. Relevance (20% weight): How well answer addresses the question
4. Coherence (20% weight): Structure, clarity, and readability

Each dimension rated 1-5 with explanation:
- 1: Poor
- 2: Below Average
- 3: Average
- 4: Good
- 5: Excellent

Overall score is weighted average of 4 dimensions.

Advantages:
- Consistent evaluation across all answers
- Captures nuanced quality aspects beyond metrics
- Provides explanations for ratings
- Scales to large datasets
        """
    
    def _interpret_scores(self, avg_scores: Dict) -> str:
        """Interpret average scores"""
        
        overall = avg_scores['overall_score']['mean']
        
        if overall >= 4.0:
            return "Excellent - High quality answers across all dimensions"
        elif overall >= 3.5:
            return "Good - Solid answers with room for minor improvements"
        elif overall >= 3.0:
            return "Fair - Acceptable answers but notable gaps in quality"
        elif overall >= 2.5:
            return "Below Average - Significant quality issues"
        else:
            return "Poor - Major quality problems requiring attention"


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation")
    parser.add_argument('results_file', help="evaluation_results.json from evaluate_rag.py")
    parser.add_argument('--output', default='judge_results.json', help="Output file")
    
    args = parser.parse_args()
    
    # Load evaluation results
    print(f"\n Loading evaluation results from {args.results_file}...")
    with open(args.results_file, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)
    print(f"   ✅ Loaded!")
    
    # Initialize judge
    judge = LLMJudge()
    
    # Run evaluation
    judge_results = judge.evaluate_all(eval_results)
    
    # Combine with original results
    combined = {
        **eval_results,
        'llm_judge': judge_results
    }
    
    # Save (now with JSON-compatible types!)
    print(f"\n Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"   Saved!")
    
    print(f"\n LLM-as-Judge evaluation complete!")
    print(f"\n Results saved to: {args.output}")
    print(f"\n View results with:")
    print(f"   python view_evaluation.py {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
