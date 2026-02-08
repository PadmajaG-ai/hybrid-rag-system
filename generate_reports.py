"""
Comprehensive Report Generator
================================

Generates beautiful HTML, CSV, and text reports with:
- Overall performance summary
- Detailed metric justifications
- Results tables
- Visualizations
- Error analysis

Usage:
    python generate_reports.py judge_results.json --output-dir evaluation_output
"""

import json
import argparse
import sys
import os
from datetime import datetime
import csv


class ReportGenerator:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self, results_file: str, output_dir: str = "evaluation_output"):
        self.results_file = results_file
        self.output_dir = output_dir
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_html_report(self):
        """Generate HTML report with visualizations"""
        
        html_file = os.path.join(self.output_dir, "evaluation_report.html")
        
        print(f"\n📄 Generating HTML report...")
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RAG System Evaluation Report</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .score-box {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .score-value {{
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .score-label {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .justification {{
            background: #f9f9f9;
            padding: 20px;
            border-left: 4px solid #667eea;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .section {{
            margin: 40px 0;
        }}
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .interpretation {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4caf50;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 RAG System Evaluation Report</h1>
        <p>Comprehensive Performance Analysis</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    {self._generate_summary_section()}
    {self._generate_metrics_section()}
    {self._generate_results_table()}
    {self._generate_error_analysis()}
    
    <div class="metric-card">
        <h2>📋 Report Information</h2>
        <p><strong>Evaluation Date:</strong> {self.results.get('summary', {}).get('evaluation_date', 'N/A')}</p>
        <p><strong>Total Questions:</strong> {self.results.get('summary', {}).get('total_questions', len(self.results.get('per_question_metrics', self.results.get('per_question_results', []))))}</p>
        <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"   ✅ HTML report: {html_file}")
    
    def _generate_summary_section(self) -> str:
        """Generate summary section"""
        
        # Handle both old and new result formats
        if 'mandatory_metric' in self.results:
            mrr_score = self.results['mandatory_metric']['score']
            f1_score = self.results['custom_metric_1']['metrics']['f1_score']
            grounding = self.results['custom_metric_2']['metrics']['grounding_score']
        else:
            # New format from evaluate_rag_methods.py
            method_comp = self.results.get('method_comparison', {})
            hybrid = method_comp.get('hybrid', {})
            mrr_score = hybrid.get('mrr', {}).get('mean', 0)
            f1_score = hybrid.get('f1', {}).get('mean', 0)
            error_analysis = self.results.get('error_analysis', {})
            hall_stats = error_analysis.get('hallucination_stats', {})
            grounding = 1.0 - (hall_stats.get('hallucination_rate', 0) / 100.0)
        
        llm_judge_html = ""
        if 'llm_judge' in self.results:
            llm = self.results['llm_judge']
            if 'average_scores' in llm:
                overall = llm['average_scores']['overall_score']['mean']
            else:
                overall = llm.get('overall_score', 0)
            llm_judge_html = f"""
            <div class="score-box">
                <div class="score-label">LLM-as-Judge</div>
                <div class="score-value">{overall:.2f}/5</div>
            </div>
            """
        
        return f"""
    <div class="section">
        <h2>📊 Overall Performance Summary</h2>
        <div class="metric-grid">
            <div class="score-box">
                <div class="score-label">MRR Score</div>
                <div class="score-value">{mrr_score:.3f}</div>
            </div>
            <div class="score-box">
                <div class="score-label">F1 Score</div>
                <div class="score-value">{f1_score:.3f}</div>
            </div>
            <div class="score-box">
                <div class="score-label">Faithfulness</div>
                <div class="score-value">{grounding:.3f}</div>
            </div>
            {llm_judge_html}
        </div>
    </div>
        """
    
    def _generate_metrics_section(self) -> str:
        """Generate detailed metrics section"""
        
        html = """
    <div class="section">
        <h2>📏 Detailed Metric Justifications</h2>
        """
        
        # Handle both old and new formats
        if 'mandatory_metric' in self.results:
            # Old format
            # MRR
            mrr = self.results['mandatory_metric']
            html += f"""
        <div class="metric-card">
            <h3>1️⃣  Mean Reciprocal Rank (MRR) - Mandatory Metric</h3>
            <p><strong>Score:</strong> {mrr['score']:.4f}</p>
            <div class="interpretation">
                <strong>Interpretation:</strong> {mrr['interpretation']}
            </div>
            <div class="justification">
                <strong>Why MRR at URL Level:</strong><br>
                Measures how quickly the system identifies the correct Wikipedia URL in retrieved results.
                URL-level evaluation (not chunk-level) assesses whether the system retrieves the right 
                source document, which is critical for source attribution and trust.
                <br><br>
                <strong>Calculation:</strong> MRR = (1/N) × Σ(1/rank_i) where rank_i is the position of 
                the first chunk with the correct URL for question i.
            </div>
        </div>
        """
            
            # Answer Quality
            aq = self.results['custom_metric_1']
            html += f"""
        <div class="metric-card">
            <h3>2️⃣  Answer Quality - Custom Metric 1</h3>
            <p><strong>F1 Score:</strong> {aq['metrics']['f1_score']:.4f}</p>
            <p><strong>ROUGE-L:</strong> {aq['metrics']['rouge_l']:.4f}</p>
            <p><strong>Semantic Similarity:</strong> {aq['metrics']['semantic_similarity']:.4f}</p>
            <div class="justification">
                {aq['justification'].strip().replace(chr(10), '<br>')}
            </div>
        </div>
        """
        else:
            # New format from evaluate_rag_methods.py
            method_comp = self.results.get('method_comparison', {})
            hybrid = method_comp.get('hybrid', {})
            
            html += f"""
        <div class="metric-card">
            <h3>1️⃣  Mean Reciprocal Rank (MRR)</h3>
            <p><strong>Hybrid MRR:</strong> {hybrid.get('mrr', {}).get('mean', 0):.4f}</p>
            <div class="justification">
                Measures how quickly the system identifies relevant documents in retrieved results.
                Higher MRR indicates better ranking of relevant sources.
            </div>
        </div>
        
        <div class="metric-card">
            <h3>2️⃣  Answer Quality Metrics</h3>
            <p><strong>F1 Score:</strong> {hybrid.get('f1', {}).get('mean', 0):.4f}</p>
            <p><strong>ROUGE-L:</strong> {hybrid.get('rouge_l', {}).get('mean', 0):.4f}</p>
            <p><strong>Semantic Similarity:</strong> {hybrid.get('semantic_similarity', {}).get('mean', 0):.4f}</p>
            <div class="justification">
                Measures the quality and relevance of generated answers:
                <br>• F1 Score: Token overlap between generated and reference answers
                <br>• ROUGE-L: Longest common subsequence matching
                <br>• Semantic Similarity: Embedding-based semantic alignment
            </div>
        </div>
        """
        
        # Faithfulness
        if 'custom_metric_2' in self.results:
            # Old format
            faith = self.results['custom_metric_2']
            html += f"""
        <div class="metric-card">
            <h3>3️⃣  Faithfulness (Answer Grounding) - Custom Metric 2</h3>
            <p><strong>Grounding Score:</strong> {faith['metrics']['grounding_score']:.4f}</p>
            <p><strong>Faithful Rate:</strong> {faith['metrics']['faithful_rate']:.2%}</p>
            <p><strong>Hallucination Rate:</strong> {faith['metrics']['hallucination_rate']:.2%}</p>
            <div class="justification">
                {faith['justification'].strip().replace(chr(10), '<br>')}
            </div>
        </div>
        """
        else:
            # New format from evaluate_rag_methods.py
            error_analysis = self.results.get('error_analysis', {})
            hall_stats = error_analysis.get('hallucination_stats', {})
            grounding_score = 1.0 - (hall_stats.get('hallucination_rate', 0) / 100.0)
            
            html += f"""
        <div class="metric-card">
            <h3>3️⃣  Faithfulness (Answer Grounding)</h3>
            <p><strong>Grounding Score:</strong> {grounding_score:.4f}</p>
            <p><strong>Hallucination Rate:</strong> {hall_stats.get('hallucination_rate', 0):.1f}%</p>
            <p><strong>Faithful Rate:</strong> {hall_stats.get('faithful_rate', 0):.1f}%</p>
            <div class="justification">
                Measures whether generated answers are grounded in retrieved context:
                <br>• Hallucination Rate: Answers with content not supported by retrieved documents
                <br>• Faithful Rate: Answers well-grounded in retrieved context
                <br>• Grounding Score: Semantic overlap between answer and context
            </div>
        </div>
        """
        
        # LLM Judge
        if 'llm_judge' in self.results:
            judge = self.results['llm_judge']
            avg_scores = judge['average_scores']
            html += f"""
        <div class="metric-card">
            <h3>4️⃣  LLM-as-Judge - Innovative Evaluation</h3>
            <p><strong>Overall Score:</strong> {avg_scores['overall_score']['mean']:.2f}/5.0</p>
            <p><strong>Factual Accuracy:</strong> {avg_scores['factual_accuracy']['mean']:.2f}/5.0</p>
            <p><strong>Completeness:</strong> {avg_scores['completeness']['mean']:.2f}/5.0</p>
            <p><strong>Relevance:</strong> {avg_scores['relevance']['mean']:.2f}/5.0</p>
            <p><strong>Coherence:</strong> {avg_scores['coherence']['mean']:.2f}/5.0</p>
            <div class="interpretation">
                <strong>Interpretation:</strong> {judge['interpretation']}
            </div>
            <div class="justification">
                {judge['methodology'].strip().replace(chr(10), '<br>')}
            </div>
        </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_results_table(self) -> str:
        """Generate results table"""
        
        html = """
    <div class="section">
        <h2>📋 Detailed Results Table</h2>
        <div class="metric-card">
            <p>First 10 questions shown. See CSV for complete results.</p>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Question</th>
                    <th>MRR</th>
                    <th>F1</th>
                    <th>ROUGE-L</th>
                    <th>Semantic</th>
                    <th>Grounding</th>
                </tr>
        """
    
        # Handle both formats
        # New format: has method_comparison and error_analysis, and per_question_results/per_question_metrics with method-specific metrics
        # Old format: has mandatory_metric, custom_metric_1, custom_metric_2
        if 'mandatory_metric' in self.results and 'custom_metric_1' in self.results:
            # Old format
            per_question = self.results['per_question_results']
            aq_details = self.results['custom_metric_1']['details']
            faith_details = self.results['custom_metric_2']['details']
            mrr_details = self.results['mandatory_metric']['details']
            
            for i, result in enumerate(per_question[:10], 1):
                html += f"""
                <tr>
                    <td>qa_{i:03d}</td>
                    <td>{result['question'][:60]}...</td>
                    <td>{mrr_details['reciprocal_ranks'][i-1]:.3f}</td>
                    <td>{aq_details['f1_score']['scores'][i-1]:.3f}</td>
                    <td>{aq_details['rouge_l']['scores'][i-1]:.3f}</td>
                    <td>{aq_details['semantic_similarity']['scores'][i-1]:.3f}</td>
                    <td>{faith_details['grounding_score']['scores'][i-1]:.3f}</td>
                </tr>
            """
        else:
            # New format from evaluate_rag_methods.py
            per_question = self.results.get('per_question_results', self.results.get('per_question_metrics', []))
            
            for i, result in enumerate(per_question[:10], 1):
                mrr = result.get('hybrid', {}).get('mrr_score', 0)
                f1 = result.get('hybrid', {}).get('f1_score', 0)
                rouge = result.get('hybrid', {}).get('rouge_l', 0)
                semantic = result.get('hybrid', {}).get('semantic_similarity', 0)
                grounding = result.get('hybrid', {}).get('grounding_score', 0)
                
                html += f"""
                <tr>
                    <td>qa_{i:03d}</td>
                    <td>{result.get('question', '')[:60]}...</td>
                    <td>{mrr:.3f}</td>
                    <td>{f1:.3f}</td>
                    <td>{rouge:.3f}</td>
                    <td>{semantic:.3f}</td>
                    <td>{grounding:.3f}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
    </div>
        """
        return html
    
    def _generate_error_analysis(self) -> str:
        """Generate error analysis section"""
        
        html = """
    <div class="section">
        <h2>🔍 Error Analysis</h2>
        <div class="metric-card">
        """
        
        # Handle both formats
        if 'custom_metric_2' in self.results and 'per_question_results' in self.results:
            # Old format
            # Hallucination analysis
            faith_details = self.results['custom_metric_2']['details']
            hallucination_flags = faith_details['hallucination_flags']
            per_question = self.results['per_question_results']
            
            total = len(hallucination_flags)
            hallucinated = sum(hallucination_flags)
            
            html += f"""
            <h3>⚠️  Hallucination Detection</h3>
            <p><strong>Total Questions:</strong> {total}</p>
            <p><strong>Faithful Answers:</strong> {total - hallucinated} ({(1 - hallucinated/total)*100:.1f}%)</p>
            <p><strong>Potential Hallucinations:</strong> {hallucinated} ({hallucinated/total*100:.1f}%)</p>
        """
            
            if hallucinated > 0:
                html += "<h4>Examples of Potential Hallucinations:</h4>"
                hall_indices = [i for i, flag in enumerate(hallucination_flags) if flag][:3]
                
                for idx in hall_indices:
                    result = per_question[idx]
                    html += f"""
                <div class="justification" style="border-left-color: #ef4444;">
                    <strong>qa_{idx+1:03d}:</strong> {result['question']}<br>
                    <strong>Generated Answer:</strong> {result['generated_answer'][:200]}...<br>
                    <strong>Issue:</strong> Low grounding in retrieved context (potential hallucination)
                </div>
                """
        else:
            # New format from evaluate_rag_methods.py
            error_analysis = self.results.get('error_analysis', {})
            error_dist = error_analysis.get('error_distribution', {})
            hall_stats = error_analysis.get('hallucination_stats', {})
            
            html += f"""
            <h3>⚠️  Hallucination Detection</h3>
            <p><strong>Total Questions:</strong> {error_analysis.get('total_questions', 0)}</p>
            <p><strong>Hallucination Rate:</strong> {hall_stats.get('hallucination_rate', 0):.1f}%</p>
            <p><strong>Faithful Answers:</strong> {hall_stats.get('faithful_answers', 0)} ({hall_stats.get('faithful_rate', 0):.1f}%)</p>
            
            <h3>📊 Error Distribution</h3>
            <p><strong>Successful:</strong> {error_dist.get('successful', {}).get('count', 0)} ({error_dist.get('successful', {}).get('percentage', 0):.1f}%)</p>
            <p><strong>Retrieval Failure:</strong> {error_dist.get('retrieval_failure', {}).get('count', 0)} ({error_dist.get('retrieval_failure', {}).get('percentage', 0):.1f}%)</p>
            <p><strong>Incomplete:</strong> {error_dist.get('incomplete', {}).get('count', 0)} ({error_dist.get('incomplete', {}).get('percentage', 0):.1f}%)</p>
            <p><strong>Hallucination:</strong> {error_dist.get('hallucination', {}).get('count', 0)} ({error_dist.get('hallucination', {}).get('percentage', 0):.1f}%)</p>
            <p><strong>Over-generalization:</strong> {error_dist.get('over_generalization', {}).get('count', 0)} ({error_dist.get('over_generalization', {}).get('percentage', 0):.1f}%)</p>
            <p><strong>Factual Error:</strong> {error_dist.get('factual_error', {}).get('count', 0)} ({error_dist.get('factual_error', {}).get('percentage', 0):.1f}%)</p>
        """
        
        html += """
        </div>
    </div>
        """
        return html
    
    def generate_csv_report(self):
        """Generate CSV report with all results"""
        
        csv_file = os.path.join(self.output_dir, "evaluation_report.csv")
        
        print(f"\n📄 Generating CSV report...")
        
        # Handle both old and new formats
        # New format: has method_comparison and error_analysis, and per_question_results/per_question_metrics with method-specific metrics
        # Old format: has mandatory_metric, custom_metric_1, custom_metric_2
        if 'mandatory_metric' in self.results and 'custom_metric_1' in self.results:
            # Old format
            per_question = self.results['per_question_results']
            aq_details = self.results['custom_metric_1']['details']
            faith_details = self.results['custom_metric_2']['details']
            mrr_details = self.results['mandatory_metric']['details']
            use_old_format = True
        else:
            # New format from evaluate_rag_methods.py
            # Try per_question_results first, then per_question_metrics
            per_question = self.results.get('per_question_results', self.results.get('per_question_metrics', []))
            use_old_format = False
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'Question_ID',
                'Question',
                'Generated_Answer',
                'Ground_Truth',
                'Dense_MRR',
                'Sparse_MRR',
                'Hybrid_MRR',
                'F1_Score',
                'Hallucinated'
            ]
            
            if 'llm_judge' in self.results:
                header.extend([
                    'Judge_Overall',
                    'Judge_Factual',
                    'Judge_Complete',
                    'Judge_Relevant',
                    'Judge_Coherent'
                ])
            
            writer.writerow(header)
            
            # Data
            if use_old_format:
                for i, result in enumerate(per_question):
                    row = [
                        f"qa_{i+1:03d}",
                        result['question'],
                        result['generated_answer'],
                        result['ground_truth_answer'],
                        mrr_details['reciprocal_ranks'][i],
                        mrr_details['reciprocal_ranks'][i],
                        mrr_details['reciprocal_ranks'][i],
                        aq_details['f1_score']['scores'][i],
                        'Yes' if faith_details['hallucination_flags'][i] else 'No'
                    ]
                    
                    if 'llm_judge' in self.results:
                        judgment = self.results['llm_judge']['per_question_judgments'][i]
                        row.extend([
                            judgment['overall_score'],
                            judgment['factual_accuracy']['score'],
                            judgment['completeness']['score'],
                            judgment['relevance']['score'],
                            judgment['coherence']['score']
                        ])
                    
                    writer.writerow(row)
            else:
                # New format - extract from per_question_metrics
                for i, result in enumerate(per_question):
                    dense_mrr = result.get('dense', {}).get('mrr', 0)
                    sparse_mrr = result.get('sparse', {}).get('mrr', 0)
                    hybrid_mrr = result.get('hybrid', {}).get('mrr', 0)
                    f1 = result.get('hybrid', {}).get('f1', 0)
                    hallucinated = result.get('error_category', '') == 'hallucination'
                    
                    row = [
                        f"qa_{i+1:03d}",
                        result.get('question', ''),
                        result.get('generated_answer', ''),
                        result.get('ground_truth_answer', ''),
                        dense_mrr,
                        sparse_mrr,
                        hybrid_mrr,
                        f1,
                        'Yes' if hallucinated else 'No'
                    ]
                    
                    if 'llm_judge' in self.results and 'per_question_judgments' in self.results['llm_judge']:
                        judgment = self.results['llm_judge']['per_question_judgments'][i]
                        row.extend([
                            judgment.get('overall_score', 0),
                            judgment.get('factual_accuracy', {}).get('score', 0),
                            judgment.get('completeness', {}).get('score', 0),
                            judgment.get('relevance', {}).get('score', 0),
                            judgment.get('coherence', {}).get('score', 0)
                        ])
                    
                    writer.writerow(row)
        
        print(f"   ✅ CSV report: {csv_file}")
    
    def generate_all_reports(self):
        """Generate all report formats"""
        
        print("\n" + "="*70)
        print("📊 GENERATING COMPREHENSIVE REPORTS")
        print("="*70)
        
        self.generate_html_report()
        self.generate_csv_report()
        
        print("\n✅ All reports generated!")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive evaluation reports")
    parser.add_argument('results_file', help="Results JSON file (from evaluate_rag.py or llm_judge.py)")
    parser.add_argument('--output-dir', default='evaluation_output', help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"❌ Results file not found: {args.results_file}")
        sys.exit(1)
    
    generator = ReportGenerator(args.results_file, args.output_dir)
    generator.generate_all_reports()
    
    print(f"\n📁 Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
