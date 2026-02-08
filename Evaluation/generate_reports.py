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
        <p><strong>Evaluation Date:</strong> {self.results['summary']['evaluation_date']}</p>
        <p><strong>Total Questions:</strong> {self.results['summary']['total_questions']}</p>
        <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"   ✅ HTML report: {html_file}")
    
    def _generate_summary_section(self) -> str:
        """Generate summary section"""
        
        mrr_score = self.results['mandatory_metric']['score']
        f1_score = self.results['custom_metric_1']['metrics']['f1_score']
        grounding = self.results['custom_metric_2']['metrics']['grounding_score']
        
        llm_judge_html = ""
        if 'llm_judge' in self.results:
            overall = self.results['llm_judge']['average_scores']['overall_score']['mean']
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
        
        # Faithfulness
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
        
        html += """
        </div>
    </div>
        """
        return html
    
    def generate_csv_report(self):
        """Generate CSV report with all results"""
        
        csv_file = os.path.join(self.output_dir, "evaluation_report.csv")
        
        print(f"\n📄 Generating CSV report...")
        
        per_question = self.results['per_question_results']
        aq_details = self.results['custom_metric_1']['details']
        faith_details = self.results['custom_metric_2']['details']
        mrr_details = self.results['mandatory_metric']['details']
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'Question_ID',
                'Question',
                'Generated_Answer',
                'Ground_Truth',
                'MRR',
                'F1_Score',
                'ROUGE_L',
                'Semantic_Similarity',
                'Grounding_Score',
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
            for i, result in enumerate(per_question):
                row = [
                    f"qa_{i+1:03d}",
                    result['question'],
                    result['generated_answer'],
                    result['ground_truth_answer'],
                    mrr_details['reciprocal_ranks'][i],
                    aq_details['f1_score']['scores'][i],
                    aq_details['rouge_l']['scores'][i],
                    aq_details['semantic_similarity']['scores'][i],
                    faith_details['grounding_score']['scores'][i],
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
