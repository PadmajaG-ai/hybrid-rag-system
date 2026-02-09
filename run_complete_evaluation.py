"""
COMPLETE Automated RAG Evaluation Pipeline
===========================================

Includes:
- Method Comparison (Dense/Sparse/Hybrid)
- Hallucination Detection
- Error Analysis
- LLM-as-Judge

Usage:
    python run_complete_evaluation.py qa_pairs.json
"""

import subprocess
import sys
import os
import argparse
import time
from datetime import datetime


class Colors:
    """Terminal colors"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class CompletePipeline:
    """Complete evaluation pipeline with all features"""
    
    def __init__(self, qa_pairs_file: str, output_dir: str = "evaluation_output"):
        self.qa_pairs_file = qa_pairs_file
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.eval_results_file = os.path.join(output_dir, "evaluation_results.json")
        self.judge_results_file = os.path.join(output_dir, "judge_results.json")
        self.html_report_file = os.path.join(output_dir, "evaluation_report.html")
        self.csv_report_file = os.path.join(output_dir, "evaluation_report.csv")
        self.summary_file = os.path.join(output_dir, "evaluation_summary.txt")
        
        self.step_results = {
            'evaluation': False,
            'llm_judge': False,
            'reports': False,
            'summary': False
        }
        
        self.step_times = {}
    
    def print_banner(self):
        """Print pipeline banner"""
        print("\n" + "="*70)
        print(f"{Colors.BOLD}{Colors.CYAN} COMPLETE RAG EVALUATION PIPELINE{Colors.ENDC}")
        print("="*70)
        print(f" Input: {Colors.BLUE}{self.qa_pairs_file}{Colors.ENDC}")
        print(f" Output: {Colors.BLUE}{self.output_dir}/{Colors.ENDC}")
        print(f" Started: {Colors.GREEN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
        print("="*70 + "\n")
        
        print("   Pipeline Steps:")
        print("   [1/4] Evaluation (Method Comparison + Hallucination + Error Analysis)")
        print("   [2/4] LLM-as-Judge (Factual, Complete, Relevant, Coherent)")
        print("   [3/4] Generate Reports (HTML, CSV, JSON)")
        print("   [4/4] Create Summary\n")
    
    def print_step_header(self, step_num: int, step_name: str):
        """Print step header"""
        print("\n" + "="*70)
        print(f"{Colors.BOLD}[STEP {step_num}/4] {step_name}{Colors.ENDC}")
        print("="*70)
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.GREEN}✓{Colors.ENDC} {message}")
    
    def print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.RED}✗{Colors.ENDC} {message}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠{Colors.ENDC} {message}")
    
    def print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.CYAN}ℹ{Colors.ENDC} {message}")
    
    def step_1_evaluation(self):
        """Run complete evaluation"""
        self.print_step_header(1, "Complete Evaluation")
        
        print("Evaluating 3 methods + Hallucination + Error Analysis:")
        print("   • Dense-only retrieval (all-MiniLM-L6-v2)")
        print("   • Sparse-only retrieval (BM25)")
        print("   • Hybrid retrieval (RRF fusion)")
        print("   • Hallucination detection")
        print("   • Error categorization (6 types)")
        print("\nThis will take approximately 15-20 minutes...\n")
        
        step_start = time.time()
        
        cmd = [
            sys.executable,
            "evaluate_rag_methods.py",
            self.qa_pairs_file,
            "--output", self.eval_results_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=False, check=True)
            
            elapsed = time.time() - step_start
            self.step_times['evaluation'] = elapsed
            
            if os.path.exists(self.eval_results_file):
                # Parse and show summary
                import json
                try:
                    with open(self.eval_results_file, 'r') as f:
                        results = json.load(f)
                    
                    print("\n" + "-"*70)
                    print(f"{Colors.BOLD}EVALUATION RESULTS:{Colors.ENDC}")
                    print("-"*70)
                    
                    # Method comparison
                    if 'method_comparison' in results:
                        method_comp = results['method_comparison']
                        print(f"\n{Colors.BOLD}Method Comparison:{Colors.ENDC}")
                        for method in ['dense', 'sparse', 'hybrid']:
                            mrr = method_comp[method]['mrr']['mean']
                            f1 = method_comp[method]['f1']['mean']
                            print(f"   {method.capitalize():10s} → MRR: {mrr:.4f}  |  F1: {f1:.4f}")
                        
                        if 'improvements' in method_comp:
                            imp = method_comp['improvements']
                            print(f"\n   {Colors.GREEN}Hybrid Improvement:{Colors.ENDC}")
                            print(f"      vs Dense:  MRR {imp['hybrid_vs_dense']['mrr']:+.2f}%")
                            print(f"      vs Sparse: MRR {imp['hybrid_vs_sparse']['mrr']:+.2f}%")
                    
                    # Error analysis
                    if 'error_analysis' in results:
                        error_analysis = results['error_analysis']
                        print(f"\n{Colors.BOLD}Error Analysis:{Colors.ENDC}")
                        
                        success_rate = error_analysis.get('success_rate', 0)
                        print(f"   Success Rate: {Colors.GREEN}{success_rate:.1f}%{Colors.ENDC}")
                        
                        error_dist = error_analysis.get('error_distribution', {})
                        print(f"   Retrieval Failure: {error_dist.get('retrieval_failure', {}).get('count', 0)}")
                        print(f"   Incomplete: {error_dist.get('incomplete', {}).get('count', 0)}")
                        print(f"   Hallucination: {error_dist.get('hallucination', {}).get('count', 0)}")
                        print(f"   Over-generalization: {error_dist.get('over_generalization', {}).get('count', 0)}")
                        print(f"   Factual Error: {error_dist.get('factual_error', {}).get('count', 0)}")
                        
                        # Hallucination stats
                        hall_stats = error_analysis.get('hallucination_stats', {})
                        print(f"\n{Colors.BOLD}Hallucination Detection:{Colors.ENDC}")
                        print(f"   Hallucination Rate: {hall_stats.get('hallucination_rate', 0):.1f}%")
                        print(f"   Faithful Rate: {hall_stats.get('faithful_rate', 0):.1f}%")
                    
                    print("-"*70)
                except:
                    pass
                
                self.print_success(f"Evaluation complete! ({elapsed/60:.1f} minutes)")
                self.print_success(f"Results saved to: {self.eval_results_file}")
                self.step_results['evaluation'] = True
                return True
            else:
                self.print_error("Evaluation file not created!")
                return False
                
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - step_start
            self.step_times['evaluation'] = elapsed
            self.print_error(f"Evaluation failed after {elapsed/60:.1f} minutes!")
            return False
        except Exception as e:
            elapsed = time.time() - step_start
            self.step_times['evaluation'] = elapsed
            self.print_error(f"Unexpected error: {e}")
            return False
    
    def step_2_llm_judge(self):
        """Run LLM-as-Judge evaluation"""
        self.print_step_header(2, "LLM-as-Judge Evaluation")
        
        print("Evaluating answers on 4 dimensions:")
        print("   • Factual Accuracy (1-5)")
        print("   • Completeness (1-5)")
        print("   • Relevance (1-5)")
        print("   • Coherence (1-5)")
        print("\nThis will take approximately 10-15 minutes...\n")
        
        step_start = time.time()
        
        cmd = [
            sys.executable,
            "llm_judge.py",
            self.eval_results_file,
            "--output", self.judge_results_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=False, check=True)
            
            elapsed = time.time() - step_start
            self.step_times['llm_judge'] = elapsed
            
            if os.path.exists(self.judge_results_file):
                # Show summary
                import json
                try:
                    with open(self.judge_results_file, 'r') as f:
                        results = json.load(f)
                    
                    if 'llm_judge' in results:
                        llm = results['llm_judge']
                        scores = llm.get('average_scores', {})
                        
                        print("\n" + "-"*70)
                        print(f"{Colors.BOLD}LLM JUDGE RESULTS:{Colors.ENDC}")
                        print("-"*70)
                        
                        for metric in ['factual_accuracy', 'completeness', 'relevance', 'coherence']:
                            score = scores.get(metric, {}).get('mean', 0)
                            print(f"   {metric.replace('_', ' ').title():20s} → {score:.2f}/5.0")
                        
                        overall = scores.get('overall_score', {}).get('mean', 0)
                        print(f"\n   {Colors.GREEN}Overall Score:{Colors.ENDC} {overall:.2f}/5.0")
                        print("-"*70)
                except:
                    pass
                
                self.print_success(f"LLM Judge complete! ({elapsed/60:.1f} minutes)")
                self.print_success(f"Results saved to: {self.judge_results_file}")
                self.step_results['llm_judge'] = True
                return True
            else:
                self.print_warning("LLM Judge file not created!")
                return False
                
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - step_start
            self.step_times['llm_judge'] = elapsed
            self.print_warning(f"LLM Judge failed after {elapsed/60:.1f} minutes")
            self.print_info("Continuing without LLM Judge (optional)")
            return False
        except Exception as e:
            elapsed = time.time() - step_start
            self.step_times['llm_judge'] = elapsed
            self.print_warning(f"LLM Judge error: {e}")
            return False
    
    def step_3_generate_reports(self):
        """Generate reports (HTML, CSV, etc)"""
        self.print_step_header(3, "Generate Reports")
        
        print("Generating comprehensive reports (HTML, CSV):\n")
        
        step_start = time.time()
        
        cmd = [
            sys.executable,
            "generate_reports.py",
            self.judge_results_file if os.path.exists(self.judge_results_file) else self.eval_results_file,
            "--output-dir", self.output_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=False, check=True)
            
            elapsed = time.time() - step_start
            self.step_times['reports'] = elapsed
            
            self.print_success(f"Reports generated! ({elapsed:.1f} seconds)")
            self.print_success(f"Files saved to: {self.output_dir}/")
            self.step_results['reports'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - step_start
            self.step_times['reports'] = elapsed
            self.print_warning(f"Report generation failed after {elapsed:.1f} seconds")
            return False
        except Exception as e:
            elapsed = time.time() - step_start
            self.step_times['reports'] = elapsed
            self.print_warning(f"Report generation error: {e}")
            return False
    
    def step_4_create_summary(self):
        """Create summary"""
        self.print_step_header(4, "Create Summary")
        
        step_start = time.time()
        
        try:
            import json
            with open(self.eval_results_file, 'r') as f:
                eval_results = json.load(f)
            
            # Create summary
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("COMPLETE RAG EVALUATION SUMMARY\n")
                f.write("="*70 + "\n\n")
                
                summary = eval_results.get('summary', {})
                f.write(f"Total Questions: {summary.get('total_questions', 'N/A')}\n")
                f.write(f"Evaluation Date: {summary.get('evaluation_date', 'N/A')}\n\n")
                
                # Method comparison
                method_comp = eval_results.get('method_comparison', {})
                if method_comp:
                    f.write("="*70 + "\n")
                    f.write("METHOD COMPARISON\n")
                    f.write("="*70 + "\n\n")
                    for method in ['dense', 'sparse', 'hybrid']:
                        f.write(f"{method.upper()}:\n")
                        f.write(f"  MRR: {method_comp[method]['mrr']['mean']:.4f}\n")
                        f.write(f"  F1:  {method_comp[method]['f1']['mean']:.4f}\n\n")
                
                # Error analysis
                error_analysis = eval_results.get('error_analysis', {})
                if error_analysis:
                    f.write("="*70 + "\n")
                    f.write("ERROR ANALYSIS\n")
                    f.write("="*70 + "\n\n")
                    f.write(f"Success Rate: {error_analysis.get('success_rate', 0):.1f}%\n\n")
                    
                    error_dist = error_analysis.get('error_distribution', {})
                    for cat in ['successful', 'retrieval_failure', 'incomplete', 
                               'hallucination', 'over_generalization', 'factual_error']:
                        if cat in error_dist:
                            count = error_dist[cat]['count']
                            pct = error_dist[cat]['percentage']
                            f.write(f"  {cat.replace('_', ' ').title()}: {count} ({pct:.1f}%)\n")
                    
                    f.write("\n")
                    hall_stats = error_analysis.get('hallucination_stats', {})
                    f.write(f"Hallucination Rate: {hall_stats.get('hallucination_rate', 0):.1f}%\n")
                    f.write(f"Faithful Rate: {hall_stats.get('faithful_rate', 0):.1f}%\n\n")
                
                f.write("="*70 + "\n")
                f.write("NEXT STEPS\n")
                f.write("="*70 + "\n\n")
                f.write("1. View dashboard:\n")
                f.write("   streamlit run evaluation_dashboard_with_methods.py\n\n")
                f.write("2. Take screenshots:\n")
                f.write("   - Method Comparison tab\n")
                f.write("   - LLM Judge tab\n")
                f.write("   - Hallucination Detection tab\n")
                f.write("   - Error Analysis tab\n\n")
            
            elapsed = time.time() - step_start
            self.step_times['summary'] = elapsed
            
            self.print_success(f"Summary created! ({elapsed:.1f} seconds)")
            self.print_success(f"Summary saved to: {self.summary_file}")
            self.step_results['summary'] = True
            return True
            
        except Exception as e:
            elapsed = time.time() - step_start
            self.step_times['summary'] = elapsed
            self.print_warning(f"Failed to create summary: {e}")
            return False
    
    def print_final_summary(self, total_elapsed: float):
        """Print final summary"""
        print("\n" + "="*70)
        print(f"{Colors.BOLD}{Colors.GREEN} PIPELINE COMPLETE!{Colors.ENDC}")
        print("="*70)
        
        success_count = sum(self.step_results.values())
        total_steps = len(self.step_results)
        
        print(f"\n{Colors.BOLD}Results:{Colors.ENDC}")
        print(f"   {success_count}/{total_steps} steps completed")
        print(f"     Total time: {total_elapsed/60:.1f} minutes")
        
        print(f"\n{Colors.BOLD}Generated Files:{Colors.ENDC}")
        
        output_files = [
            ("Evaluation Results", self.eval_results_file),
            ("LLM Judge Results", self.judge_results_file),
            ("Summary", self.summary_file)
        ]
        
        for name, path in output_files:
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024
                print(f"    {name:20s} → {os.path.basename(path)} ({size:.1f} KB)")
            else:
                print(f"    {name:20s} (not created)")
        
        
    
    def run(self):
        """Run complete pipeline"""
        self.print_banner()
        
        pipeline_start = time.time()
        
        # Step 1: Complete evaluation (CRITICAL)
        if not self.step_1_evaluation():
            print("\n" + "="*70)
            print(f"{Colors.RED} PIPELINE FAILED AT STEP 1{Colors.ENDC}")
            print("="*70)
            print("Evaluation failed - cannot continue.")
            return False
        
        # Step 2: LLM Judge (OPTIONAL)
        self.step_2_llm_judge()
        
        # Step 3: Reports (OPTIONAL)
        self.step_3_generate_reports()
        
        # Step 4: Summary
        self.step_4_create_summary()
        
        # Final summary
        total_elapsed = time.time() - pipeline_start
        self.print_final_summary(total_elapsed)
        
        return True


def check_requirements():
    """Check required files"""
    required_scripts = [
        'evaluate_rag_methods.py',
        'llm_judge.py',
        'evaluation_dashboard_with_methods.py'
    ]
    
    required_data = [
        'corpus_chunks.json',
        'chroma_db',
        'bm25_index.pkl'
    ]
    
    missing_scripts = [f for f in required_scripts if not os.path.exists(f)]
    missing_data = [f for f in required_data if not os.path.exists(f)]
    
    if missing_scripts or missing_data:
        print(f"\n{Colors.RED} Missing required files:{Colors.ENDC}\n")
        
        if missing_scripts:
            print(f"{Colors.BOLD}Missing scripts:{Colors.ENDC}")
            for file in missing_scripts:
                print(f"   {file}")
        
        if missing_data:
            print(f"\n{Colors.BOLD}Missing data/indices:{Colors.ENDC}")
            for file in missing_data:
                print(f"   {file}")
        
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Complete RAG Evaluation Pipeline")
    parser.add_argument('qa_pairs_file', help="Q&A pairs JSON file")
    parser.add_argument('--output-dir', default='evaluation_output', help="Output directory")
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check input file
    if not os.path.exists(args.qa_pairs_file):
        print(f"\n{Colors.RED} Input file not found: {args.qa_pairs_file}{Colors.ENDC}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = CompletePipeline(args.qa_pairs_file, args.output_dir)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW} Pipeline interrupted by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED} Pipeline error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
