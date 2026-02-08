"""
COMPLETE RAG Evaluation Dashboard
==================================

ALL FEATURES INCLUDED:
- Overview
- Method Comparison (Dense vs Sparse vs Hybrid)
- LLM-as-Judge
- Hallucination Detection
- Error Analysis
- Question Explorer

Usage:
    streamlit run evaluation_dashboard_COMPLETE.py
"""

import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="RAG Evaluation Interactive Dashboard",
    page_icon="",
    layout="wide"
)


@st.cache_data
def load_results(filename: str):
    """Load evaluation results"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None


def show_overview(results):
    """Show overall metrics overview"""
    st.header(" Performance Overview")
    
    method_comp = results.get('method_comparison', {})
    error_analysis = results.get('error_analysis', {})
    
    if not method_comp:
        st.warning(" No method comparison data available")
        return
    
    # Overall metrics for hybrid (primary method)
    hybrid = method_comp.get('hybrid', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        mrr = hybrid.get('mrr', {}).get('mean', 0)
        st.metric("MRR", f"{mrr:.4f}", help="Mean Reciprocal Rank")
    
    with col2:
        f1 = hybrid.get('f1', {}).get('mean', 0)
        st.metric("F1 Score", f"{f1:.4f}", help="F1 Score")
    
    with col3:
        rouge = hybrid.get('rouge_l', {}).get('mean', 0)
        st.metric("ROUGE-L", f"{rouge:.4f}", help="ROUGE-L Score")
    
    with col4:
        semantic = hybrid.get('semantic_similarity', {}).get('mean', 0)
        st.metric("Semantic", f"{semantic:.4f}", help="Semantic Similarity")
    
    with col5:
        grounding = hybrid.get('grounding', {}).get('mean', 0)
        st.metric("Grounding", f"{grounding:.4f}", help="Grounding Score")
    
    st.markdown("---")
    
    # Success rate and error summary
    if error_analysis:
        st.subheader(" Quality Metrics")
        
        success_rate = error_analysis.get('success_rate', 0)
        hall_stats = error_analysis.get('hallucination_stats', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Success Rate", f"{success_rate:.1f}%", 
                     help="Percentage of high-quality answers")
        
        with col2:
            hall_rate = hall_stats.get('hallucination_rate', 0)
            st.metric("Hallucination Rate", f"{hall_rate:.1f}%",
                     delta=f"-{hall_rate:.1f}%", delta_color="inverse",
                     help="Percentage of hallucinated answers")
        
        with col3:
            faithful_rate = hall_stats.get('faithful_rate', 0)
            st.metric("Faithful Rate", f"{faithful_rate:.1f}%",
                     help="Percentage of faithful answers")


def show_method_comparison(results):
    """Show method comparison tab"""
    st.header(" Method Comparison: Dense vs Sparse vs Hybrid")
    
    method_comp = results.get('method_comparison', {})
    
    if not method_comp:
        st.warning(" No method comparison data")
        return
    
    # Comparison table
    st.subheader(" Performance Comparison")
    
    comparison_data = []
    for method in ['dense', 'sparse', 'hybrid']:
        comparison_data.append({
            'Method': method.capitalize(),
            'MRR': method_comp[method]['mrr']['mean'],
            'F1': method_comp[method]['f1']['mean'],
            'ROUGE-L': method_comp[method]['rouge_l']['mean'],
            'Semantic': method_comp[method]['semantic_similarity']['mean'],
            'Grounding': method_comp[method]['grounding']['mean']
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comp_df.style.highlight_max(axis=0, subset=['MRR', 'F1', 'ROUGE-L', 'Semantic', 'Grounding'], 
                                     color='lightgreen'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Bar charts
    st.subheader(" Metric Comparison")
    
    col1, col2 = st.columns(2)
    
    metrics = ['mrr', 'f1', 'rouge_l', 'semantic_similarity', 'grounding']
    metric_labels = ['MRR', 'F1', 'ROUGE-L', 'Semantic', 'Grounding']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        fig = go.Figure(data=[
            go.Bar(name='Dense', x=[label], y=[method_comp['dense'][metric]['mean']], 
                  marker_color='#667eea', text=[f"{method_comp['dense'][metric]['mean']:.3f}"],
                  textposition='auto'),
            go.Bar(name='Sparse', x=[label], y=[method_comp['sparse'][metric]['mean']], 
                  marker_color='#764ba2', text=[f"{method_comp['sparse'][metric]['mean']:.3f}"],
                  textposition='auto'),
            go.Bar(name='Hybrid', x=[label], y=[method_comp['hybrid'][metric]['mean']], 
                  marker_color='#f093fb', text=[f"{method_comp['hybrid'][metric]['mean']:.3f}"],
                  textposition='auto')
        ])
        fig.update_layout(title=label, barmode='group', height=300, yaxis=dict(range=[0, 1]))
        
        if i % 2 == 0:
            with col1:
                st.plotly_chart(fig, use_container_width=True)
        else:
            with col2:
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Improvement analysis
    st.subheader(" Improvement Analysis")
    
    improvements = method_comp.get('improvements', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Hybrid vs Dense:**")
        vs_dense = improvements.get('hybrid_vs_dense', {})
        st.write(f"• MRR: {vs_dense.get('mrr', 0):+.2f}%")
        st.write(f"• F1: {vs_dense.get('f1', 0):+.2f}%")
    
    with col2:
        st.markdown("**Hybrid vs Sparse:**")
        vs_sparse = improvements.get('hybrid_vs_sparse', {})
        st.write(f"• MRR: {vs_sparse.get('mrr', 0):+.2f}%")
        st.write(f"• F1: {vs_sparse.get('f1', 0):+.2f}%")


def show_llm_judge(results):
    """Show LLM-as-Judge tab"""
    st.header(" LLM-as-Judge Evaluation")
    
    llm_judge = results.get('llm_judge', {})
    
    if not llm_judge:
        st.warning(" No LLM-as-Judge data available")
        st.info("Run: `python llm_judge.py evaluation_results.json --output judge_results.json`")
        return
    
    avg_scores = llm_judge.get('average_scores', {})
    overall = avg_scores.get('overall_score', {}).get('mean', 0)
    
    # Overall score gauge
    st.subheader(" Overall Score")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Score (out of 5.0)"},
            delta = {'reference': 3.0},
            gauge = {
                'axis': {'range': [None, 5]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 2], 'color': "#ffcccb"},
                    {'range': [2, 3], 'color': "#ffffcc"},
                    {'range': [3, 4], 'color': "#ccffcc"},
                    {'range': [4, 5], 'color': "#90ee90"}
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Mean", f"{overall:.2f}/5.0")
        st.metric("Std Dev", f"{avg_scores.get('overall_score', {}).get('std', 0):.2f}")
    
    with col3:
        interpretation = llm_judge.get('interpretation', '')
        if interpretation:
            st.info(f"**Interpretation:**\n\n{interpretation}")
    
    st.markdown("---")
    
    # Dimension breakdown
    st.subheader(" Dimension Breakdown")
    
    dimensions = ['factual_accuracy', 'completeness', 'relevance', 'coherence']
    dimension_labels = {
        'factual_accuracy': 'Factual Accuracy',
        'completeness': 'Completeness',
        'relevance': 'Relevance',
        'coherence': 'Coherence'
    }
    
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    
    for i, dim in enumerate(dimensions):
        with cols[i]:
            score = avg_scores.get(dim, {}).get('mean', 0)
            std = avg_scores.get(dim, {}).get('std', 0)
            st.metric(dimension_labels[dim], f"{score:.2f}/5.0", f"±{std:.2f}")
    
    # Bar chart
    dim_data = []
    for dim in dimensions:
        dim_data.append({
            'Dimension': dimension_labels[dim],
            'Score': avg_scores.get(dim, {}).get('mean', 0),
            'Std': avg_scores.get(dim, {}).get('std', 0)
        })
    
    dim_df = pd.DataFrame(dim_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dim_df['Dimension'],
        y=dim_df['Score'],
        error_y=dict(type='data', array=dim_df['Std']),
        marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe'],
        text=dim_df['Score'].apply(lambda x: f"{x:.2f}"),
        textposition='outside'
    ))
    fig.update_layout(
        title='LLM Judge Scores by Dimension',
        yaxis_title='Score (out of 5.0)',
        yaxis=dict(range=[0, 5.5]),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def show_hallucination_detection(results):
    """Show hallucination detection tab"""
    st.header(" Hallucination Detection")
    
    error_analysis = results.get('error_analysis', {})
    
    if not error_analysis:
        st.warning(" No hallucination detection data available")
        return
    
    hall_stats = error_analysis.get('hallucination_stats', {})
    
    # Summary metrics
    st.subheader(" Hallucination Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_hall = hall_stats.get('total_hallucinations', 0)
        st.metric("Total Hallucinations", total_hall)
    
    with col2:
        hall_rate = hall_stats.get('hallucination_rate', 0)
        st.metric("Hallucination Rate", f"{hall_rate:.1f}%",
                 delta=f"-{hall_rate:.1f}%", delta_color="inverse")
    
    with col3:
        faithful = hall_stats.get('faithful_answers', 0)
        st.metric("Faithful Answers", faithful)
    
    with col4:
        faithful_rate = hall_stats.get('faithful_rate', 0)
        st.metric("Faithful Rate", f"{faithful_rate:.1f}%")
    
    st.markdown("---")
    
    # Per-question hallucination analysis
    st.subheader(" Per-Question Analysis")
    
    per_question = results.get('per_question_results', [])
    
    if per_question:
        hall_data = []
        for i, result in enumerate(per_question, 1):
            hall_det = result.get('hallucination_detection', {})
            hall_data.append({
                'Q#': i,
                'Question': result['question'][:80] + '...',
                'Hallucination': ' Yes' if hall_det.get('is_hallucination') else ' No',
                'Faithful': ' Yes' if hall_det.get('is_faithful') else ' No',
                'Grounding': hall_det.get('grounding_score', 0),
                'Explanation': hall_det.get('explanation', '')
            })
        
        hall_df = pd.DataFrame(hall_data)
        
        # Filter options
        filter_option = st.selectbox(
            "Filter by:",
            ["All", "Hallucinations Only", "Faithful Only", "Low Grounding (<0.5)"]
        )
        
        if filter_option == "Hallucinations Only":
            hall_df = hall_df[hall_df['Hallucination'] == ' Yes']
        elif filter_option == "Faithful Only":
            hall_df = hall_df[hall_df['Faithful'] == ' Yes']
        elif filter_option == "Low Grounding (<0.5)":
            hall_df = hall_df[hall_df['Grounding'] < 0.5]
        
        st.dataframe(
            hall_df.style.background_gradient(cmap='RdYlGn', subset=['Grounding'], vmin=0, vmax=1),
            use_container_width=True,
            height=400
        )
        
        # Grounding score distribution
        st.subheader(" Grounding Score Distribution")
        
        grounding_scores = [result.get('hallucination_detection', {}).get('grounding_score', 0) 
                           for result in per_question]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=grounding_scores, nbinsx=20, marker_color='#667eea'))
        fig.update_layout(
            title='Distribution of Grounding Scores',
            xaxis_title='Grounding Score',
            yaxis_title='Frequency',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def show_error_analysis(results):
    """Show error analysis tab"""
    st.header("🔍 Error Analysis")
    
    error_analysis = results.get('error_analysis', {})
    
    if not error_analysis:
        st.warning(" No error analysis data available")
        return
    
    error_dist = error_analysis.get('error_distribution', {})
    
    # Summary metrics
    st.subheader(" Error Distribution")
    
    # Create distribution data
    categories = {
        'successful': ' Successful',
        'retrieval_failure': ' Retrieval Failure',
        'incomplete': ' Incomplete',
        'hallucination': ' Hallucination',
        'over_generalization': ' Over-generalization',
        'factual_error': ' Factual Error'
    }
    
    dist_data = []
    for cat_key, cat_label in categories.items():
        if cat_key in error_dist:
            dist_data.append({
                'Category': cat_label,
                'Count': error_dist[cat_key]['count'],
                'Percentage': error_dist[cat_key]['percentage']
            })
    
    dist_df = pd.DataFrame(dist_data)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        success_pct = error_dist.get('successful', {}).get('percentage', 0)
        st.metric("Success Rate", f"{success_pct:.1f}%")
    
    with col2:
        error_count = sum(error_dist[cat]['count'] for cat in error_dist if cat != 'successful')
        st.metric("Total Errors", error_count)
    
    with col3:
        critical_count = (error_dist.get('retrieval_failure', {}).get('count', 0) + 
                         error_dist.get('hallucination', {}).get('count', 0))
        st.metric("Critical Errors", critical_count)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=dist_df['Category'],
            values=dist_df['Count'],
            hole=.3,
            marker_colors=['#90ee90', '#ffcccb', '#ffffcc', '#ff6b6b', '#ffa07a', '#ff8c94']
        )])
        fig.update_layout(title='Error Distribution', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart
        fig = go.Figure(data=[go.Bar(
            x=dist_df['Category'],
            y=dist_df['Percentage'],
            marker_color=['#90ee90', '#ffcccb', '#ffffcc', '#ff6b6b', '#ffa07a', '#ff8c94'],
            text=dist_df['Percentage'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside'
        )])
        fig.update_layout(
            title='Error Percentage by Category',
            yaxis_title='Percentage (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Per-question error details
    st.subheader(" Per-Question Error Details")
    
    per_question = results.get('per_question_results', [])
    
    if per_question:
        error_data = []
        for i, result in enumerate(per_question, 1):
            error_cat = result.get('error_analysis', {})
            error_data.append({
                'Q#': i,
                'Question': result['question'][:80] + '...',
                'Category': categories.get(error_cat.get('category', ''), 'Unknown'),
                'Severity': error_cat.get('severity', 'unknown').upper(),
                'Explanation': error_cat.get('explanation', ''),
                'MRR': error_cat.get('metrics', {}).get('mrr', 0),
                'F1': error_cat.get('metrics', {}).get('f1', 0),
                'Grounding': error_cat.get('metrics', {}).get('grounding', 0)
            })
        
        error_df = pd.DataFrame(error_data)
        
        # Filter options
        filter_option = st.selectbox(
            "Filter by category:",
            ["All"] + list(categories.values())
        )
        
        if filter_option != "All":
            error_df = error_df[error_df['Category'] == filter_option]
        
        st.dataframe(
            error_df.style.applymap(
                lambda x: 'background-color: #ffcccb' if x == 'CRITICAL' else 
                         ('background-color: #ffffcc' if x == 'MAJOR' else 
                         ('background-color: #ccffcc' if x == 'MINOR' else '')),
                subset=['Severity']
            ),
            use_container_width=True,
            height=400
        )


def show_questions(results):
    """Show question explorer"""
    st.header(" Question Explorer")
    
    per_question = results.get('per_question_results', [])
    
    if not per_question:
        st.warning(" No per-question results")
        return
    
    st.subheader(" Select a Question")
    
    question_options = [f"Q{i}: {q['question'][:80]}..." for i, q in enumerate(per_question, 1)]
    selected = st.selectbox("Choose from questions:", question_options)
    
    idx = int(selected.split(':')[0][1:]) - 1
    result = per_question[idx]
    
    st.markdown("---")
    
    # Question and answers
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(" Question")
        st.write(result['question'])
        
        st.subheader(" Reference Answer")
        st.write(result['ground_truth_answer'])
    
    with col2:
        st.subheader(" Generated Answer")
        st.write(result['generated_answer'])
    
    st.markdown("---")
    
    # Method performance
    if 'dense' in result and 'sparse' in result and 'hybrid' in result:
        st.subheader(" Method Performance")
        
        methods_data = []
        for method in ['dense', 'sparse', 'hybrid']:
            methods_data.append({
                'Method': method.capitalize(),
                'MRR': result[method]['mrr_score'],
                'F1': result[method]['f1_score'],
                'ROUGE-L': result[method]['rouge_l'],
                'Rank': result[method]['rank']
            })
        
        methods_df = pd.DataFrame(methods_data)
        st.dataframe(
            methods_df.style.highlight_max(axis=0, subset=['MRR', 'F1', 'ROUGE-L'], color='lightgreen')
                           .highlight_min(axis=0, subset=['Rank'], color='lightgreen'),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Error analysis for this question
    if 'error_analysis' in result:
        st.subheader(" Error Analysis")
        
        error_cat = result['error_analysis']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Category", error_cat['category'].replace('_', ' ').title())
        
        with col2:
            severity_color = {'critical': '🔴', 'major': '🟡', 'minor': '🟢', 'none': '✅'}
            st.metric("Severity", f"{severity_color.get(error_cat['severity'], '')} {error_cat['severity'].upper()}")
        
        with col3:
            hall = result.get('hallucination_detection', {})
            st.metric("Grounding", f"{hall.get('grounding_score', 0):.3f}")
        
        st.info(f"**Explanation:** {error_cat['explanation']}")


def main():
    """Main dashboard"""
    
    st.title("RAG Evaluation interactive Dashboard")
    st.markdown("**All Features: Method Comparison | LLM Judge | Hallucination | Error Analysis**")
    st.markdown("---")
    
    st.sidebar.header(" Configuration")
    
    # Look for files in evaluation_output/
    available = []
    for f in ['evaluation_output/evaluation_results.json', 'evaluation_output/judge_results.json']:
        if os.path.exists(f):
            available.append(f)
    
    if not available:
        st.error(" No results files found in evaluation_output/!")
        st.info("Expected files:\n- evaluation_output/evaluation_results.json\n- evaluation_output/judge_results.json")
        st.stop()
    
    selected_file = st.sidebar.selectbox(" Results File:", available, index=0)
    
    # Load results
    results = load_results(selected_file)
    
    if not results:
        st.stop()
    
    st.sidebar.success(f" Loaded: {os.path.basename(selected_file)}")
    
    # Check for features
    has_method_comp = 'method_comparison' in results
    has_llm_judge = 'llm_judge' in results
    has_error_analysis = 'error_analysis' in results
    
    if has_method_comp:
        st.sidebar.info(" Method comparison ")
    if has_llm_judge:
        st.sidebar.info(" LLM Judge ")
    if has_error_analysis:
        st.sidebar.info(" Error analysis ")
        st.sidebar.info(" Hallucination detection ")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Dataset")
    summary = results.get('summary', {})
    st.sidebar.metric("Questions", summary.get('total_questions', 'N/A'))
    
    # Create tabs
    tab_names = [" Overview"]
    
    if has_method_comp:
        tab_names.append(" Method Comparison")
    
    if has_llm_judge:
        tab_names.append(" LLM Judge")
    
    if has_error_analysis:
        tab_names.append(" Hallucination")
        tab_names.append(" Error Analysis")
    
    tab_names.append(" Questions")
    
    tabs = st.tabs(tab_names)
    
    # Show tabs
    current_tab = 0
    
    with tabs[current_tab]:
        show_overview(results)
    current_tab += 1
    
    if has_method_comp:
        with tabs[current_tab]:
            show_method_comparison(results)
        current_tab += 1
    
    if has_llm_judge:
        with tabs[current_tab]:
            show_llm_judge(results)
        current_tab += 1
    
    if has_error_analysis:
        with tabs[current_tab]:
            show_hallucination_detection(results)
        current_tab += 1
        
        with tabs[current_tab]:
            show_error_analysis(results)
        current_tab += 1
    
    with tabs[current_tab]:
        show_questions(results)


if __name__ == "__main__":
    main()
