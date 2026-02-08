"""
Hybrid RAG System - Streamlit UI with Flan-T5-base
===================================================

Uses Flan-T5-base for better question answering!

Flan-T5 is BETTER for RAG because:
- Instruction-tuned (follows instructions)
- Better at Q&A tasks
- More coherent answers
- Same size as DistilGPT2

Usage:
    streamlit run rag_ui_flan.py
"""

import streamlit as st
import time
from typing import List, Dict
import sys

# Try importing required modules
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from build_bm25_index import BM25Indexer
    from hybrid_rag_system import HybridRAGSystem
except ImportError as e:
    st.error(f" Missing dependency: {e}")
    st.info("Install with: pip install -r requirements_complete.txt")
    st.stop()

# Try importing transformers
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    st.error(" Transformers not installed!")
    st.info("Install with: pip install transformers torch")

# Page config
st.set_page_config(
    page_title="Hybrid RAG with Flan-T5",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .source-box {
        border-left: 3px solid #1E88E5;
        padding-left: 15px;
        margin: 10px 0;
    }
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        font-size: 1.1rem;
        line-height: 1.6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'flan_model' not in st.session_state:
    st.session_state.flan_model = None
if 'flan_tokenizer' not in st.session_state:
    st.session_state.flan_tokenizer = None
if 'search_history' not in st.session_state:
    st.session_state.search_history = []


@st.cache_resource
def load_rag_system():
    """Load hybrid RAG system (cached)"""
    with st.spinner("🔧 Loading Hybrid RAG System..."):
        try:
            system = HybridRAGSystem(
                chroma_collection_name="wikipedia_corpus",
                bm25_index_file="bm25_index.pkl",
                rrf_k=60
            )
            return system
        except Exception as e:
            st.error(f"Failed to load RAG system: {e}")
            st.info("Make sure you've run: build_vector_index.py and build_bm25_index.py")
            st.info("ChromaDB storage should be in ./chroma_db directory")
            return None


@st.cache_resource
def load_flan_t5():
    """Load Flan-T5-base model (cached)"""
    
    if not LLM_AVAILABLE:
        return None, None
    
    with st.spinner("Loading Flan-T5-base model... (first time: ~900MB download)"):
        try:
            model_name = "google/flan-t5-base"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Set to eval mode
            model.eval()
            
            return model, tokenizer
            
        except Exception as e:
            st.error(f"Failed to load Flan-T5: {e}")
            return None, None


def generate_answer_flan_t5(model, tokenizer, query: str, context: str, max_length: int = 200) -> str:
    """
    Generate answer using Flan-T5-base
    
    Flan-T5 is instruction-tuned, so we use a clear prompt format
    """
    
    if model is None or tokenizer is None:
        return "Flan-T5 model not available. Please check installation."
    
    try:
        # Build instruction-style prompt (Flan-T5 works best with this format)
        prompt = f"""Answer the question based on the context below.

Context:
{context[:1500]}

Question: {query}

Answer:"""
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_beams=4,  # Beam search for better quality
                early_stopping=True,
                temperature=0.7,
                do_sample=False,  # Greedy for consistency
                no_repeat_ngram_size=3
            )
        
        # Decode
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up
        answer = answer.strip()
        
        if not answer or len(answer) < 10:
            answer = "I don't have enough information in the provided context to answer this question confidently."
        
        return answer
        
    except Exception as e:
        return f"Error generating answer: {e}"


def display_chunk(chunk: Dict, rank: int):
    """Display a single chunk result"""
    
    with st.container():
        # Header with rank and scores
        col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
        
        with col1:
            st.markdown(f"### #{rank}")
        
        with col2:
            st.markdown(f"**RRF:** `{chunk['rrf_score']:.6f}`")
        
        with col3:
            ranks = chunk['component_ranks']
            if 'dense' in ranks:
                st.markdown(f"**Dense:** #{ranks['dense']} ({ranks['dense_score']:.3f})")
            else:
                st.markdown("**Dense:** Not in top-K")
        
        with col4:
            ranks = chunk['component_ranks']
            if 'sparse' in ranks:
                st.markdown(f"**Sparse:** #{ranks['sparse']} ({ranks['sparse_score']:.3f})")
            else:
                st.markdown("**Sparse:** Not in top-K")
        
        # Title and URL
        st.markdown(f"** {chunk['metadata']['title']}**")
        st.markdown(f" [{chunk['metadata']['url']}]({chunk['metadata']['url']})")
        
        # Text
        with st.expander(" View Full Text", expanded=False):
            st.write(chunk['text'])
        
        st.markdown("---")


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header"> Hybrid RAG with Flan-T5</div>', unsafe_allow_html=True)
    st.markdown("""
    **Dense** (all-MiniLM-L6-v2) + **Sparse** (BM25) + **RRF** + **Flan-T5-base**
    
    ✨ *Flan-T5 is instruction-tuned for better question answering!*
    """)
    
    # Sidebar
    with st.sidebar:
        st.header(" Configuration")
        
        # Model info
        st.subheader(" LLM Model")
        st.info("""
        **Flan-T5-base**
        - Size: 248MB
        - Type: Seq2Seq (Encoder-Decoder)
        - Training: Instruction-tuned
        - Best for: Q&A, RAG systems
        """)
        
        # Retrieval parameters
        st.subheader(" Retrieval Parameters")
        top_k_per_method = st.slider("Top-K per method", 5, 50, 20, key="top_k")
        top_n_final = st.slider("Final top-N (RRF)", 3, 20, 10, key="top_n")
        rrf_k = st.slider("RRF constant k", 10, 100, 60, key="rrf_k")
        
        # Generation parameters
        st.subheader("Generation Parameters")
        max_answer_tokens = st.slider("Max answer tokens", 50, 300, 150, key="max_tokens")
        
        st.markdown("---")
        
        # System status
        if st.session_state.rag_system:
            st.success("RAG System Loaded")
            st.metric("Chunks Indexed", 
                     st.session_state.rag_system.chroma_collection.count())
        
        if st.session_state.flan_model:
            st.success(" Flan-T5 Model Loaded")
        
        # Search history
        if st.session_state.search_history:
            st.subheader(" Recent Searches")
            for i, q in enumerate(reversed(st.session_state.search_history[-5:]), 1):
                st.text(f"{i}. {q[:40]}...")
        
        st.markdown("---")
        
        # Help
        with st.expander(" About Flan-T5"):
            st.markdown("""
            **Why Flan-T5 for RAG?**
            
            **Instruction-tuned**: Follows prompts better
            **Q&A optimized**: Trained for question answering
            **Coherent**: More natural responses
            **Efficient**: 248MB (manageable size)
            
            Flan-T5 > DistilGPT2 for RAG tasks!
            """)
    
    # Load systems
    if st.session_state.rag_system is None:
        st.session_state.rag_system = load_rag_system()
    
    if st.session_state.rag_system is None:
        st.error(" Failed to load RAG system. Please check error messages above.")
        st.stop()
    
    # Load Flan-T5
    if st.session_state.flan_model is None:
        model, tokenizer = load_flan_t5()
        st.session_state.flan_model = model
        st.session_state.flan_tokenizer = tokenizer
    
    # Main query input
    st.markdown("###  Ask a Question")
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is machine learning? How does photosynthesis work?",
        key="query_input",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button(" Search & Answer", type="primary", use_container_width=True)
    with col2:
        if st.session_state.search_history:
            if st.button(" Clear History", use_container_width=True):
                st.session_state.search_history = []
                st.rerun()
    
    # Process query
    if search_button and query:
        # Add to history
        st.session_state.search_history.append(query)
        
        # Start timer
        start_time = time.time()
        
        # Search
        with st.spinner(" Retrieving relevant chunks..."):
            results = st.session_state.rag_system.hybrid_search(
                query,
                top_k_per_method=top_k_per_method,
                top_n_final=top_n_final
            )
        
        retrieval_time = time.time() - start_time
        
        # Generate answer
        answer = None
        generation_time = 0
        
        if st.session_state.flan_model:
            with st.spinner(" Generating answer with Flan-T5..."):
                gen_start = time.time()
                
                # Get context
                context = st.session_state.rag_system.get_context_for_llm(
                    results['hybrid_results'],
                    max_tokens=2000
                )
                
                # Generate with Flan-T5
                answer = generate_answer_flan_t5(
                    st.session_state.flan_model,
                    st.session_state.flan_tokenizer,
                    query,
                    context,
                    max_length=max_answer_tokens
                )
                
                generation_time = time.time() - gen_start
        
        total_time = time.time() - start_time
        
        # Display results
        st.markdown("---")
        
        # Response time metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(" Retrieval", f"{retrieval_time:.2f}s")
        with col2:
            st.metric(" Generation", f"{generation_time:.2f}s")
        with col3:
            st.metric(" Total", f"{total_time:.2f}s")
        
        st.markdown("---")
        
        # Generated answer (featured display)
        if answer:
            st.subheader(" Answer from Flan-T5")
            st.markdown(f"""
            <div class="answer-box">
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Source attribution
            st.caption(" Answer generated from top retrieved Wikipedia chunks")
            st.markdown("---")
        
        # Retrieved chunks in tabs
        st.subheader(f" Retrieved Chunks (Top {len(results['hybrid_results'])})")
        
        tab1, tab2, tab3 = st.tabs([" Hybrid (RRF)", " Dense Results", " Sparse Results"])
        
        with tab1:
            st.markdown("**Reciprocal Rank Fusion Results**")
            st.markdown(f"*Formula: RRF_score(d) = Σ 1/(k + rank_i(d)) where k={rrf_k}*")
            st.markdown("---")
            
            for chunk in results['hybrid_results']:
                display_chunk(chunk, chunk['rrf_rank'])
        
        with tab2:
            st.markdown("**Dense Vector Retrieval (all-MiniLM-L6-v2)**")
            st.markdown("---")
            
            for i, chunk in enumerate(results['dense_results'][:top_n_final], 1):
                with st.container():
                    st.markdown(f"### #{i}")
                    st.markdown(f"**Similarity:** `{chunk['score']:.4f}`")
                    st.markdown(f"** {chunk['metadata']['title']}**")
                    st.markdown(f" [{chunk['metadata']['url']}]({chunk['metadata']['url']})")
                    with st.expander(" View Text"):
                        st.write(chunk['text'])
                    st.markdown("---")
        
        with tab3:
            st.markdown("**Sparse Keyword Retrieval (BM25)**")
            st.markdown("---")
            
            for i, chunk in enumerate(results['sparse_results'][:top_n_final], 1):
                with st.container():
                    st.markdown(f"### #{i}")
                    st.markdown(f"**BM25 Score:** `{chunk['score']:.4f}`")
                    st.markdown(f"** {chunk['metadata']['title']}**")
                    st.markdown(f" [{chunk['metadata']['url']}]({chunk['metadata']['url']})")
                    with st.expander(" View Text"):
                        st.write(chunk['text'])
                    st.markdown("---")
    
    elif search_button and not query:
        st.warning(" Please enter a question!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.9rem;">
         Powered by <b>Flan-T5-base</b> • Dense (all-MiniLM-L6-v2) + Sparse (BM25) + RRF
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
