# 🔍 Hybrid RAG System with Method Comparison

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Retrieval-Augmented Generation (RAG) system that combines **dense** (semantic) and **sparse** (keyword-based) retrieval methods using Reciprocal Rank Fusion (RRF). Includes automated evaluation pipeline with hallucination detection, error analysis, and LLM-as-Judge scoring.

## ✨ Features

### Core RAG System
- **Hybrid Retrieval**: Combines dense (all-MiniLM-L6-v2) and sparse (BM25) methods
- **Reciprocal Rank Fusion (RRF)**: Intelligent result fusion with k=60
- **Answer Generation**: Flan-T5-base for contextual answer synthesis
- **Interactive UI**: Streamlit demo with real-time query execution

### Evaluation Framework
- **Method Comparison**: Dense-only vs Sparse-only vs Hybrid performance
- **Hallucination Detection**: Grounding score analysis and faithful answer tracking
- **Error Analysis**: 6-category error classification
- **LLM-as-Judge**: Automated scoring on 4 dimensions
- **Comprehensive Metrics**: MRR, F1, ROUGE-L, Semantic Similarity, Grounding Score

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/hybrid-rag-system.git
cd hybrid-rag-system
pip install -r requirements.txt

# Build indices (one-time)
python build_vector_index.py corpus_chunks.json
python build_bm25_index.py corpus_chunks.json

# Run demo
streamlit run rag_ui_flan.py

# Run evaluation
python run_complete_evaluation.py qa_pairs.json

# View dashboard
streamlit evaluation_dashboard_with_methods.py
```

---

## 📁 Project Structure

```
hybrid-rag-system/
├── Data Collection/          # URL generation & text extraction
├── Index Building/           # Dense & sparse index creation
├── RAG System/              # Core system & UI
├── Q&A Generation/          # Automatic question generation
├── Evaluation/              # Complete evaluation pipeline
├── Generated Data/          # Corpus & Q&A pairs
├── Generated Indices/       # ChromaDB & BM25 indices
└── Evaluation Results/      # Evaluation outputs
```

---

## 📊 Results

| Method | MRR | F1 | ROUGE-L |
|--------|-----|----|---------| 
| Dense | 0.742 | 0.683 | 0.621 |
| Sparse | 0.698 | 0.654 | 0.597 |
| **Hybrid** | **0.801** | **0.729** | **0.668** |

**Hybrid shows 7-15% improvement over single methods!**

---

## 🛠️ Technologies

- ChromaDB, Rank-BM25, Sentence Transformers
- Hugging Face Transformers (Flan-T5)
- Streamlit, Plotly, Pandas

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

