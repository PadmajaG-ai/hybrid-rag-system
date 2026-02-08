"""
Hybrid RAG System with Reciprocal Rank Fusion (RRF)
====================================================

COMPLETE VERSION with Answer Generation

Combines:
- Dense retrieval (all-MiniLM-L6-v2 + ChromaDB)
- Sparse retrieval (BM25)
- Reciprocal Rank Fusion (RRF)
- Answer Generation (Flan-T5-base)

RRF Formula: RRF_score(d) = Σ 1/(k + rank_i(d)) where k=60

Usage:
    python hybrid_rag_system.py "What is machine learning?"
"""

import sys
from typing import List, Dict
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers not installed!")
    sys.exit(1)

try:
    import chromadb
except ImportError:
    print("❌ chromadb not installed!")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except ImportError:
    print("❌ transformers not installed!")
    sys.exit(1)

from build_bm25_index import BM25Indexer


class HybridRAGSystem:
    """
    Hybrid RAG system combining:
    - Dense retrieval (all-MiniLM-L6-v2 + ChromaDB)
    - Sparse retrieval (BM25)
    - Reciprocal Rank Fusion (RRF)
    - Answer Generation (Flan-T5-base)
    """
    
    def __init__(self, 
                 chroma_collection_name: str = "wikipedia_corpus",
                 bm25_index_file: str = "bm25_index.pkl",
                 rrf_k: int = 60,
                 generator_model: str = "google/flan-t5-base"):
        """
        Initialize hybrid system
        
        Args:
            chroma_collection_name: ChromaDB collection name
            bm25_index_file: Path to BM25 index file
            rrf_k: RRF constant (default: 60)
            generator_model: Model for answer generation
        """
        print("\n🚀 Initializing Hybrid RAG System")
        print("="*70)
        
        self.rrf_k = rrf_k
        
        # Load dense retrieval (ChromaDB + all-MiniLM-L6-v2)
        print("\n📥 Loading dense retrieval system...")
        self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.chroma_collection = self.chroma_client.get_collection(chroma_collection_name)
        print(f"   ✅ Dense: {self.chroma_collection.count()} chunks indexed")
        
        # Load sparse retrieval (BM25)
        print("\n📥 Loading sparse retrieval system...")
        self.bm25_indexer = BM25Indexer.load_index(bm25_index_file)
        print(f"   ✅ Sparse: {len(self.bm25_indexer.chunks)} chunks indexed")
        
        # Load answer generator (Flan-T5)
        print(f"\n📥 Loading answer generator ({generator_model})...")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
        self.generator.eval()  # Set to evaluation mode
        print(f"   ✅ Generator loaded!")
        
        print("\n" + "="*70)
        print("✅ Hybrid system ready!")
        print(f"   RRF constant k = {self.rrf_k}")
        print("="*70)
    
    def dense_retrieval(self, query: str, top_k: int = 20) -> List[Dict]:
        """Dense vector retrieval using ChromaDB"""
        
        # Embed query
        query_embedding = self.dense_model.encode([query])[0].tolist()
        
        # Search ChromaDB
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        dense_results = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            similarity = 1 - distance
            
            dense_results.append({
                'chunk_id': results['ids'][0][i],
                'score': similarity,
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'rank': i + 1,
                'method': 'dense'
            })
        
        return dense_results
    
    def sparse_retrieval(self, query: str, top_k: int = 20) -> List[Dict]:
        """Sparse BM25 retrieval"""
        
        results = self.bm25_indexer.search(query, top_k=top_k)
        
        # Format results
        for result in results:
            result['method'] = 'sparse'
        
        return results
    
    def reciprocal_rank_fusion(self, 
                               dense_results: List[Dict], 
                               sparse_results: List[Dict],
                               top_n: int = 10) -> List[Dict]:
        """
        Combine dense and sparse results using RRF
        
        RRF_score(d) = Σ 1/(k + rank_i(d))
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            top_n: Number of final results to return
        
        Returns:
            Combined and reranked results
        """
        
        # Calculate RRF scores
        rrf_scores = defaultdict(lambda: {'score': 0, 'data': None, 'ranks': {}})
        
        # Add dense retrieval scores
        for result in dense_results:
            chunk_id = result['chunk_id']
            rank = result['rank']
            rrf_scores[chunk_id]['score'] += 1 / (self.rrf_k + rank)
            rrf_scores[chunk_id]['data'] = result
            rrf_scores[chunk_id]['ranks']['dense'] = rank
            rrf_scores[chunk_id]['ranks']['dense_score'] = result['score']
        
        # Add sparse retrieval scores
        for result in sparse_results:
            chunk_id = result['chunk_id']
            rank = result['rank']
            rrf_scores[chunk_id]['score'] += 1 / (self.rrf_k + rank)
            
            # If chunk wasn't in dense results, use sparse data
            if rrf_scores[chunk_id]['data'] is None:
                rrf_scores[chunk_id]['data'] = result
            
            rrf_scores[chunk_id]['ranks']['sparse'] = rank
            rrf_scores[chunk_id]['ranks']['sparse_score'] = result['score']
        
        # Sort by RRF score and take top-N
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:top_n]
        
        # Format final results
        final_results = []
        for i, (chunk_id, data) in enumerate(sorted_results, 1):
            result = data['data'].copy()
            result['rrf_score'] = data['score']
            result['rrf_rank'] = i
            result['component_ranks'] = data['ranks']
            final_results.append(result)
        
        return final_results
    
    def hybrid_search(self, query: str, top_k_per_method: int = 20, top_n_final: int = 10) -> Dict:
        """
        Complete hybrid search pipeline
        
        Args:
            query: Search query
            top_k_per_method: Number of results to retrieve from each method
            top_n_final: Number of final results after RRF
        
        Returns:
            Dictionary with all results and metadata
        """
        
        print(f"\n🔍 Hybrid Search: '{query}'")
        print(f"   Retrieving top-{top_k_per_method} from each method")
        print(f"   Final top-{top_n_final} via RRF (k={self.rrf_k})")
        
        # Dense retrieval
        print("\n   📊 Dense retrieval...")
        dense_results = self.dense_retrieval(query, top_k=top_k_per_method)
        print(f"      ✅ Retrieved {len(dense_results)} chunks")
        
        # Sparse retrieval
        print("\n   📋 Sparse retrieval...")
        sparse_results = self.sparse_retrieval(query, top_k=top_k_per_method)
        print(f"      ✅ Retrieved {len(sparse_results)} chunks")
        
        # RRF fusion
        print(f"\n   🔗 Reciprocal Rank Fusion...")
        hybrid_results = self.reciprocal_rank_fusion(
            dense_results, 
            sparse_results, 
            top_n=top_n_final
        )
        print(f"      ✅ Final top-{len(hybrid_results)} chunks")
        
        return {
            'query': query,
            'dense_results': dense_results,
            'sparse_results': sparse_results,
            'hybrid_results': hybrid_results,
            'metadata': {
                'top_k_per_method': top_k_per_method,
                'top_n_final': top_n_final,
                'rrf_k': self.rrf_k
            }
        }
    
    def get_context_for_llm(self, hybrid_results: List[Dict], max_tokens: int = 2000) -> str:
        """
        Build context string for LLM from hybrid results
        
        Args:
            hybrid_results: Results from hybrid_search
            max_tokens: Maximum tokens for context
        
        Returns:
            Formatted context string
        """
        
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(hybrid_results, 1):
            text = result['text']
            tokens = len(text.split()) * 0.75  # Approximate tokens
            
            if current_tokens + tokens > max_tokens:
                break
            
            context_part = (
                f"[Source {i}: {result['metadata']['title']}]\n"
                f"{text}\n"
            )
            
            context_parts.append(context_part)
            current_tokens += tokens
        
        context = "\n---\n\n".join(context_parts)
        return context
    
    def generate_answer(self, question: str, context: str, max_length: int = 150) -> str:
        """
        Generate answer using Flan-T5 based on question and context
        
        THIS IS THE MISSING METHOD THAT WAS CAUSING THE ERROR!
        
        Args:
            question: User's question
            context: Retrieved context from documents
            max_length: Maximum length of generated answer
        
        Returns:
            Generated answer string
        """
        
        # Format prompt for Flan-T5
        prompt = f"""Answer the following question based on the given context.

Context:
{context[:1500]}

Question: {question}

Answer:"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate answer
        with torch.no_grad():
            outputs = self.generator.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=20,
                num_beams=4,
                temperature=0.7,
                do_sample=False,
                early_stopping=True
            )
        
        # Decode answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        return answer
    
    def query(self, question: str, top_k: int = 20, top_n: int = 10, verbose: bool = True) -> Dict:
        """
        Complete RAG query: retrieve + generate answer
        
        Args:
            question: User's question
            top_k: Top-K per retrieval method
            top_n: Final top-N after fusion
            verbose: Print progress
        
        Returns:
            Dictionary with answer and retrieval results
        """
        
        # Retrieve documents
        search_results = self.hybrid_search(
            question,
            top_k_per_method=top_k,
            top_n_final=top_n
        )
        
        # Build context
        context = self.get_context_for_llm(search_results['hybrid_results'])
        
        # Generate answer
        if verbose:
            print("\n💭 Generating answer...")
        
        answer = self.generate_answer(question, context)
        
        if verbose:
            print(f"   ✅ Answer generated!")
        
        return {
            'question': question,
            'answer': answer,
            'context': context,
            'retrieved_chunks': search_results['hybrid_results'],
            'metadata': search_results['metadata']
        }


def print_results(search_results: Dict):
    """Pretty print hybrid search results"""
    
    query = search_results['query']
    hybrid_results = search_results['hybrid_results']
    
    print("\n" + "="*70)
    print("🎯 HYBRID SEARCH RESULTS (RRF)")
    print("="*70)
    print(f"Query: '{query}'")
    print(f"Top {len(hybrid_results)} chunks after fusion")
    print("="*70)
    
    for result in hybrid_results:
        print(f"\n📄 Rank #{result['rrf_rank']} | RRF Score: {result['rrf_score']:.6f}")
        print(f"   Title: {result['metadata']['title']}")
        print(f"   Chunk ID: {result['chunk_id']}")
        
        # Show component ranks
        ranks = result['component_ranks']
        if 'dense' in ranks:
            print(f"   Dense: Rank #{ranks['dense']}, Score: {ranks['dense_score']:.4f}")
        else:
            print(f"   Dense: Not in top-K")
        
        if 'sparse' in ranks:
            print(f"   Sparse: Rank #{ranks['sparse']}, Score: {ranks['sparse_score']:.4f}")
        else:
            print(f"   Sparse: Not in top-K")
        
        text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
        print(f"\n   📝 Text: {text_preview}")
        print("-" * 70)


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python hybrid_rag_system.py 'your query here'")
        print("\nExample:")
        print("  python hybrid_rag_system.py 'What is machine learning?'")
        sys.exit(1)
    
    query = ' '.join(sys.argv[1:])
    
    # Initialize system
    system = HybridRAGSystem()
    
    # Complete RAG query (retrieve + generate)
    result = system.query(query, top_k=20, top_n=10)
    
    # Print retrieval results
    print_results({
        'query': result['question'],
        'hybrid_results': result['retrieved_chunks']
    })
    
    # Print generated answer
    print("\n" + "="*70)
    print("💬 GENERATED ANSWER")
    print("="*70)
    print(f"\n{result['answer']}\n")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
