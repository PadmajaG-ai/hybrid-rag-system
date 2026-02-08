"""
BM25 Sparse Keyword Retrieval
==============================

Implements BM25 algorithm for keyword-based retrieval over chunks

Usage:
    python build_bm25_index.py corpus_chunks.json
"""

import json
import pickle
import sys
from typing import List, Dict
from collections import Counter
import math

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("❌ Error: rank-bm25 not installed!")
    print("Install with: pip install rank-bm25")
    sys.exit(1)

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


class BM25Indexer:
    """Build BM25 index for sparse keyword retrieval"""
    
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple tokenization: lowercase and split
        tokens = text.lower().split()
        # Remove punctuation
        tokens = [''.join(c for c in token if c.isalnum()) for token in tokens]
        # Remove empty tokens
        tokens = [t for t in tokens if t]
        return tokens
    
    def load_corpus(self, corpus_file: str) -> Dict:
        """Load corpus from JSON file"""
        print(f"\n📂 Loading corpus from: {corpus_file}")
        
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            
            self.chunks = corpus.get('chunks', [])
            print(f"   ✅ Loaded {len(self.chunks)} chunks")
            
            return corpus
            
        except FileNotFoundError:
            print(f"   ❌ File not found: {corpus_file}")
            sys.exit(1)
    
    def build_index(self, corpus_file: str):
        """Build BM25 index from corpus"""
        
        print("\n" + "="*70)
        print("🔍 Building BM25 Sparse Index")
        print("="*70)
        
        # Load corpus
        corpus = self.load_corpus(corpus_file)
        
        # Tokenize all chunks
        print(f"\n📝 Tokenizing {len(self.chunks)} chunks...")
        self.tokenized_corpus = []
        
        for i, chunk in enumerate(self.chunks):
            tokens = self.tokenize(chunk['text'])
            self.tokenized_corpus.append(tokens)
            
            if (i + 1) % 1000 == 0:
                print(f"   Progress: {i + 1}/{len(self.chunks)}")
        
        print(f"   ✅ Tokenized all chunks")
        
        # Build BM25 index
        print(f"\n🏗️  Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"   ✅ BM25 index built!")
        
        # Save index
        self.save_index()
        
        print("\n" + "="*70)
        print("✅ BM25 INDEX COMPLETE")
        print("="*70)
        print(f"📊 Statistics:")
        print(f"   Total chunks indexed: {len(self.chunks)}")
        print(f"   Vocabulary size: {len(set(token for doc in self.tokenized_corpus for token in doc))}")
        print(f"   Algorithm: BM25 (Okapi)")
        print("="*70)
    
    def save_index(self, filename: str = 'bm25_index.pkl'):
        """Save BM25 index to file"""
        print(f"\n💾 Saving BM25 index...")
        
        index_data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"   ✅ Saved to {filename}")
    
    @staticmethod
    def load_index(filename: str = 'bm25_index.pkl'):
        """Load BM25 index from file"""
        print(f"📂 Loading BM25 index from {filename}...")
        
        with open(filename, 'rb') as f:
            index_data = pickle.load(f)
        
        indexer = BM25Indexer()
        indexer.bm25 = index_data['bm25']
        indexer.chunks = index_data['chunks']
        indexer.tokenized_corpus = index_data['tokenized_corpus']
        
        print(f"   ✅ Loaded {len(indexer.chunks)} chunks")
        
        return indexer
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using BM25"""
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-K indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Format results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'rank': rank,
                'chunk_id': self.chunks[idx]['chunk_id'],
                'score': float(scores[idx]),
                'text': self.chunks[idx]['text'],
                'metadata': {
                    'title': self.chunks[idx]['title'],
                    'url': self.chunks[idx]['url'],
                    'chunk_index': self.chunks[idx]['chunk_index'],
                    'token_count': self.chunks[idx]['token_count']
                }
            })
        
        return results


def test_search(indexer: BM25Indexer, query: str, top_k: int = 5):
    """Test BM25 search"""
    
    print(f"\n🔍 Testing BM25 Search")
    print(f"   Query: '{query}'")
    print(f"   Top-K: {top_k}")
    
    results = indexer.search(query, top_k=top_k)
    
    print(f"\n📄 Top {top_k} Results:")
    print("-" * 70)
    
    for result in results:
        print(f"\n{result['rank']}. BM25 Score: {result['score']:.4f}")
        print(f"   Title: {result['metadata']['title']}")
        print(f"   Chunk ID: {result['chunk_id']}")
        print(f"   Text preview: {result['text'][:200]}...")
    
    print("-" * 70)


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python build_bm25_index.py corpus_chunks.json")
        sys.exit(1)
    
    corpus_file = sys.argv[1]
    
    # Build index
    indexer = BM25Indexer()
    indexer.build_index(corpus_file)
    
    # Test with sample query
    print("\n" + "="*70)
    test_query = "machine learning algorithms"
    test_search(indexer, test_query, top_k=5)
    print("="*70)
    
    print(f"\n✅ Done! BM25 index saved to bm25_index.pkl")
    print(f"\n💡 To use this index:")
    print(f"   from build_bm25_index import BM25Indexer")
    print(f"   indexer = BM25Indexer.load_index('bm25_index.pkl')")
    print(f"   results = indexer.search('your query', top_k=5)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
