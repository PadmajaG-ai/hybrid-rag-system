"""
Dense Vector Indexing with ChromaDB
====================================

This script:
1. Loads your corpus_chunks.json
2. Embeds each chunk using all-MiniLM-L6-v2 (sentence-transformers)
3. Stores embeddings in ChromaDB vector database
4. Enables retrieval of top-K similar chunks via cosine similarity

Usage:
    python build_vector_index.py corpus_chunks.json
    python build_vector_index.py corpus_chunks.json --collection wikipedia_rag
"""

import json
import sys
import argparse
from typing import List, Dict
from datetime import datetime
import time

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed!")
    print("Install with: pip install sentence-transformers")
    sys.exit(1)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Error: chromadb not installed!")
    print("Install with: pip install chromadb")
    sys.exit(1)


class DenseVectorIndexer:
    """Build dense vector index using all-MiniLM-L6-v2 and ChromaDB"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", collection_name: str = "wikipedia_corpus"):
        """
        Initialize indexer
        
        Args:
            model_name: Sentence transformer model name
            collection_name: ChromaDB collection name
        """
        print(f"\n Initializing Dense Vector Indexer")
        print(f"   Model: {model_name}")
        print(f"   Collection: {collection_name}")
        
        # Load sentence transformer model
        print(f"\n Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Initialize ChromaDB client with PERSISTENT storage
        print(f"\n Initializing ChromaDB client (persistent storage)")
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )
        
        # Create or get collection
        self.collection_name = collection_name
        self.collection = None
        
    def load_corpus(self, corpus_file: str) -> Dict:
        """Load corpus from JSON file"""
        print(f"\n Loading corpus from: {corpus_file}")
        
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            
            chunks = corpus.get('chunks', [])
            metadata = corpus.get('metadata', {})
            
            print(f"   Loaded {len(chunks)} chunks")
            print(f"   Total URLs: {metadata.get('total_urls', 'N/A')}")
            print(f"   Successful URLs: {metadata.get('successful_urls', 'N/A')}")
            
            return corpus
            
        except FileNotFoundError:
            print(f" File not found: {corpus_file}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f" Invalid JSON file")
            sys.exit(1)
    
    def create_collection(self, reset: bool = False):
        """Create or reset ChromaDB collection"""
        
        if reset:
            print(f"\n Resetting collection: {self.collection_name}")
            try:
                self.client.delete_collection(self.collection_name)
                print(f" Old collection deleted")
            except:
                pass
        
        print(f"\n Creating collection: {self.collection_name}")
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Wikipedia corpus for RAG",
                "model": "all-MiniLM-L6-v2",
                "created_at": datetime.now().isoformat()
            }
        )
        
        print(f"Collection ready")
    
    def embed_chunks(self, chunks: List[Dict], batch_size: int = 32) -> List[List[float]]:
        """
        Create embeddings for all chunks
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Number of chunks to process at once
        
        Returns:
            List of embedding vectors
        """
        print(f"\nCreating embeddings for {len(chunks)} chunks")
        print(f"   Batch size: {batch_size}")
        print(f"   Estimated time: {len(chunks) / batch_size * 0.5:.1f} seconds")
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings in batches
        embeddings = []
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
            
            # Progress update
            if (i + batch_size) % (batch_size * 10) == 0:
                progress = min(i + batch_size, len(texts))
                elapsed = time.time() - start_time
                rate = progress / elapsed
                remaining = (len(texts) - progress) / rate
                print(f"   Progress: {progress}/{len(texts)} | ETA: {remaining:.0f}s")
        
        elapsed = time.time() - start_time
        print(f" Created {len(embeddings)} embeddings in {elapsed:.1f}s")
        print(f" Rate: {len(embeddings) / elapsed:.1f} embeddings/second")
        
        return embeddings
    
    def index_chunks(self, chunks: List[Dict], embeddings: List[List[float]], batch_size: int = 100):
        """
        Index chunks with embeddings in ChromaDB
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
            batch_size: Batch size for indexing
        """
        print(f"\n Indexing {len(chunks)} chunks in ChromaDB")
        print(f"   Batch size: {batch_size}")
        
        start_time = time.time()
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            # Prepare data for ChromaDB
            ids = [chunk['chunk_id'] for chunk in batch_chunks]
            documents = [chunk['text'] for chunk in batch_chunks]
            metadatas = [
                {
                    'title': chunk['title'],
                    'url': chunk['url'],
                    'chunk_index': chunk['chunk_index'],
                    'token_count': chunk['token_count']
                }
                for chunk in batch_chunks
            ]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            # Progress update
            if (i + batch_size) % (batch_size * 5) == 0:
                progress = min(i + batch_size, len(chunks))
                print(f"   Progress: {progress}/{len(chunks)}")
        
        elapsed = time.time() - start_time
        print(f"   Indexed {len(chunks)} chunks in {elapsed:.1f}s")
        
        # Verify
        count = self.collection.count()
        print(f"   Collection now contains {count} documents")
    
    def build_index(self, corpus_file: str, reset: bool = True, batch_size: int = 32):
        """
        Complete pipeline: Load corpus → Embed → Index
        
        Args:
            corpus_file: Path to corpus_chunks.json
            reset: Whether to reset existing collection
            batch_size: Batch size for embedding
        """
        print("\n" + "="*70)
        print("🚀 Building Dense Vector Index")
        print("="*70)
        
        # Load corpus
        corpus = self.load_corpus(corpus_file)
        chunks = corpus['chunks']
        
        # Create collection
        self.create_collection(reset=reset)
        
        # Create embeddings
        embeddings = self.embed_chunks(chunks, batch_size=batch_size)
        
        # Index in ChromaDB
        self.index_chunks(chunks, embeddings, batch_size=100)
        
        # Summary
        print("\n" + "="*70)
        print(" INDEXING COMPLETE")
        print("="*70)
        print(f" Statistics:")
        print(f"   Total chunks indexed: {len(chunks)}")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        print(f"   Collection name: {self.collection_name}")
        print(f"   Model: all-MiniLM-L6-v2")
        print(f"   Similarity metric: Cosine similarity (default in ChromaDB)")
        print("="*70)
        
        return self.collection


def test_retrieval(collection, model, query: str, top_k: int = 5):
    """Test retrieval with a sample query"""
    
    print(f"\n Testing Retrieval")
    print(f"   Query: '{query}'")
    print(f"   Top-K: {top_k}")
    
    # Embed query
    query_embedding = model.encode([query])[0].tolist()
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Display results
    print(f"\n Top {top_k} Results:")
    print("-" * 70)
    
    for i in range(len(results['ids'][0])):
        chunk_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        similarity = 1 - distance  # Convert distance to similarity
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]
        
        print(f"\n{i+1}. Similarity: {similarity:.4f}")
        print(f"   Title: {metadata['title']}")
        print(f"   Chunk ID: {chunk_id}")
        print(f"   URL: {metadata['url']}")
        print(f"   Text preview: {document[:200]}...")
    
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Build dense vector index using all-MiniLM-L6-v2 and ChromaDB'
    )
    parser.add_argument('corpus_file', help='Path to corpus_chunks.json')
    parser.add_argument('--collection', default='wikipedia_corpus',
                       help='ChromaDB collection name (default: wikipedia_corpus)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding (default: 32)')
    parser.add_argument('--no-reset', action='store_true',
                       help='Do not reset existing collection')
    parser.add_argument('--test-query', type=str,
                       help='Test query after indexing')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results for test query (default: 5)')
    
    args = parser.parse_args()
    
    # Build index
    indexer = DenseVectorIndexer(
        model_name="all-MiniLM-L6-v2",
        collection_name=args.collection
    )
    
    collection = indexer.build_index(
        corpus_file=args.corpus_file,
        reset=not args.no_reset,
        batch_size=args.batch_size
    )
    
    # Test query if provided
    if args.test_query:
        test_retrieval(
            collection=collection,
            model=indexer.model,
            query=args.test_query,
            top_k=args.top_k
        )
    
    print(f"\n Done! Your vector index is ready.")
    print(f"\n To use this index:")
    print(f"   1. Load collection: client.get_collection('{args.collection}')")
    print(f"   2. Query with: collection.query(query_embeddings=[...], n_results=K)")
    print(f"   3. See query_vector_index.py for complete examples")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
