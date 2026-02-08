"""
Wikipedia Text Extraction and Chunking for RAG System
======================================================

This script:
1. Extracts full text from Wikipedia URLs
2. Cleans the text (removes HTML, tables, references)
3. Chunks text into 200-400 tokens with 50-token overlap
4. Creates unique chunk IDs
5. Stores with metadata (URL, title, chunk_id)

Usage:
    python extract_and_chunk.py fixed_urls.json
    python extract_and_chunk.py fixed_urls.json random_urls.json --output corpus_chunks.json
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import sys
import hashlib
from typing import List, Dict, Tuple
from datetime import datetime
import time

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️  Warning: tiktoken not installed. Using approximate word-based tokenization.")
    print("   Install with: pip install tiktoken")


class WikipediaTextExtractor:
    """Extract and clean text from Wikipedia pages"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikipediaRAGCorpus/1.0 (Educational Project; Python/requests)'
        })
        
        # Initialize tokenizer if available
        if TIKTOKEN_AVAILABLE:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        else:
            self.tokenizer = None
    
    def extract_text_from_url(self, url: str) -> Dict:
        """Extract clean text from Wikipedia URL"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title
            title_elem = soup.find('h1', class_='firstHeading')
            title = title_elem.text if title_elem else url.split('/')[-1]
            
            # Get main content
            content_div = soup.find('div', id='mw-content-text')
            
            if not content_div:
                return {'success': False, 'error': 'No content found'}
            
            # Remove unwanted elements
            for element in content_div.find_all([
                'script', 'style', 'sup', 'table', 
                'div.navbox', 'div.reflist', 'div.refbegin',
                'div.thumb', 'div.infobox', 'span.mw-editsection'
            ]):
                element.decompose()
            
            # Get clean text
            text = content_div.get_text(separator=' ', strip=True)
            
            # Clean text
            text = self._clean_text(text)
            
            return {
                'success': True,
                'title': title,
                'text': text,
                'url': url
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove citation markers like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove special Wikipedia markers
        text = re.sub(r'\[edit\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()


class TextChunker:
    """Chunk text into token-sized pieces with overlap"""
    
    def __init__(self, min_tokens: int = 200, max_tokens: int = 400, overlap_tokens: int = 50):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        if TIKTOKEN_AVAILABLE:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: ~0.75 tokens per word
            return int(len(text.split()) * 0.75)
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Chunk text into pieces with token limits and overlap
        
        Returns list of chunks with metadata
        """
        chunks = []
        
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed max, save current chunk
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                # Save chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = self._generate_chunk_id(metadata['url'], chunk_index)
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'url': metadata['url'],
                    'title': metadata['title'],
                    'chunk_index': chunk_index,
                    'text': chunk_text,
                    'token_count': current_tokens
                })
                
                # Calculate overlap
                overlap_text = self._get_overlap(current_chunk, self.overlap_tokens)
                current_chunk = overlap_text
                current_tokens = self.count_tokens(' '.join(current_chunk))
                chunk_index += 1
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Save last chunk if it meets minimum
        if current_chunk and current_tokens >= self.min_tokens:
            chunk_text = ' '.join(current_chunk)
            chunk_id = self._generate_chunk_id(metadata['url'], chunk_index)
            
            chunks.append({
                'chunk_id': chunk_id,
                'url': metadata['url'],
                'title': metadata['title'],
                'chunk_index': chunk_index,
                'text': chunk_text,
                'token_count': current_tokens
            })
        
        return chunks
    
    def _get_overlap(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get last N tokens worth of sentences for overlap"""
        overlap = []
        token_count = 0
        
        # Work backwards from end
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if token_count + sentence_tokens > overlap_tokens:
                break
            overlap.insert(0, sentence)
            token_count += sentence_tokens
        
        return overlap
    
    def _generate_chunk_id(self, url: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        # Create hash from URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{url_hash}_chunk_{chunk_index:04d}"


class CorpusBuilder:
    """Main class to build the RAG corpus"""
    
    def __init__(self):
        self.extractor = WikipediaTextExtractor()
        self.chunker = TextChunker(min_tokens=200, max_tokens=400, overlap_tokens=50)
    
    def process_url(self, url_data: Dict) -> Dict:
        """Process a single URL: extract, clean, chunk"""
        url = url_data['url']
        
        print(f"Processing: {url_data.get('title', url)[:60]}...", end=' ')
        
        # Extract text
        result = self.extractor.extract_text_from_url(url)
        
        if not result['success']:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            return {'success': False, 'url': url, 'error': result.get('error')}
        
        # Chunk text
        metadata = {
            'url': url,
            'title': result['title']
        }
        
        chunks = self.chunker.chunk_text(result['text'], metadata)
        
        print(f"✅ {len(chunks)} chunks ({self.chunker.count_tokens(result['text'])} tokens)")
        
        return {
            'success': True,
            'url': url,
            'title': result['title'],
            'chunks': chunks,
            'total_chunks': len(chunks),
            'original_word_count': url_data.get('word_count', 0)
        }
    
    def build_corpus(self, url_files: List[str], output_file: str = 'corpus_chunks.json'):
        """Build complete corpus from URL files"""
        
        print("\n" + "="*70)
        print("🏗️  Building RAG Corpus")
        print("="*70)
        
        # Load all URLs
        all_urls = []
        for url_file in url_files:
            print(f"\n📄 Loading: {url_file}")
            try:
                with open(url_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    urls = data.get('urls', [])
                    all_urls.extend(urls)
                    print(f"   ✅ Loaded {len(urls)} URLs")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n📊 Total URLs to process: {len(all_urls)}")
        print(f"⏱️  Estimated time: {len(all_urls) * 2 / 60:.1f} minutes")
        print()
        
        input("Press Enter to start processing...")
        print()
        
        # Process each URL
        all_chunks = []
        successful = 0
        failed = 0
        start_time = time.time()
        
        for i, url_data in enumerate(all_urls, 1):
            print(f"[{i}/{len(all_urls)}] ", end='')
            
            result = self.process_url(url_data)
            
            if result['success']:
                all_chunks.extend(result['chunks'])
                successful += 1
            else:
                failed += 1
            
            # Rate limiting
            time.sleep(0.2)
            
            # Progress update every 50 URLs
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(all_urls) - i) / rate
                print(f"\n⏱️  Progress: {i}/{len(all_urls)} | "
                      f"Success: {successful} | Failed: {failed} | "
                      f"ETA: {remaining:.0f}s\n")
        
        # Save corpus
        print(f"\n{'='*70}")
        print("💾 Saving Corpus")
        print(f"{'='*70}")
        
        corpus = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_urls': len(all_urls),
                'successful_urls': successful,
                'failed_urls': failed,
                'total_chunks': len(all_chunks),
                'chunk_config': {
                    'min_tokens': 200,
                    'max_tokens': 400,
                    'overlap_tokens': 50
                },
                'tokenizer': 'tiktoken cl100k_base' if TIKTOKEN_AVAILABLE else 'word-based approximation'
            },
            'chunks': all_chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved corpus to: {output_file}")
        
        # Statistics
        print(f"\n{'='*70}")
        print("📊 CORPUS STATISTICS")
        print(f"{'='*70}")
        print(f"Total URLs processed: {len(all_urls)}")
        print(f"  ✅ Successful: {successful}")
        print(f"  ❌ Failed: {failed}")
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        if all_chunks:
            token_counts = [chunk['token_count'] for chunk in all_chunks]
            print(f"\nChunk token distribution:")
            print(f"  Minimum: {min(token_counts)} tokens")
            print(f"  Maximum: {max(token_counts)} tokens")
            print(f"  Average: {sum(token_counts) / len(token_counts):.0f} tokens")
        
        print(f"\nTotal processing time: {(time.time() - start_time) / 60:.1f} minutes")
        print(f"{'='*70}")
        
        return corpus


def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python extract_and_chunk.py <url_file1.json> [url_file2.json ...] [--output corpus.json]")
        print("\nExamples:")
        print("  python extract_and_chunk.py fixed_urls.json")
        print("  python extract_and_chunk.py fixed_urls.json random_urls.json")
        print("  python extract_and_chunk.py fixed_urls.json random_urls.json --output corpus_chunks.json")
        sys.exit(1)
    
    # Parse arguments
    url_files = []
    output_file = 'corpus_chunks.json'
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--output':
            output_file = sys.argv[i + 1]
            i += 2
        else:
            url_files.append(sys.argv[i])
            i += 1
    
    if not url_files:
        print("❌ No URL files specified!")
        sys.exit(1)
    
    # Build corpus
    builder = CorpusBuilder()
    builder.build_corpus(url_files, output_file)
    
    print(f"\n✅ Done! Your RAG corpus is ready in: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Load corpus_chunks.json in your RAG system")
    print(f"  2. Index chunks in vector database")
    print(f"  3. Use chunk_id to track source documents")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
