"""
Robust Automated Question Generation - Generates ALL Questions in One Run
==========================================================================

Automatically generates 100 Q&A pairs without manual intervention.
Keeps running until target is reached!

Usage:
    python generate_questions_auto.py corpus_chunks.json --num-questions 100
"""

import json
import random
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict
import sys
import time

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except ImportError:
    print("❌ Missing dependencies!")
    print("Install: pip install transformers torch")
    sys.exit(1)


class RobustQuestionGenerator:
    """Generate questions with automatic retry and progress tracking"""
    
    QUESTION_TYPES = {
        'factual': {
            'templates': [
                "Generate a factual question from this text:\n\n{text}\n\nQuestion:",
                "What factual question can be asked about:\n\n{text}\n\nQuestion:",
                "Create a 'what is' question from:\n\n{text}\n\nQuestion:",
                "Ask a factual question about:\n\n{text}\n\nQuestion:",
            ],
            'target_percentage': 30
        },
        'comparative': {
            'templates': [
                "Generate a comparative question from:\n\n{text}\n\nQuestion:",
                "What comparison can be made from:\n\n{text}\n\nQuestion:",
                "Create a 'difference between' question from:\n\n{text}\n\nQuestion:",
                "Ask a comparison question about:\n\n{text}\n\nQuestion:",
            ],
            'target_percentage': 20
        },
        'inferential': {
            'templates': [
                "Generate a 'why' or 'how' question from:\n\n{text}\n\nQuestion:",
                "What cause-and-effect question can be asked:\n\n{text}\n\nQuestion:",
                "Create a reasoning question from:\n\n{text}\n\nQuestion:",
                "Ask an explanatory question about:\n\n{text}\n\nQuestion:",
            ],
            'target_percentage': 30
        },
        'multi-hop': {
            'templates': [
                "Generate a complex multi-step question from:\n\n{text}\n\nQuestion:",
                "What detailed question requires multiple facts:\n\n{text}\n\nQuestion:",
                "Create a comprehensive question from:\n\n{text}\n\nQuestion:",
                "Ask a complex question about:\n\n{text}\n\nQuestion:",
            ],
            'target_percentage': 20
        }
    }
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize generator"""
        print(f"\n🤖 Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        print("   ✅ Model loaded!")
        
        self.generated_questions = set()
        self.used_chunks = defaultdict(int)  # Track chunk usage
    
    def load_corpus(self, corpus_file: str) -> List[Dict]:
        """Load corpus chunks"""
        print(f"\n📂 Loading corpus...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        chunks = corpus.get('chunks', [])
        print(f"   ✅ Loaded {len(chunks)} chunks")
        return chunks
    
    def generate_question(self, text: str, question_type: str, attempt: int = 1) -> str:
        """Generate question with retry logic"""
        
        templates = self.QUESTION_TYPES[question_type]['templates']
        template = random.choice(templates)
        
        # Use different text portions on retries
        start_pos = (attempt - 1) * 200
        text_portion = text[start_pos:start_pos + 500] if len(text) > start_pos else text[:500]
        
        prompt = template.format(text=text_portion)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=100,
                num_beams=4,
                temperature=0.8 + (attempt * 0.1),  # Increase temperature on retries
                do_sample=True,
                top_p=0.9,
                no_repeat_ngram_size=3
            )
        
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Post-process
        if question and not question.endswith('?'):
            question += '?'
        
        return question
    
    def generate_answer(self, text: str, question: str) -> str:
        """Generate answer"""
        
        prompt = f"""Based on this context, answer the question.

Context: {text[:1000]}

Question: {question}

Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=4,
                temperature=0.7,
                do_sample=False,
                no_repeat_ngram_size=3
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return answer
    
    def is_valid_question(self, question: str) -> Tuple[bool, str]:
        """Validate question with reason"""
        
        if not question:
            return False, "Empty question"
        
        # Must be reasonable length
        if len(question) < 15:
            return False, "Too short"
        
        if len(question) > 250:
            return False, "Too long"
        
        # Must end with ?
        if not question.endswith('?'):
            return False, "No question mark"
        
        # Check for duplicate (case insensitive)
        q_lower = question.lower()
        if q_lower in self.generated_questions:
            return False, "Duplicate"
        
        # Must contain question word (relaxed check)
        question_indicators = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 
                              'is', 'are', 'does', 'do', 'can', 'could', 'would', 'should']
        first_words = question.lower().split()[:5]
        if not any(indicator in first_words for indicator in question_indicators):
            return False, "No question word"
        
        return True, "Valid"
    
    def is_valid_answer(self, answer: str) -> Tuple[bool, str]:
        """Validate answer with reason"""
        
        if not answer:
            return False, "Empty answer"
        
        if len(answer) < 10:
            return False, "Too short"
        
        if len(answer) > 1500:
            return False, "Too long"
        
        return True, "Valid"
    
    def generate_single_qa(self, chunk: Dict, question_type: str, max_retries: int = 5) -> Dict:
        """Generate single Q&A pair with retry logic"""
        
        for attempt in range(1, max_retries + 1):
            try:
                # Generate question
                question = self.generate_question(chunk['text'], question_type, attempt)
                
                # Validate question
                valid, reason = self.is_valid_question(question)
                if not valid:
                    if attempt < max_retries:
                        continue
                    else:
                        return None
                
                # Generate answer
                answer = self.generate_answer(chunk['text'], question)
                
                # Validate answer
                valid, reason = self.is_valid_answer(answer)
                if not valid:
                    if attempt < max_retries:
                        continue
                    else:
                        return None
                
                # Success!
                self.generated_questions.add(question.lower())
                self.used_chunks[chunk['chunk_id']] += 1
                
                return {
                    'question': question,
                    'answer': answer,
                    'question_type': question_type,
                    'source_chunk_id': chunk['chunk_id'],
                    'source_title': chunk['title'],
                    'source_url': chunk['url'],
                    'source_text': chunk['text'][:300] + "..."
                }
                
            except Exception as e:
                if attempt >= max_retries:
                    return None
                continue
        
        return None
    
    def generate_all_qa_pairs(self, chunks: List[Dict], num_questions: int = 100) -> List[Dict]:
        """
        Generate ALL Q&A pairs automatically - keeps going until target reached!
        """
        
        print(f"\n" + "="*70)
        print(f"🎯 AUTOMATIC GENERATION: {num_questions} Q&A PAIRS")
        print("="*70)
        print("⏳ This will run until completion - no manual intervention needed!")
        print("="*70)
        
        qa_pairs = []
        
        # Calculate targets for each type
        targets = {}
        for q_type, config in self.QUESTION_TYPES.items():
            targets[q_type] = int(num_questions * config['target_percentage'] / 100)
        
        # Adjust to ensure total = num_questions
        total_target = sum(targets.values())
        if total_target < num_questions:
            targets['factual'] += (num_questions - total_target)
        
        print(f"\n📊 Target Distribution:")
        for q_type, target in targets.items():
            print(f"   {q_type:15s}: {target} questions")
        print()
        
        # Track progress per type
        progress = {q_type: 0 for q_type in self.QUESTION_TYPES.keys()}
        
        # Generate questions by type
        for question_type, target_count in targets.items():
            print(f"\n{'='*70}")
            print(f"📝 Generating {question_type.upper()} questions (target: {target_count})")
            print(f"{'='*70}")
            
            attempts = 0
            max_attempts = target_count * 20  # Allow many attempts
            
            while progress[question_type] < target_count and attempts < max_attempts:
                attempts += 1
                
                # Select random chunk (prefer less-used chunks)
                available_chunks = [c for c in chunks if self.used_chunks[c['chunk_id']] < 3]
                if not available_chunks:
                    available_chunks = chunks  # Use any if all heavily used
                
                chunk = random.choice(available_chunks)
                
                # Generate Q&A
                qa_data = self.generate_single_qa(chunk, question_type)
                
                if qa_data:
                    # Success!
                    qa_id = f"qa_{len(qa_pairs) + 1:03d}"
                    qa_data['id'] = qa_id
                    qa_pairs.append(qa_data)
                    progress[question_type] += 1
                    
                    # Show progress
                    total_progress = len(qa_pairs)
                    percentage = (total_progress / num_questions) * 100
                    print(f"   ✅ {qa_id} | Type: {question_type:12s} | Progress: {total_progress}/{num_questions} ({percentage:5.1f}%)")
                    print(f"      Q: {qa_data['question'][:80]}...")
                
                # Save checkpoint every 10 questions
                if len(qa_pairs) % 10 == 0 and len(qa_pairs) > 0:
                    self._save_checkpoint(qa_pairs, f"checkpoint_{len(qa_pairs)}.json")
            
            print(f"   ✅ Completed {progress[question_type]}/{target_count} {question_type} questions")
        
        print(f"\n{'='*70}")
        print(f"🎉 GENERATION COMPLETE!")
        print(f"{'='*70}")
        print(f"✅ Total generated: {len(qa_pairs)} Q&A pairs")
        print(f"✅ Target achieved: {len(qa_pairs) >= num_questions}")
        
        # Show final distribution
        type_counts = defaultdict(int)
        for qa in qa_pairs:
            type_counts[qa['question_type']] += 1
        
        print(f"\n📊 Final Distribution:")
        for q_type, count in sorted(type_counts.items()):
            percentage = (count / len(qa_pairs)) * 100
            print(f"   {q_type:15s}: {count:3d} ({percentage:5.1f}%)")
        
        return qa_pairs
    
    def _save_checkpoint(self, qa_pairs: List[Dict], filename: str):
        """Save checkpoint"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({'qa_pairs': qa_pairs}, f, indent=2, ensure_ascii=False)
        except:
            pass  # Silent fail for checkpoints
    
    def save_qa_pairs(self, qa_pairs: List[Dict], output_file: str):
        """Save final Q&A pairs"""
        
        type_counts = defaultdict(int)
        for qa in qa_pairs:
            type_counts[qa['question_type']] += 1
        
        output = {
            'metadata': {
                'total_questions': len(qa_pairs),
                'question_types': dict(type_counts),
                'generation_method': 'Flan-T5-base (robust automatic)',
                'source': 'Wikipedia corpus'
            },
            'qa_pairs': qa_pairs
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved {len(qa_pairs)} Q&A pairs to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Robust automatic Q&A generation")
    parser.add_argument('corpus_file', help="Path to corpus_chunks.json")
    parser.add_argument('--output', default='qa_pairs.json', help="Output file")
    parser.add_argument('--num-questions', type=int, default=100, help="Number of questions")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("\n" + "="*70)
    print("🚀 ROBUST AUTOMATIC Q&A GENERATION")
    print("="*70)
    print(f"📋 Target: {args.num_questions} questions")
    print(f"📁 Output: {args.output}")
    print(f"🎲 Seed: {args.seed}")
    print("="*70)
    
    # Initialize
    generator = RobustQuestionGenerator()
    
    # Load corpus
    chunks = generator.load_corpus(args.corpus_file)
    
    # Generate ALL questions automatically
    qa_pairs = generator.generate_all_qa_pairs(chunks, num_questions=args.num_questions)
    
    # Save
    generator.save_qa_pairs(qa_pairs, args.output)
    
    print(f"\n✅ DONE! Generated {len(qa_pairs)} Q&A pairs")
    print(f"📁 Saved to: {args.output}")
    print(f"\n💡 View results:")
    print(f"   python view_questions.py {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Checkpoint files saved.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
