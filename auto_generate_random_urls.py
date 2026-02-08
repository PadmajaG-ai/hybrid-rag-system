"""
Automated Random Wikipedia URL Generator
Automatically generates 300 random Wikipedia URLs with no user prompts
- Loads existing URLs if available and continues from there
- Keeps generating until target is reached
- Saves metadata automatically
- Can be run multiple times safely
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Set
import os

class AutoRandomURLGenerator:
    def __init__(self, min_words: int = 200, target_count: int = 300):
        self.min_words = min_words
        self.target_count = target_count
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()

        # Add User-Agent header (prevents 403 Forbidden errors)
        self.session.headers.update({
            'User-Agent': 'WikipediaRandomURLGenerator/1.0 (Educational/Research Project; Python/requests)'
        })

    def load_existing_urls(self, filename: str = 'random_urls.json') -> tuple:
        """Load existing URLs from JSON file if it exists"""
        if not os.path.exists(filename):
            return [], set()

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            urls = data.get('urls', [])
            titles = {url['title'] for url in urls}

            print(f"✅ Loaded {len(urls)} existing URLs from {filename}")
            return urls, titles
        except Exception as e:
            print(f"⚠️  Error loading existing file: {e}")
            print("Starting fresh...")
            return [], set()

    def get_random_pages(self, count: int = 50) -> List[Dict]:
        """Fetch random Wikipedia pages in batch"""
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': count
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data['query']['random']
        except Exception as e:
            print(f"❌ Error fetching random pages: {e}")
            return []

    def get_page_content(self, page_id: int) -> Dict:
        """Get page content and word count for a specific page"""
        params = {
            'action': 'query',
            'format': 'json',
            'pageids': page_id,
            'prop': 'extracts|info',
            'exintro': False,
            'explaintext': True,
            'inprop': 'url'
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            page_data = data['query']['pages'][str(page_id)]

            if 'extract' in page_data:
                text = page_data['extract']
                word_count = len(text.split())

                return {
                    'page_id': page_id,
                    'title': page_data.get('title', ''),
                    'url': page_data.get('fullurl', ''),
                    'word_count': word_count,
                    'valid': word_count >= self.min_words
                }
            else:
                return {'page_id': page_id, 'valid': False}

        except Exception as e:
            return {'page_id': page_id, 'valid': False}

    def generate_urls(self, batch_size: int = 50) -> List[Dict]:
        """
        Automatically generate URLs until target is reached
        Loads existing URLs and continues from there if available
        Keeps retrying until target is reached - no manual re-runs needed
        """
        # Load existing URLs
        valid_urls, seen_titles = self.load_existing_urls('random_urls.json')

        initial_count = len(valid_urls)

        if initial_count >= self.target_count:
            print(f"\n✅ Already have {initial_count} URLs (target: {self.target_count})")
            print("No additional URLs needed.")
            return valid_urls

        needed = self.target_count - initial_count

        print(f"\n{'='*70}")
        if initial_count > 0:
            print(f"Continuing from {initial_count} existing URLs")
            print(f"Generating {needed} more URLs to reach {self.target_count}")
        else:
            print(f"Generating {self.target_count} random Wikipedia URLs")
        print(f"Minimum {self.min_words} words per page")
        print(f"Auto-retry enabled - will keep trying until target is reached")
        print(f"{'='*70}\n")

        overall_start_time = time.time()
        retry_round = 0

        # Outer loop: Keep retrying until we reach the target
        while len(valid_urls) < self.target_count:
            retry_round += 1
            current_count = len(valid_urls)
            remaining = self.target_count - current_count

            if retry_round > 1:
                print(f"\n{'='*70}")
                print(f"🔄 Auto-retry Round {retry_round}")
                print(f"Current: {current_count} | Still need: {remaining}")
                print(f"{'='*70}\n")

            attempts = 0
            max_attempts_per_round = remaining * 5
            consecutive_failures = 0
            round_start_time = time.time()
            last_count = current_count

            # Inner loop: Generate URLs for this round
            while len(valid_urls) < self.target_count and attempts < max_attempts_per_round:
                # Fetch batch of random pages
                random_pages = self.get_random_pages(batch_size)

                if not random_pages:
                    print("⚠️  Failed to fetch random pages. Retrying in 2 seconds...")
                    time.sleep(2)
                    attempts += batch_size
                    consecutive_failures += 1

                    if consecutive_failures >= 5:
                        print(f"\n⚠️  Network issues detected. Waiting 5 seconds before retry...")
                        time.sleep(5)
                        consecutive_failures = 0  # Reset and try again
                        continue
                    continue

                consecutive_failures = 0

                # Check each page
                for page in random_pages:
                    if len(valid_urls) >= self.target_count:
                        break

                    # Skip duplicates
                    if page['title'] in seen_titles:
                        continue

                    page_info = self.get_page_content(page['id'])

                    if page_info['valid']:
                        seen_titles.add(page_info['title'])
                        valid_urls.append({
                            'url': page_info['url'],
                            'title': page_info['title'],
                            'word_count': page_info['word_count']
                        })

                        # Show progress
                        status = f"✓ [{len(valid_urls)}/{self.target_count}]"
                        words = f"({page_info['word_count']} words)"
                        print(f"{status} {page_info['title'][:50]}... {words}")

                        # Auto-save every 25 URLs to prevent data loss
                        if len(valid_urls) % 25 == 0:
                            self.save_to_json(valid_urls, 'random_urls.json', silent=True)

                    # Rate limiting
                    time.sleep(0.1)
                    attempts += 1

                # Progress update every 50 URLs
                if len(valid_urls) % 50 == 0 and len(valid_urls) > last_count:
                    elapsed = time.time() - overall_start_time
                    total_new = len(valid_urls) - initial_count
                    rate = total_new / elapsed if elapsed > 0 else 0
                    eta = (self.target_count - len(valid_urls)) / rate if rate > 0 else 0
                    print(f"\n⏱️  Progress: {len(valid_urls)}/{self.target_count} | "
                          f"Total new: {total_new} | ETA: {eta:.0f}s\n")
                    last_count = len(valid_urls)

            # Check if we made progress this round
            progress_this_round = len(valid_urls) - current_count
            if progress_this_round > 0:
                round_time = time.time() - round_start_time
                print(f"\n✅ Round {retry_round} complete: Added {progress_this_round} URLs in {round_time:.1f}s")
                # Save progress after each round
                self.save_to_json(valid_urls, 'random_urls.json', silent=True)
            else:
                print(f"\n⚠️  Round {retry_round}: No progress made. Waiting 3 seconds before retry...")
                time.sleep(3)

        total_time = time.time() - overall_start_time
        total_new = len(valid_urls) - initial_count

        print(f"\n🎉 TARGET REACHED! Added {total_new} new URLs in {total_time:.1f} seconds")
        print(f"Total rounds needed: {retry_round}")

        return valid_urls

    def save_to_json(self, urls: List[Dict], filename: str = 'random_urls.json', silent: bool = False):
        """Save URLs to JSON file with metadata"""
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_urls': len(urls),
                'min_words_requirement': self.min_words,
                'target_count': self.target_count
            },
            'urls': urls
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        if not silent:
            print(f"✅ Saved {len(urls)} URLs to {filename}")

    def save_to_txt(self, urls: List[Dict], filename: str = 'random_urls.txt'):
        """Save URLs to simple text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Random Wikipedia URLs - Generated on {datetime.now().isoformat()}\n")
            f.write(f"# Total URLs: {len(urls)}\n")
            f.write(f"# Minimum words per page: {self.min_words}\n\n")

            for item in urls:
                f.write(f"{item['url']}\n")

        print(f"✅ Saved {len(urls)} URLs to {filename}")

    def print_statistics(self, urls: List[Dict]):
        """Print detailed statistics"""
        print("\n" + "=" * 70)
        print("📊 STATISTICS")
        print("=" * 70)
        print(f"Total URLs: {len(urls)}")

        if urls:
            word_counts = [url['word_count'] for url in urls]
            print(f"\n📝 Word Count:")
            print(f"   Minimum: {min(word_counts)} words")
            print(f"   Maximum: {max(word_counts)} words")
            print(f"   Average: {sum(word_counts) / len(urls):.0f} words")

            # Check if all meet requirement
            below_min = [url for url in urls if url['word_count'] < self.min_words]
            if below_min:
                print(f"\n⚠️  {len(below_min)} URLs below {self.min_words} words (should not happen!)")
            else:
                print(f"\n✅ All URLs meet the {self.min_words} word minimum!")

    def run(self):
        """Main execution - fully automated with auto-retry"""
        print("\n" + "=" * 70)
        print("Automated Random Wikipedia URL Generator")
        print("With Auto-Retry Until Target Reached")
        print("=" * 70)

        # Create backup if file exists
        if os.path.exists('random_urls.json'):
            backup_name = f'random_urls_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            import shutil
            shutil.copy('random_urls.json', backup_name)
            print(f"💾 Backed up existing file to {backup_name}")

        # Generate URLs (will auto-retry until target is reached)
        random_urls = self.generate_urls(batch_size=50)

        # Save final results to files
        print("\n💾 Saving final files...")
        self.save_to_json(random_urls, 'random_urls.json')
        self.save_to_txt(random_urls, 'random_urls.txt')

        # Print statistics
        self.print_statistics(random_urls)

        print("\n" + "=" * 70)
        print("📁 FILES CREATED/UPDATED")
        print("=" * 70)
        print("  ✅ random_urls.json (structured data with metadata)")
        print("  ✅ random_urls.txt (plain text, one URL per line)")

        print("\n" + "=" * 70)
        print("✅ DONE! All URLs generated successfully.")
        print("=" * 70)


def main():
    """Entry point - completely automated, no user prompts"""
    try:
        generator = AutoRandomURLGenerator(min_words=200, target_count=300)
        generator.run()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user. Exiting...")
        print("Your progress has been saved. Run the script again to continue.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nYou can run this script again to retry.")


if __name__ == "__main__":
    main()
