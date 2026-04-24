"""
add_urls.py — Add website URLs to your existing FAISS index anytime.

Usage:
    python add_urls.py --urls https://pesce.ac.in/
    python add_urls.py --file urls.txt        # one URL per line
    python add_urls.py --crawl https://pesce.ac.in/ --depth 2
"""

import argparse
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = "faiss_index"
EMBEDDINGS = FakeEmbeddings(size=384)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Real browser headers to avoid being blocked
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# ── Scraping ──────────────────────────────────────────────────────────────────

def scrape_url(url: str) -> str:
    """Fetch and extract clean text from a URL with debug info."""
    try:
        print(f"  → Sending request to {url}")
        response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        print(f"  → Status code: {response.status_code}")
        print(f"  → Content length: {len(response.text)} characters")

        if response.status_code != 200:
            print(f"  ⚠️  Non-200 status, skipping.")
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "noscript", "iframe", "svg", "form"]):
            tag.decompose()

        # Try to get main content first, fall back to full body
        main = (
            soup.find("main") or
            soup.find("div", {"id": "content"}) or
            soup.find("div", {"class": "content"}) or
            soup.find("div", {"id": "main"}) or
            soup.find("article") or
            soup.find("body")
        )

        if main:
            text = main.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")

        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
        result = "\n".join(lines)

        print(f"  → Extracted {len(result)} characters of clean text")

        if len(result) < 100:
            print(f"  ⚠️  Very little text extracted — site may use JavaScript rendering.")

        return result

    except requests.exceptions.SSLError:
        print(f"  ⚠️  SSL error, retrying without SSL verification...")
        try:
            response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
            return "\n".join(lines)
        except Exception as e2:
            print(f"  ❌ Retry also failed: {e2}")
            return ""

    except Exception as e:
        print(f"  ❌ Failed to scrape {url}: {e}")
        return ""


def crawl_site(start_url: str, depth: int = 1) -> list:
    visited = set()
    to_visit = [(start_url, 0)]
    found_urls = []
    base_domain = urlparse(start_url).netloc

    while to_visit:
        url, current_depth = to_visit.pop(0)
        if url in visited or current_depth > depth:
            continue
        visited.add(url)
        found_urls.append(url)
        print(f"  🔍 Crawling (depth {current_depth}): {url}")

        try:
            response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = urljoin(url, a_tag["href"])
                parsed = urlparse(href)
                if parsed.netloc == base_domain and href not in visited:
                    clean = parsed._replace(fragment="").geturl()
                    to_visit.append((clean, current_depth + 1))
            time.sleep(0.5)
        except Exception:
            pass

    return found_urls


# ── FAISS Helpers ─────────────────────────────────────────────────────────────

def load_or_create_db():
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"📂 Loading existing FAISS index from '{FAISS_INDEX_PATH}'...")
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            EMBEDDINGS,
            allow_dangerous_deserialization=True
        )
        print("✅ Existing index loaded.")
        return db
    else:
        print("🆕 No existing index found — a new one will be created.")
        return None


def add_urls_to_index(urls: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_docs = []
    for url in urls:
        print(f"\n🌐 Scraping: {url}")
        text = scrape_url(url)
        if not text:
            print(f"  ⚠️  No content extracted from {url}, skipping.")
            continue

        chunks = splitter.split_text(text)
        docs = [
            Document(page_content=chunk, metadata={"source": url})
            for chunk in chunks
        ]
        all_docs.extend(docs)
        print(f"  ✅ {len(docs)} chunks added from {url}")

    if not all_docs:
        print("\n❌ No content scraped from any URL.")
        print("💡 Possible reasons:")
        print("   1. The website uses JavaScript to load content (needs Selenium)")
        print("   2. The website is blocking automated requests")
        print("   3. Check your internet connection")
        return

    print(f"\n📦 Total chunks to add: {len(all_docs)}")
    db = load_or_create_db()

    if db is None:
        db = FAISS.from_documents(all_docs, EMBEDDINGS)
        print(f"✅ New FAISS index created with {len(all_docs)} chunks.")
    else:
        db.add_documents(all_docs)
        print(f"✅ Added {len(all_docs)} new chunks to existing index.")

    db.save_local(FAISS_INDEX_PATH)
    print(f"💾 Index saved to '{FAISS_INDEX_PATH}'")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    parser = argparse.ArgumentParser(description="Add URLs to FAISS index")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--urls", nargs="+", metavar="URL")
    group.add_argument("--file", metavar="FILE")
    group.add_argument("--crawl", metavar="START_URL")
    parser.add_argument("--depth", type=int, default=1)

    args = parser.parse_args()

    if args.urls:
        urls = args.urls
    elif args.file:
        with open(args.file, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
        print(f"📄 Loaded {len(urls)} URLs from '{args.file}'")
    elif args.crawl:
        print(f"🕷️  Crawling '{args.crawl}' with depth={args.depth}...")
        urls = crawl_site(args.crawl, depth=args.depth)
        print(f"🔗 Found {len(urls)} pages to scrape")

    add_urls_to_index(urls)


if __name__ == "__main__":
    main()