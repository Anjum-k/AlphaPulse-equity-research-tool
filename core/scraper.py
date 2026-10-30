import logging
from typing import List, Optional
import requests
# pyrefly: ignore[missing-import]
from bs4 import BeautifulSoup
try:
    from langchain_core.documents import Document
except ImportError:
    class Document:  # type: ignore
        """Lightweight fallback Document if langchain is not yet installed."""
        def __init__(self, page_content: str, metadata: Optional[dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

        def __repr__(self) -> str:
            return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

from core.config import DEFAULT_USER_AGENT
from core.utils import clean_url, extract_domain

logger = logging.getLogger(__name__)


class NewsArticleScraper:
    """Robust scraper for financial news articles with clean extraction and error handling."""

    def __init__(self, timeout: int = 15, user_agent: Optional[str] = None):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent or DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        })

    def scrape_url(self, url: str) -> Optional[Document]:
        """
        Scrape a single news article URL and return a LangChain Document.
        Returns None if fetching or parsing fails.
        """
        cleaned_url = clean_url(url)
        try:
            response = self.session.get(cleaned_url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove noise: scripts, styles, iframes, navigation, footers, svg
            for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript", "svg"]):
                tag.decompose()

            # Extract title
            title = None
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()
            elif soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)

            if not title:
                title = f"Article from {extract_domain(cleaned_url)}"

            # Extract publication time if available
            published_time = None
            time_meta = soup.find("meta", property="article:published_time") or soup.find("meta", attrs={"name": "date"})
            if time_meta and time_meta.get("content"):
                published_time = time_meta["content"].strip()

            # Extract main article body
            article_body = soup.find("article") or soup.find("main") or soup.find("div", class_=lambda c: c and any(k in c.lower() for k in ["article-body", "article-content", "caas-body", "story-content", "entry-content"]))

            paragraphs: List[str] = []
            if article_body:
                paragraphs = [p.get_text(separator=" ", strip=True) for p in article_body.find_all("p")]
            
            # Fallback to all paragraphs in the document
            if not paragraphs or sum(len(p) for p in paragraphs) < 200:
                paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]

            # Filter out very short boilerplate or disclaimers
            valid_paragraphs = [p for p in paragraphs if len(p.split()) > 5]
            full_text = "\n\n".join(valid_paragraphs)

            if not full_text or len(full_text) < 100:
                logger.warning("Extracted text too short for URL: %s", cleaned_url)
                return None

            metadata = {
                "source": cleaned_url,
                "title": title,
                "domain": extract_domain(cleaned_url),
            }
            if published_time:
                metadata["published_time"] = published_time

            return Document(page_content=full_text, metadata=metadata)

        except requests.RequestException as e:
            logger.error("HTTP error fetching %s: %s", cleaned_url, str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error scraping %s: %s", cleaned_url, str(e))
            return None

    def scrape_multiple_urls(self, urls: List[str], progress_callback=None) -> List[Document]:
        """
        Scrape multiple URLs with optional progress callback: callback(current_index, total, url, success).
        """
        documents: List[Document] = []
        total = len(urls)

        for idx, url in enumerate(urls, start=1):
            doc = self.scrape_url(url)
            success = doc is not None
            if success:
                documents.append(doc)

            if progress_callback:
                progress_callback(idx, total, url, success)

        return documents


if __name__ == "__main__":
    import sys
    print("Testing NewsArticleScraper on sample URL...")
    test_url = "https://finance.yahoo.com/news/apple-reports-fourth-quarter-results-203000624.html"
    scraper = NewsArticleScraper()
    doc = scraper.scrape_url(test_url)
    if doc:
        print(f"Scraped Title: {doc.metadata.get('title')}")
        print(f"Domain: {doc.metadata.get('domain')}")
        print(f"Characters Extracted: {len(doc.page_content)}")
        print(f"Excerpt: {doc.page_content[:200]}...")
    else:
        print("Failed to scrape test URL or returned empty content.")
