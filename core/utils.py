import re
from urllib.parse import urlparse
from typing import List, Set


def is_valid_url(url: str) -> bool:
    """Validate whether a string is a well-formed HTTP/HTTPS URL."""
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def clean_url(url: str) -> str:
    """Clean whitespace and trailing slashes from a URL."""
    return url.strip().rstrip("/")


def extract_domain(url: str) -> str:
    """Extract clean domain name for display (e.g., 'reuters.com')."""
    try:
        domain = urlparse(url).netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return url


def parse_urls_from_text(raw_text: str) -> List[str]:
    """
    Parse URLs from a multi-line string or file content.
    Handles comments (lines starting with #), empty lines, and inline whitespace.
    Deduplicates URLs while preserving original order.
    """
    urls: List[str] = []
    seen: Set[str] = set()

    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue

        # Extract any potential URL in the line using regex or direct check
        candidate_urls = re.findall(r"https?://[^\s<>\"']+", cleaned)
        if candidate_urls:
            for cand in candidate_urls:
                cleaned_cand = clean_url(cand)
                if is_valid_url(cleaned_cand) and cleaned_cand not in seen:
                    seen.add(cleaned_cand)
                    urls.append(cleaned_cand)
        elif is_valid_url(cleaned):
            cleaned_url = clean_url(cleaned)
            if cleaned_url not in seen:
                seen.add(cleaned_url)
                urls.append(cleaned_url)

    return urls
