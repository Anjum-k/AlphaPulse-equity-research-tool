import unittest
from core.utils import is_valid_url, clean_url, parse_urls_from_text, extract_domain


class TestEquityResearchCore(unittest.TestCase):

    def test_url_validation(self):
        self.assertTrue(is_valid_url("https://finance.yahoo.com/news/apple"))
        self.assertTrue(is_valid_url("http://reuters.com/business"))
        self.assertFalse(is_valid_url("not_a_url"))
        self.assertFalse(is_valid_url("ftp://example.com"))

    def test_clean_url(self):
        self.assertEqual(clean_url("  https://cnbc.com/article/  "), "https://cnbc.com/article")

    def test_domain_extraction(self):
        self.assertEqual(extract_domain("https://www.reuters.com/markets"), "reuters.com")
        self.assertEqual(extract_domain("https://finance.yahoo.com/news"), "finance.yahoo.com")

    def test_batch_url_parsing(self):
        raw_text = """
        # Morning Equity Digest
        https://finance.yahoo.com/news/apple-q4
        https://www.bloomberg.com/news/articles/nvidia

        # Duplicate link
        https://finance.yahoo.com/news/apple-q4/
        invalid-line-here
        https://www.reuters.com/markets/asia
        """
        urls = parse_urls_from_text(raw_text)
        self.assertEqual(len(urls), 3)
        self.assertIn("https://finance.yahoo.com/news/apple-q4", urls)
        self.assertIn("https://www.bloomberg.com/news/articles/nvidia", urls)
        self.assertIn("https://www.reuters.com/markets/asia", urls)


if __name__ == "__main__":
    unittest.main()
