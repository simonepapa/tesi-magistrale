#!/usr/bin/env python3
"""
News Scraper Template
=====================
This is a template for creating news scrapers.
Customize the selectors and logic based on your target website.

Output format:
Each article should be a dict with keys:
- link: URL of the article
- title: Article title
- date: Publication date (YYYY-MM-DD format preferred)
- content: Full article text
- quartiere: (optional) Neighborhood/category

Articles are saved as JSON files in: scraper/news/news_{source_name}/
"""

import json
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Optional


# CONFIGURATION - Customize these for your target website
BASE_URL = "https://www.example.com"  # Base URL of the news website
SOURCE_NAME = "example"  # Used for output folder name

# CSS selectors for article list page
ARTICLE_LIST_SELECTOR = "article.news-item"  # Selector for article containers
ARTICLE_LINK_SELECTOR = "a.article-link"  # Selector for article links
PAGINATION_SELECTOR = "a.next-page"  # Selector for next page link

# CSS selectors for article detail page  
TITLE_SELECTOR = "h1.article-title"
DATE_SELECTOR = "time.publish-date"
CONTENT_SELECTOR = "div.article-body"

# Request settings
REQUEST_DELAY = 1.0  # Seconds between requests
REQUEST_TIMEOUT = 30
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7',
}

# Categories/neighborhoods to scrape
CATEGORIES = [
    "category1",
    "category2",
    "category3",
]


# SCRAPER CLASS
class NewsScraper:
    """Generic news scraper class."""
    
    def __init__(self):
        """Initialize scraper with session."""
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.output_dir = Path(__file__).parent / "news" / f"news_{SOURCE_NAME}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page.
        
        :param url: str: URL to fetch
        :returns: Optional[BeautifulSoup]: Parsed HTML or None on error
        """
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            time.sleep(REQUEST_DELAY)  # Rate limiting
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def parse_date(self, date_str: str) -> str:
        """Parse date string to YYYY-MM-DD format.
        
        :param date_str: str: Date string in various formats
        :returns: str: Date in YYYY-MM-DD format
        """
        # Customize date parsing based on website's date format
        formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%d %B %Y',  # "15 gennaio 2025"
            '%Y-%m-%dT%H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Return original if parsing fails
        return date_str
    
    def get_article_links(self, category_url: str) -> List[str]:
        """Get all article links from a category/list page.
        
        :param category_url: str: URL of the category page
        :returns: List[str]: List of article URLs
        """
        links = []
        current_url = category_url
        
        while current_url:
            soup = self.fetch_page(current_url)
            if not soup:
                break
            
            # Find article containers
            articles = soup.select(ARTICLE_LIST_SELECTOR)
            for article in articles:
                link_elem = article.select_one(ARTICLE_LINK_SELECTOR)
                if link_elem and link_elem.get('href'):
                    href = link_elem['href']
                    # Handle relative URLs
                    if href.startswith('/'):
                        href = BASE_URL + href
                    links.append(href)
            
            # Check for pagination
            next_page = soup.select_one(PAGINATION_SELECTOR)
            if next_page and next_page.get('href'):
                next_href = next_page['href']
                if next_href.startswith('/'):
                    next_href = BASE_URL + next_href
                current_url = next_href
            else:
                current_url = None
        
        return links
    
    def scrape_article(self, url: str) -> Optional[dict]:
        """Scrape a single article page.
        
        :param url: str: Article URL
        :returns: Optional[dict]: Article data or None on error
        """
        soup = self.fetch_page(url)
        if not soup:
            return None
        
        # Extract title
        title_elem = soup.select_one(TITLE_SELECTOR)
        title = title_elem.get_text(strip=True) if title_elem else ""
        
        # Extract date
        date_elem = soup.select_one(DATE_SELECTOR)
        date_str = ""
        if date_elem:
            # Try datetime attribute first, then text content
            date_str = date_elem.get('datetime', '') or date_elem.get_text(strip=True)
            date_str = self.parse_date(date_str)
        
        # Extract content
        content_elem = soup.select_one(CONTENT_SELECTOR)
        content = ""
        if content_elem:
            # Get text with paragraph separation
            paragraphs = content_elem.find_all('p')
            if paragraphs:
                content = '\n\n'.join(p.get_text(strip=True) for p in paragraphs)
            else:
                content = content_elem.get_text(strip=True)
        
        return {
            'link': url,
            'title': title,
            'date': date_str,
            'content': content
        }
    
    def scrape_category(self, category: str) -> List[dict]:
        """Scrape all articles from a category.
        
        :param category: str: Category/neighborhood name
        :returns: List[dict]: List of article data
        """
        print(f"\nScraping category: {category}")
        
        # Build category URL (customize based on website structure)
        category_url = f"{BASE_URL}/news/{category}"
        
        # Get article links
        links = self.get_article_links(category_url)
        print(f"  Found {len(links)} articles")
        
        # Scrape each article
        articles = []
        for i, link in enumerate(links, 1):
            print(f"  [{i}/{len(links)}] {link[:60]}...")
            article = self.scrape_article(link)
            if article:
                article['quartiere'] = category  # Add category info
                articles.append(article)
        
        return articles
    
    def save_articles(self, articles: List[dict], category: str):
        """Save articles to JSON file.
        
        :param articles: List[dict]: Articles to save
        :param category: str: Category name for filename
        """
        output_file = self.output_dir / f"notizie_{category}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(articles)} articles to {output_file}")
    
    def run(self):
        """Run the scraper for all categories."""
        print(f"Starting {SOURCE_NAME} scraper...")
        print(f"Output directory: {self.output_dir}")
        
        for category in CATEGORIES:
            articles = self.scrape_category(category)
            if articles:
                self.save_articles(articles, category)
        
        print("\nScraping complete!")


# MAIN
if __name__ == "__main__":
    scraper = NewsScraper()
    scraper.run()
