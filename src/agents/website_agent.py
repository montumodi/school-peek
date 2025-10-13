"""
Website Search Agent - Agent 1
Performs nested deep search on https://adalovelace.org.uk/
Uses Google ADK (Agent Development Kit) with built-in patterns
"""
import sys
import os
from typing import AsyncGenerator
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types


class WebsiteSearchAgent(BaseAgent):
    """Agent for deep searching the Ada Lovelace school website using Google ADK"""
    
    name: str = "WebsiteSearchAgent"
    description: str = "Searches the Ada Lovelace school website for information about policies, timetables, events, facilities, and general school information."
    
    # Instance variables (need type annotations for Pydantic)
    base_url: str = "https://adalovelace.org.uk"
    visited_urls: set = None
    max_depth: int = 3
    max_pages: int = 20
    
    def __init__(self, base_url: str = "https://adalovelace.org.uk", **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.visited_urls = set()
        self.max_depth = 3
        self.max_pages = 20
    
    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """Execute website search based on user query"""
        # Get the user's query from context
        query = ""
        if context.new_message and context.new_message.parts:
            for part in context.new_message.parts:
                if part.text:
                    query = part.text
                    break
        
        if not query:
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="No query provided.")]
                )
            )
            return
        
        # Clear visited URLs for new search
        self.visited_urls.clear()
        
        # Perform the search
        result = await self._search_website(query)
        
        # Yield the result
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=result)]
            )
        )
    
    async def _search_website(self, query: str) -> str:
        """Main search logic"""
        try:
            # Start with homepage
            homepage_content = self._fetch_page_content(self.base_url)
            
            # Check if query terms are in homepage
            if query.lower() in homepage_content.lower():
                return f"Found information on homepage:\n\n{homepage_content[:1000]}"
            
            # Extract links and search deeper
            links = self._extract_links(self.base_url)
            results = []
            
            for link in links[:10]:  # Limit to 10 links
                if link in self.visited_urls:
                    continue
                    
                content = self._fetch_page_content(link)
                if query.lower() in content.lower():
                    results.append(f"Found in {link}:\n{content[:500]}")
                
                if len(results) >= 3:  # Limit results
                    break
            
            if results:
                return "Search results:\n\n" + "\n\n---\n\n".join(results)
            else:
                return f"No specific information found for '{query}' on the school website. The website contains general school information."
                
        except Exception as e:
            return f"Error searching website: {str(e)}"
    
    def _fetch_page_content(self, url: str) -> str:
        """Fetch and parse page content"""
        try:
            if url in self.visited_urls:
                return ""
            
            self.visited_urls.add(url)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            content_div = soup.find('div', class_='page-content')
            if content_div:
                text = content_div.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text
            text = ' '.join(text.split())
            return text[:5000]  # Limit content length
            
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"
    
    def _extract_links(self, url: str) -> list:
        """Extract all links from a page"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for tag in soup.find_all('a', href=True):
                link = urljoin(url, tag['href'])
                if link.startswith(self.base_url) and link not in self.visited_urls:
                    links.append(link)
            
            return links[:20]  # Limit to 20 links
            
        except Exception as e:
            return []
