"""
Website Search Agent - Agent 1
Performs nested deep search on https://adalovelace.org.uk/
Uses Google's native function calling (ADK)
"""
import sys
import os
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import requests
import bs4
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool


class WebsiteSearchAgent:
    """Agent for deep searching the Ada Lovelace school website"""
    
    def __init__(self, gemini_api_key: str, base_url: str = "https://adalovelace.org.uk"):
        self.base_url = base_url
        self.visited_urls = set()
        self.max_depth = 3
        self.max_pages = 20
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        
        # Initialize model with function calling
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            tools=[self._create_tools()]
        )
        
        # Create chat for maintaining context
        self.chat = None
    
    def _create_tools(self) -> Tool:
        """Create function declarations for tools"""
        search_page_func = FunctionDeclaration(
            name="search_page",
            description="Search a specific page on the Ada Lovelace school website. Input should be a relative URL path or empty string for homepage.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The relative URL path to search, or empty string for homepage"
                    }
                },
                "required": ["path"]
            }
        )
        
        extract_links_func = FunctionDeclaration(
            name="extract_links",
            description="Extract all links from a given URL on the Ada Lovelace school website.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to extract links from"
                    }
                },
                "required": ["url"]
            }
        )
        
        search_nested_func = FunctionDeclaration(
            name="search_nested",
            description="Perform a deep nested search starting from a URL, following links up to a certain depth.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The starting URL for nested search"
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query to look for"
                    }
                },
                "required": ["url", "query"]
            }
        )
        
        return Tool(function_declarations=[search_page_func, extract_links_func, search_nested_func])
    
    
    def _fetch_page_content(self, url: str) -> str:
        """Fetch and parse page content"""
        try:
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
            return f"Error fetching page: {str(e)}"
    
    def _search_page(self, path: str = "") -> str:
        """Search a specific page on the website"""
        if path and not path.startswith('http'):
            url = urljoin(self.base_url, path)
        elif path:
            url = path
        else:
            url = self.base_url
        
        if not url.startswith(self.base_url):
            return "URL is outside the Ada Lovelace school domain"
        
        if url in self.visited_urls:
            return f"Already visited {url}"
        
        self.visited_urls.add(url)
        content = self._fetch_page_content(url)
        return f"Content from {url}:\n{content}"
    
    def _extract_links(self, url: str) -> str:
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
            
            return "Links found:\n" + "\n".join(links[:10])  # Limit to 10 links
            
        except Exception as e:
            return f"Error extracting links: {str(e)}"
    
    def _search_nested(self, url: str, query: str) -> str:
        """Perform nested search starting from a URL"""
        results = []
        to_visit = [(url, 0)]  # (url, depth)
        
        while to_visit and len(results) < 5:
            current_url, depth = to_visit.pop(0)
            
            if depth > self.max_depth or current_url in self.visited_urls:
                continue
            
            content = self._fetch_page_content(current_url)
            
            # Check if content is relevant to query
            if query.lower() in content.lower():
                results.append(f"Found in {current_url}:\n{content[:500]}")
            
            # Extract and queue child links if not at max depth
            if depth < self.max_depth:
                try:
                    response = requests.get(current_url, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for tag in soup.find_all('a', href=True):
                        link = urljoin(current_url, tag['href'])
                        if link.startswith(self.base_url) and link not in self.visited_urls:
                            to_visit.append((link, depth + 1))
                except:
                    pass
        
        if results:
            return "Nested search results:\n" + "\n\n".join(results)
        else:
            return f"No results found for '{query}' in nested search"
    
    def _execute_function(self, function_name: str, args: Dict[str, Any]) -> str:
        """Execute a function by name with given arguments"""
        if function_name == "search_page":
            return self._search_page(args.get("path", ""))
        elif function_name == "extract_links":
            return self._extract_links(args["url"])
        elif function_name == "search_nested":
            return self._search_nested(args["url"], args["query"])
        else:
            return f"Unknown function: {function_name}"
    
    def search(self, query: str, max_iterations: int = 10) -> str:
        """Main search method using Google's function calling"""
        self.visited_urls.clear()
        
        # Create new chat session for this search
        self.chat = self.model.start_chat()
        
        try:
            prompt = f"""You are a helpful assistant that searches the Ada Lovelace school website for information.

Your task is to answer this question: {query}

You have access to these functions to search the website:
- search_page: Search a specific page
- extract_links: Get all links from a page  
- search_nested: Do a deep nested search with a query

Use the functions strategically to find the information. Start with the homepage and navigate as needed.
Provide a clear answer based on what you find."""

            response = self.chat.send_message(prompt)
            
            # Handle function calling loop
            iteration = 0
            while iteration < max_iterations:
                # Check if model wants to call a function
                if not response.candidates[0].content.parts:
                    break
                    
                function_call = None
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        break
                
                if not function_call:
                    # No more function calls, we have the final answer
                    break
                
                # Execute the function
                function_name = function_call.name
                function_args = dict(function_call.args)
                
                print(f"Calling function: {function_name} with args: {function_args}")
                
                function_result = self._execute_function(function_name, function_args)
                
                # Send function result back to model
                response = self.chat.send_message(
                    genai.types.content_types.to_content({
                        "parts": [{
                            "function_response": {
                                "name": function_name,
                                "response": {"result": function_result}
                            }
                        }]
                    })
                )
                
                iteration += 1
            
            # Get final text response
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        return part.text
            
            return "No results found"
            
        except Exception as e:
            return f"Error during search: {str(e)}"
