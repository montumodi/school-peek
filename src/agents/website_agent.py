"""
Website Search Agent - Agent 1
Performs nested deep search on https://adalovelace.org.uk/
"""
import sys
import os
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import requests
import bs4
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


class WebsiteSearchAgent:
    """Agent for deep searching the Ada Lovelace school website"""
    
    def __init__(self, gemini_api_key: str, base_url: str = "https://adalovelace.org.uk"):
        self.base_url = base_url
        self.visited_urls = set()
        self.max_depth = 3
        self.max_pages = 20
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.3
        )
        
        # Create tools
        self.tools = [
            Tool(
                name="search_page",
                func=self._search_page,
                description="Search a specific page on the Ada Lovelace school website. Input should be a relative URL path or empty string for homepage."
            ),
            Tool(
                name="extract_links",
                func=self._extract_links,
                description="Extract all links from a given URL on the Ada Lovelace school website. Input should be a URL."
            ),
            Tool(
                name="search_nested",
                func=self._search_nested,
                description="Perform a deep nested search starting from a URL, following links up to a certain depth. Input should be a URL and search query separated by '|'."
            )
        ]
        
        # Create agent
        self.agent = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent"""
        template = """You are a helpful assistant that searches the Ada Lovelace school website for information.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=10)
    
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
    
    def _search_nested(self, input_str: str) -> str:
        """Perform nested search starting from a URL"""
        parts = input_str.split('|')
        if len(parts) != 2:
            return "Input format should be: URL|search_query"
        
        start_url, query = parts[0].strip(), parts[1].strip()
        
        results = []
        to_visit = [(start_url, 0)]  # (url, depth)
        
        while to_visit and len(results) < 5:
            url, depth = to_visit.pop(0)
            
            if depth > self.max_depth or url in self.visited_urls:
                continue
            
            content = self._fetch_page_content(url)
            
            # Check if content is relevant to query
            if query.lower() in content.lower():
                results.append(f"Found in {url}:\n{content[:500]}")
            
            # Extract and queue child links if not at max depth
            if depth < self.max_depth:
                try:
                    response = requests.get(url, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for tag in soup.find_all('a', href=True):
                        link = urljoin(url, tag['href'])
                        if link.startswith(self.base_url) and link not in self.visited_urls:
                            to_visit.append((link, depth + 1))
                except:
                    pass
        
        if results:
            return "Nested search results:\n" + "\n\n".join(results)
        else:
            return f"No results found for '{query}' in nested search"
    
    def search(self, query: str) -> str:
        """Main search method"""
        self.visited_urls.clear()
        try:
            result = self.agent.invoke({"input": query})
            return result.get("output", "No results found")
        except Exception as e:
            return f"Error during search: {str(e)}"
