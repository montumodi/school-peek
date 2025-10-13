"""
Agent Coordinator - Manages multiple agents
Coordinates between Website Search Agent and Gmail Search Agent
"""
import sys
import os
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from .website_agent import WebsiteSearchAgent
from .gmail_agent import GmailSearchAgent


class AgentCoordinator:
    """Coordinates multiple agents to answer user queries"""
    
    def __init__(self, gemini_api_key: str, gmail_credentials_path: str = None):
        """
        Initialize the coordinator with access to both agents
        
        Args:
            gemini_api_key: API key for Gemini LLM
            gmail_credentials_path: Path to Gmail OAuth credentials (optional)
        """
        self.gemini_api_key = gemini_api_key
        
        # Initialize specialized agents
        self.website_agent = WebsiteSearchAgent(gemini_api_key)
        self.gmail_agent = GmailSearchAgent(gemini_api_key, credentials_path=gmail_credentials_path)
        
        # Initialize coordinator LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.3
        )
        
        # Create coordinator tools
        self.tools = [
            Tool(
                name="search_website",
                func=self._search_website,
                description="Search the Ada Lovelace school website for information. Use this for general school information, policies, timetables, events, etc. Input should be the search query."
            ),
            Tool(
                name="search_gmail",
                func=self._search_gmail,
                description="Search user's Gmail for school-related emails. Use this for recent communications, announcements sent via email, parent letters, etc. Input should be the search query."
            ),
            Tool(
                name="search_both",
                func=self._search_both,
                description="Search both the website and Gmail for comprehensive information. Use this when you need to check multiple sources. Input should be the search query."
            )
        ]
        
        # Create coordinator agent
        self.agent = self._create_coordinator_agent()
    
    def _create_coordinator_agent(self) -> AgentExecutor:
        """Create the coordinator agent that decides which sub-agents to use"""
        template = """You are a helpful school information coordinator that helps answer questions about Ada Lovelace School.

You have access to the following information sources:
{tools}

Guidelines for choosing tools:
1. Use "search_website" for:
   - General school information (policies, curriculum, staff)
   - School calendar, events, timetables
   - Facilities, departments, contact information
   - Static school information

2. Use "search_gmail" for:
   - Recent announcements and communications
   - Parent letters and circulars
   - Time-sensitive information
   - Personal or class-specific messages

3. Use "search_both" when:
   - You need comprehensive information from multiple sources
   - The question could be answered by either source
   - You want to provide the most complete answer

Use the following format:

Question: the input question you must answer
Thought: think about which information source(s) would best answer this question
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: a comprehensive, well-formatted answer based on the information gathered

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def _search_website(self, query: str) -> str:
        """Search the school website"""
        try:
            return self.website_agent.search(query)
        except Exception as e:
            return f"Website search error: {str(e)}"
    
    def _search_gmail(self, query: str) -> str:
        """Search Gmail for school emails"""
        try:
            return self.gmail_agent.search(query)
        except Exception as e:
            return f"Gmail search error: {str(e)}"
    
    def _search_both(self, query: str) -> str:
        """Search both website and Gmail"""
        results = []
        
        # Search website
        try:
            website_results = self.website_agent.search(query)
            results.append(f"=== Website Results ===\n{website_results}")
        except Exception as e:
            results.append(f"Website search error: {str(e)}")
        
        # Search Gmail
        try:
            gmail_results = self.gmail_agent.search(query)
            results.append(f"\n\n=== Gmail Results ===\n{gmail_results}")
        except Exception as e:
            results.append(f"\nGmail search error: {str(e)}")
        
        return "\n".join(results)
    
    def answer_query(self, query: str) -> str:
        """
        Main method to answer user queries using appropriate agents
        
        Args:
            query: User's question
            
        Returns:
            Answer based on agent coordination
        """
        try:
            result = self.agent.invoke({"input": query})
            return result.get("output", "I couldn't find a suitable answer to your question.")
        except Exception as e:
            # Fallback: try website search directly
            try:
                return self.website_agent.search(query)
            except:
                return f"I encountered an error processing your question: {str(e)}"
