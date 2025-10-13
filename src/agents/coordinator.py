"""
Agent Coordinator - Manages multiple agents
Coordinates between Website Search Agent and Gmail Search Agent
Uses Google's native function calling (ADK)
"""
import sys
import os
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

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
        """Create function declarations for coordinator tools"""
        search_website_func = FunctionDeclaration(
            name="search_website",
            description="Search the Ada Lovelace school website for information. Use this for general school information, policies, timetables, events, facilities, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for the website"
                    }
                },
                "required": ["query"]
            }
        )
        
        search_gmail_func = FunctionDeclaration(
            name="search_gmail",
            description="Search user's Gmail for school-related emails. Use this for recent communications, announcements, parent letters, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for Gmail"
                    }
                },
                "required": ["query"]
            }
        )
        
        search_both_func = FunctionDeclaration(
            name="search_both",
            description="Search both the website and Gmail for comprehensive information. Use when you need to check multiple sources.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for both sources"
                    }
                },
                "required": ["query"]
            }
        )
        
        return Tool(function_declarations=[search_website_func, search_gmail_func, search_both_func])
    
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
    
    def _execute_function(self, function_name: str, args: Dict[str, Any]) -> str:
        """Execute a function by name with given arguments"""
        if function_name == "search_website":
            return self._search_website(args["query"])
        elif function_name == "search_gmail":
            return self._search_gmail(args["query"])
        elif function_name == "search_both":
            return self._search_both(args["query"])
        else:
            return f"Unknown function: {function_name}"
    
    def answer_query(self, query: str, max_iterations: int = 5) -> str:
        """
        Main method to answer user queries using appropriate agents
        
        Args:
            query: User's question
            
        Returns:
            Answer based on agent coordination
        """
        # Create new chat session for this query
        self.chat = self.model.start_chat()
        
        try:
            prompt = f"""You are a helpful school information coordinator for Ada Lovelace School.

Your task is to answer this question: {query}

You have access to these functions:
- search_website: Search the school website for general information (policies, timetables, facilities, etc.)
- search_gmail: Search Gmail for recent communications (announcements, parent letters, etc.)
- search_both: Search both sources for comprehensive information

Guidelines:
1. Use search_website for general school information and static content
2. Use search_gmail for recent communications and time-sensitive information
3. Use search_both when you need comprehensive information from multiple sources

Provide a clear, well-formatted answer based on the information you find."""

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
                
                print(f"Coordinator calling: {function_name} with args: {function_args}")
                
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
            
            return "I couldn't find a suitable answer to your question."
            
        except Exception as e:
            # Fallback: try website search directly
            try:
                return self.website_agent.search(query)
            except:
                return f"I encountered an error processing your question: {str(e)}"
