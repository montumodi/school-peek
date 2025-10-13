"""
Agent Coordinator - Manages multiple agents
Coordinates between Website Search Agent and Gmail Search Agent
Uses Google ADK (Agent Development Kit) with hierarchical multi-agent pattern
"""
import sys
import os
from typing import Optional
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .website_agent import WebsiteSearchAgent
from .gmail_agent import GmailSearchAgent


class AgentCoordinator:
    """Coordinates multiple agents to answer user queries using Google ADK"""
    
    def __init__(self, gemini_api_key: str = None, gmail_credentials_path: str = None):
        """
        Initialize the coordinator with access to both agents
        
        Args:
            gemini_api_key: API key for Gemini LLM (uses GOOGLE_API_KEY env var if not provided)
            gmail_credentials_path: Path to Gmail OAuth credentials (optional)
        """
        # Set API key as environment variable for Google ADK
        if gemini_api_key:
            os.environ['GOOGLE_API_KEY'] = gemini_api_key
        
        # Initialize specialized agents
        self.website_agent = WebsiteSearchAgent()
        self.gmail_agent = GmailSearchAgent(credentials_path=gmail_credentials_path)
        
        # Create coordinator agent with sub-agents (hierarchical pattern)
        self.coordinator = LlmAgent(
            name="SchoolInfoCoordinator",
            model="gemini-2.0-flash-exp",
            description="Coordinator that helps answer questions about Ada Lovelace School by delegating to specialized agents.",
            instruction="""You are a helpful school information coordinator for Ada Lovelace School.

When users ask questions:
1. For questions about school policies, timetables, events, facilities, or general school information 
   -> delegate to WebsiteSearchAgent
2. For questions about recent emails, announcements, parent letters, or recent communications 
   -> delegate to GmailSearchAgent
3. For comprehensive questions that might need both sources 
   -> you can delegate to both agents

Be helpful, clear, and concise in your responses. If the agents don't find information, 
suggest contacting the school directly.""",
            sub_agents=[self.website_agent, self.gmail_agent]
        )
        
        # Session configuration
        self.app_name = "school_peek"
        self.user_id = "user_001"
        self.session_counter = 0
    
    def answer_query(self, query: str) -> str:
        """
        Main method to answer user queries using appropriate agents
        
        Args:
            query: User's question
            
        Returns:
            Answer based on agent coordination
        """
        try:
            # Use asyncio to run the async method
            return asyncio.run(self._answer_query_async(query))
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    async def _answer_query_async(self, query: str) -> str:
        """Async implementation of answer_query"""
        try:
            # Create session
            self.session_counter += 1
            session_id = f"session_{self.session_counter}"
            
            session_service = InMemorySessionService()
            await session_service.create_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=session_id
            )
            
            # Create runner with coordinator agent
            runner = Runner(
                agent=self.coordinator,
                app_name=self.app_name,
                session_service=session_service
            )
            
            # Prepare user message
            content = types.Content(
                role='user',
                parts=[types.Part(text=query)]
            )
            
            # Run the agent and collect response
            final_response = ""
            events = runner.run(
                user_id=self.user_id,
                session_id=session_id,
                new_message=content
            )
            
            for event in events:
                if event.is_final_response() and event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            final_response = part.text
                            break
                    if final_response:
                        break
            
            return final_response if final_response else "No response generated."
            
        except Exception as e:
            # Fallback: try website search directly if coordinator fails
            try:
                print(f"Coordinator error: {e}, trying website agent directly")
                result = ""
                async for event in self.website_agent._run_async_impl(
                    type('Context', (), {'new_message': types.Content(role='user', parts=[types.Part(text=query)])})()
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                result = part.text
                                break
                return result if result else f"Error: {str(e)}"
            except:
                return f"Error processing query: {str(e)}"
