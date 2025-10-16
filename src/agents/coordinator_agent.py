"""
Coordinator Agent

This is the main coordinator agent that orchestrates the multi-agent system.
It uses a Parallelization + Sequential pattern:
1. Runs Google Search and Gmail Search agents in parallel
2. Synthesizes the results from both agents into a comprehensive response

Based on the Multi-Agent Collaboration pattern from:
https://github.com/montumodi/agentic_design_patterns/tree/main/07.%20multi_agent_collaboration
"""

import uuid
import asyncio
import os
from typing import Optional, AsyncGenerator
from google.adk.agents import ParallelAgent, SequentialAgent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv

# Import specialized agents
from .google_search_agent import google_search_agent
from .gmail_agent import gmail_agent

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"

# Ensure API key is set
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable is not set")

APP_NAME="Google Search_agent"
USER_ID="user1234"
SESSION_ID="1234"

# Define the Parallel Agent to run both search agents concurrently
parallel_search_agent = ParallelAgent(
    name="ParallelSearchAgent",
    sub_agents=[google_search_agent, gmail_agent],
    description="Runs web and Gmail search agents in parallel to gather information."
)

# Define the Synthesis/Merger Agent to combine results
synthesis_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI assistant responsible for synthesizing information from multiple sources 
    into a comprehensive, user-friendly response for Ada Lovelace School inquiries.
    
    You will receive:
    - Web search results (from GoogleWebSearchAgent)
    - Gmail search results (from PerplexityGmailAgent)
    
    Your task:
    1. Combine information from BOTH sources intelligently
    2. Prioritize Gmail/email results as they are typically more current and official
    3. Use web search results to supplement and provide additional context
    4. Create a coherent, well-structured response that answers the user's question
    5. Clearly attribute information sources (email vs. web)
    6. Identify and reconcile any conflicting information
    
    **Output Format:**
    Provide a clear, comprehensive answer following this structure:
    
    ### [Direct Answer to the Question]
    [Provide the main answer upfront]
    
    ### Key Details
    - [Detail 1 with source attribution]
    - [Detail 2 with source attribution]
    - [Detail 3 with source attribution]
    
    ### Sources
    - Email Communications: [Summarize email findings if available]
    - School Website: [Summarize web findings if available]
    
    ### Additional Information
    [Any supplementary context or recommendations]
    
    **Important Guidelines:**
    - If email and web sources conflict, favor email sources and note the discrepancy
    - If information is missing from both sources, clearly state this
    - Maintain a helpful, professional tone appropriate for school communications
    - Be concise but thorough
    - Use bullet points for clarity
    
    Output *only* the synthesized response in the format above.""",
    description="Synthesizes results from web and email searches into comprehensive answers."
)

# Create the main Sequential Agent (Parallel search → Synthesis)
coordinator_agent = SequentialAgent(
    name="CoordinatorAgent",
    sub_agents=[parallel_search_agent, synthesis_agent],
    description="Main coordinator that orchestrates parallel searches and synthesizes results."
)


async def call_agent(query, access_token=None, refresh_token=None):
    """
    Call the coordinator agent with a query using async API.
    
    Args:
        query: User's question
        access_token: OAuth access token for Gmail access (optional)
        refresh_token: OAuth refresh token (optional)
    """
    # If OAuth tokens provided, set user credentials for Gmail agent
    if access_token:
        from .gmail_agent import set_user_credentials
        set_user_credentials(access_token, refresh_token)
    
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=coordinator_agent, app_name=APP_NAME, session_service=session_service)

    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    # Use run_async instead of run to avoid nested event loops
    full_response = ""
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response = part.text
                            author = getattr(event, 'author', 'Unknown')
                            full_response += response + "\n"
                            # Suppress verbose logging
                            # print(f"{author}: {response}")
                            # print("---")
    except Exception as e:
        print(f"Error in agent execution: {e}")
        raise
    
    return full_response
