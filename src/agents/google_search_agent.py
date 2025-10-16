"""
Google Web Search Agent

This agent specializes in searching through websites using Google's search tool.
It's designed to find information from the school website and other public sources.
"""

import os
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"

# Ensure API key is set
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Create the Google Web Search Agent
google_search_agent = LlmAgent(
    name="GoogleWebSearchAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI research assistant specializing in web searches for school-related information.
    
    Your task:
    1. Use the Google Search tool to search for information related to Ada Lovelace School.
    2. Focus on finding current, relevant information from the school website and official sources.
    3. Summarize your findings concisely and clearly.
    4. Include specific details like dates, locations, contact information when available.
    5. If you find multiple relevant pieces of information, organize them logically.
    6. Only search from site:
    
    Output format:
    - Provide a clear, structured summary of what you found
    - Include source references when possible
    - If no relevant information is found, clearly state that
    - Focus on accuracy and relevance to the user's question
    
    Output *only* the summary of findings from web searches.""",
    tools=[google_search],
    description="Specialized agent for searching school information through web search.",
    output_key="web_search_result"
)
