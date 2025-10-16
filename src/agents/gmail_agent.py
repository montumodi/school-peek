import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import base64
from datetime import datetime, timedelta
from config.config import GEMINI_API_KEY, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

GEMINI_MODEL = "gemini-2.0-flash"

# Allowed email domains for school communications
SCHOOL_EMAIL_DOMAINS = ["adalovelace.org.uk", "schoolcomms.com"]

# Global variable to store current user's credentials
_current_user_credentials: Optional[Credentials] = None

def set_user_credentials(access_token: str, refresh_token: Optional[str] = None) -> None:
    """Set the current user's Gmail credentials"""
    global _current_user_credentials
    _current_user_credentials = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/gmail.readonly"]
    )

def search_gmail_messages(query: str, max_results: int) -> str:
    """
    Search Gmail messages using a query string.
    Only searches emails from allowed school domains: adalovelace.org.uk, schoolcomms.com
    
    Args:
        query: Gmail search query (e.g., 'subject:homework', 'from:teacher@school.edu', 'is:unread')
        max_results: Maximum number of messages to return
    
    Returns:
        A formatted string with email subjects, senders, dates, and snippets
    """
    if not _current_user_credentials:
        return "Error: User credentials not set. Please log in first."
    
    if not max_results:
        max_results = 10
    
    # Add domain filters to the query
    domain_filter = " OR ".join([f"from:*@{domain}" for domain in SCHOOL_EMAIL_DOMAINS])
    filtered_query = f"({domain_filter}) {query}"
    
    try:
        service = build('gmail', 'v1', credentials=_current_user_credentials)
        
        # Search for messages with domain filter
        results = service.users().messages().list(
            userId='me',
            q=filtered_query,
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        
        if not messages:
            return f"No emails found from school domains ({', '.join(SCHOOL_EMAIL_DOMAINS)}) matching query: {query}"
        
        email_summaries = []
        for msg in messages:
            msg_data = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='full'
            ).execute()
            
            headers = msg_data['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            from_addr = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
            snippet = msg_data.get('snippet', '')
            
            email_summaries.append(
                f"Subject: {subject}\n"
                f"From: {from_addr}\n"
                f"Date: {date}\n"
                f"Preview: {snippet}\n"
            )
        
        return "\n---\n".join(email_summaries)
        
    except Exception as e:
        return f"Error searching Gmail: {str(e)}"

def get_recent_emails(days: int) -> str:
    """
    Get recent emails from the last N days from allowed school domains only.
    
    Args:
        days: Number of days to look back
    
    Returns:
        A formatted string with recent email information
    """
    if not days:
        days = 7
    date_filter = (datetime.now() - timedelta(days=days)).strftime('%Y/%m/%d')
    query = f"after:{date_filter}"
    return search_gmail_messages(query, max_results=20)

def get_unread_emails() -> str:
    """
    Get all unread emails from allowed school domains only.
    
    Returns:
        A formatted string with unread email information
    """
    return search_gmail_messages("is:unread", max_results=20)

# Create ADK tools from functions
search_gmail_tool = FunctionTool(func=search_gmail_messages)
recent_emails_tool = FunctionTool(func=get_recent_emails)
unread_emails_tool = FunctionTool(func=get_unread_emails)

# Create Gmail agent using ADK
gmail_agent = LlmAgent(
    name="GmailAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Gmail search specialist for Ada Lovelace School communications.
    
    IMPORTANT: You can ONLY search emails from these official school domains:
    - adalovelace.org.uk
    - schoolcomms.com
    
    All search results are automatically filtered to these domains only.
    
    Your role: Search and analyze official school emails to find relevant information.
    
    Available tools:
    - search_gmail_messages: Search emails with specific queries (use Gmail search syntax)
    - get_recent_emails: Get emails from last N days (from school domains only)
    - get_unread_emails: Get all unread emails (from school domains only)
    
    Gmail search syntax examples:
    - "subject:homework" - emails about homework
    - "subject:parent evening" - emails about parent evenings
    - "after:2024/10/01" - emails after a date
    - "is:unread" - unread emails
    - "has:attachment" - emails with attachments
    
    Note: You don't need to add "from:" filters as emails are already filtered to school domains.
    
    When analyzing emails:
    1. Extract key information: subjects, dates, senders
    2. Summarize important announcements or deadlines
    3. Identify patterns or recurring topics
    4. Present findings clearly with email details
    
    Output format:
    - List relevant emails with subject, sender, date
    - Provide a brief summary of each email's content
    - Highlight any urgent or time-sensitive information
    
    If no relevant emails found from school domains, clearly state this.""",
    description="Searches Gmail for official school communications from adalovelace.org.uk and schoolcomms.com",
    tools=[search_gmail_tool, recent_emails_tool, unread_emails_tool]
)
