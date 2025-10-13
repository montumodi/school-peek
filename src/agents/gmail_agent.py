"""
Gmail Search Agent - Agent 2
Searches user's Gmail account for emails from school
Uses Google ADK (Agent Development Kit) with built-in patterns
"""
import sys
import os
from typing import AsyncGenerator, Optional, ClassVar
import base64
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import pickle
except ImportError:
    print("Gmail API libraries not installed. Gmail agent will not function.")


class GmailSearchAgent(BaseAgent):
    """Agent for searching Gmail for school-related emails using Google ADK"""
    
    name: str = "GmailSearchAgent"
    description: str = "Searches Gmail for school-related emails, announcements, parent letters, and recent communications from the school."
    
    # Gmail API scopes - class variable
    SCOPES: ClassVar[list] = ['https://www.googleapis.com/auth/gmail.readonly']
    
    # Instance variables
    school_domain: str = "adalovelace.org.uk"
    credentials_path: Optional[str] = None
    token_path: str = "token.pickle"
    gmail_service: Optional[object] = None
    
    def __init__(self, credentials_path: Optional[str] = None, 
                 token_path: Optional[str] = None, 
                 school_domain: str = "adalovelace.org.uk", **kwargs):
        super().__init__(**kwargs)
        self.school_domain = school_domain
        self.credentials_path = credentials_path or os.getenv('GMAIL_CREDENTIALS_PATH')
        self.token_path = token_path or 'token.pickle'
        self.gmail_service = None
    
    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """Execute Gmail search based on user query"""
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
        
        # Authenticate and search
        if not self.gmail_service:
            if not self._authenticate():
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="Gmail authentication required. Please configure Gmail credentials.")]
                    )
                )
                return
        
        # Perform the search
        result = await self._search_gmail(query)
        
        # Yield the result
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=result)]
            )
        )
    
    def _authenticate(self) -> bool:
        """Authenticate with Gmail API"""
        try:
            creds = None
            
            # Load existing token
            if os.path.exists(self.token_path):
                with open(self.token_path, 'rb') as token:
                    creds = pickle.load(token)
            
            # If credentials are invalid or don't exist, authenticate
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif self.credentials_path and os.path.exists(self.credentials_path):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, self.SCOPES)
                    creds = flow.run_local_server(port=0)
                else:
                    return False
                
                # Save credentials for future use
                with open(self.token_path, 'wb') as token:
                    pickle.dump(creds, token)
            
            self.gmail_service = build('gmail', 'v1', credentials=creds)
            return True
            
        except Exception as e:
            print(f"Gmail authentication error: {str(e)}")
            return False
    
    async def _search_gmail(self, query: str) -> str:
        """Search Gmail for emails"""
        try:
            # Construct query to search only school domain
            gmail_query = f"from:@{self.school_domain} {query}"
            
            results = self.gmail_service.users().messages().list(
                userId='me',
                q=gmail_query,
                maxResults=5
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return f"No emails found from {self.school_domain} matching: {query}"
            
            email_summaries = []
            for msg in messages:
                msg_detail = self.gmail_service.users().messages().get(
                    userId='me', id=msg['id'], format='full'
                ).execute()
                
                headers = msg_detail['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                from_email = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
                
                body = self._get_message_body(msg_detail)
                
                email_summaries.append(f"Subject: {subject}\nFrom: {from_email}\nDate: {date}\nContent: {body[:500]}...")
            
            return "Found emails:\n\n" + "\n\n---\n\n".join(email_summaries)
            
        except Exception as e:
            return f"Error searching emails: {str(e)}"
    
    def _get_message_body(self, message: dict) -> str:
        """Extract message body from Gmail message"""
        try:
            if 'payload' in message:
                payload = message['payload']
                
                if 'parts' in payload:
                    # Multi-part message
                    for part in payload['parts']:
                        if part['mimeType'] == 'text/plain':
                            data = part['body'].get('data', '')
                            if data:
                                return base64.urlsafe_b64decode(data).decode('utf-8')
                elif 'body' in payload:
                    # Simple message
                    data = payload['body'].get('data', '')
                    if data:
                        return base64.urlsafe_b64decode(data).decode('utf-8')
            
            return "No content"
        except Exception as e:
            return f"Error extracting content: {str(e)}"
