"""
Gmail Search Agent - Agent 2
Searches user's Gmail account for emails from school
Uses Google's native function calling (ADK)
"""
import sys
import os
from typing import List, Dict, Any, Optional
import base64
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import pickle
except ImportError:
    print("Gmail API libraries not installed. Gmail agent will not function.")


class GmailSearchAgent:
    """Agent for searching Gmail for school-related emails"""
    
    # Gmail API scopes
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    
    def __init__(self, gemini_api_key: str, credentials_path: Optional[str] = None, 
                 token_path: Optional[str] = None, school_domain: str = "adalovelace.org.uk"):
        """
        Initialize Gmail Search Agent
        
        Args:
            gemini_api_key: API key for Gemini LLM
            credentials_path: Path to Gmail OAuth credentials JSON file
            token_path: Path to store/load Gmail access token
            school_domain: School email domain to filter messages
        """
        self.school_domain = school_domain
        self.credentials_path = credentials_path or os.getenv('GMAIL_CREDENTIALS_PATH')
        self.token_path = token_path or 'token.pickle'
        self.gmail_service = None
        
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
        """Create function declarations for Gmail tools"""
        search_emails_func = FunctionDeclaration(
            name="search_emails",
            description="Search Gmail for emails from school domain. Returns matching emails with subject, sender, date and content.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find emails"
                    }
                },
                "required": ["query"]
            }
        )
        
        get_recent_emails_func = FunctionDeclaration(
            name="get_recent_emails",
            description="Get recent emails from school domain within specified number of days.",
            parameters={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "string",
                        "description": "Number of days to look back (e.g., '7' for last 7 days)"
                    }
                },
                "required": ["days"]
            }
        )
        
        search_by_subject_func = FunctionDeclaration(
            name="search_by_subject",
            description="Search emails by subject line keywords.",
            parameters={
                "type": "object",
                "properties": {
                    "subject_keywords": {
                        "type": "string",
                        "description": "Keywords to search for in email subjects"
                    }
                },
                "required": ["subject_keywords"]
            }
        )
        
        return Tool(function_declarations=[search_emails_func, get_recent_emails_func, search_by_subject_func])
    
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
    
    def _get_message_body(self, message: Dict) -> str:
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
    
    def _search_emails(self, query: str) -> str:
        """Search Gmail for emails matching query"""
        if not self.gmail_service:
            if not self._authenticate():
                return "Gmail authentication required. Please configure Gmail credentials."
        
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
    
    def _get_recent_emails(self, days_str: str) -> str:
        """Get recent emails from school domain"""
        if not self.gmail_service:
            if not self._authenticate():
                return "Gmail authentication required. Please configure Gmail credentials."
        
        try:
            days = int(days_str)
            after_date = (datetime.now() - timedelta(days=days)).strftime('%Y/%m/%d')
            
            gmail_query = f"from:@{self.school_domain} after:{after_date}"
            
            results = self.gmail_service.users().messages().list(
                userId='me',
                q=gmail_query,
                maxResults=10
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return f"No emails found from {self.school_domain} in the last {days} days"
            
            email_summaries = []
            for msg in messages[:5]:  # Limit to 5 most recent
                msg_detail = self.gmail_service.users().messages().get(
                    userId='me', id=msg['id'], format='full'
                ).execute()
                
                headers = msg_detail['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                from_email = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown')
                
                body = self._get_message_body(msg_detail)
                
                email_summaries.append(f"Subject: {subject}\nFrom: {from_email}\nDate: {date}\nContent: {body[:300]}...")
            
            return f"Recent emails from last {days} days:\n\n" + "\n\n---\n\n".join(email_summaries)
            
        except Exception as e:
            return f"Error getting recent emails: {str(e)}"
    
    def _search_by_subject(self, subject_keywords: str) -> str:
        """Search emails by subject line"""
        if not self.gmail_service:
            if not self._authenticate():
                return "Gmail authentication required. Please configure Gmail credentials."
        
        try:
            gmail_query = f"from:@{self.school_domain} subject:{subject_keywords}"
            
            results = self.gmail_service.users().messages().list(
                userId='me',
                q=gmail_query,
                maxResults=5
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return f"No emails found with subject containing: {subject_keywords}"
            
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
                
                email_summaries.append(f"Subject: {subject}\nFrom: {from_email}\nDate: {date}\nContent: {body[:400]}...")
            
            return "Emails found:\n\n" + "\n\n---\n\n".join(email_summaries)
            
        except Exception as e:
            return f"Error searching by subject: {str(e)}"
    
    def _execute_function(self, function_name: str, args: Dict[str, Any]) -> str:
        """Execute a function by name with given arguments"""
        if function_name == "search_emails":
            return self._search_emails(args["query"])
        elif function_name == "get_recent_emails":
            return self._get_recent_emails(args["days"])
        elif function_name == "search_by_subject":
            return self._search_by_subject(args["subject_keywords"])
        else:
            return f"Unknown function: {function_name}"
    
    def search(self, query: str, max_iterations: int = 10) -> str:
        """Main search method using Google's function calling"""
        # Create new chat session for this search
        self.chat = self.model.start_chat()
        
        try:
            prompt = f"""You are a helpful assistant that searches Gmail for school-related emails.

Your task is to answer this question: {query}

You have access to these functions to search Gmail:
- search_emails: Search for emails matching keywords
- get_recent_emails: Get emails from last N days
- search_by_subject: Search by email subject

Use the functions to find relevant information. Provide a clear, helpful answer based on the emails you find."""

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
            return f"Error during Gmail search: {str(e)}"
