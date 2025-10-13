"""
Gmail Search Agent - Agent 2
Searches user's Gmail account for emails from school
"""
import sys
import os
from typing import List, Dict, Any, Optional
import base64
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

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
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.3
        )
        
        # Create tools
        self.tools = [
            Tool(
                name="search_emails",
                func=self._search_emails,
                description="Search Gmail for emails from school domain. Input should be a search query."
            ),
            Tool(
                name="get_recent_emails",
                func=self._get_recent_emails,
                description="Get recent emails from school domain. Input should be number of days to look back (e.g., '7' for last 7 days)."
            ),
            Tool(
                name="search_by_subject",
                func=self._search_by_subject,
                description="Search emails by subject line. Input should be the subject keywords to search for."
            )
        ]
        
        # Create agent
        self.agent = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent"""
        template = """You are a helpful assistant that searches Gmail for school-related emails.

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
    
    def search(self, query: str) -> str:
        """Main search method"""
        try:
            result = self.agent.invoke({"input": query})
            return result.get("output", "No results found")
        except Exception as e:
            return f"Error during Gmail search: {str(e)}"
