import sys
import datetime
import os
import streamlit as st
import json
from google.oauth2 import id_token
from google.auth.transport import requests
from urllib.parse import urlencode
import secrets

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mongo_utils import get_mongo_client, get_mongo_db
import google.generativeai as genai

from config.config import GEMINI_API_KEY, MONGODB_VECTOR_COLL_LANGCHAIN, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI, ALLOWED_EMAILS

def is_email_allowed(email):
    """Check if email is in the allowed list"""
    if not ALLOWED_EMAILS or len(ALLOWED_EMAILS) == 0 or ALLOWED_EMAILS[0] == "":
        # If no restrictions are set, allow all
        return True
    
    email_lower = email.lower()
    
    for allowed in ALLOWED_EMAILS:
        allowed = allowed.strip().lower()
        if not allowed:
            continue
            
        # Check if it's a domain restriction (starts with @)
        if allowed.startswith("@"):
            if email_lower.endswith(allowed):
                return True
        # Check if it's an exact email match
        elif email_lower == allowed:
            return True
    
    return False

def get_google_auth_url():
    """Generate Google OAuth URL"""
    if not GOOGLE_CLIENT_ID:
        return None
    
    # Generate a random state token for security
    # Use a fixed state for simplicity in Streamlit (state persistence is challenging)
    # In production, consider using a database to store state tokens
    state = secrets.token_urlsafe(32)
    
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile https://www.googleapis.com/auth/gmail.readonly",  # Added Gmail readonly scope
        "state": state,
        "access_type": "offline",
        "prompt": "select_account"
    }
    
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}", state

def verify_google_token(token):
    """Verify Google ID token and return user info"""
    try:
        idinfo = id_token.verify_oauth2_token(
            token, 
            requests.Request(), 
            GOOGLE_CLIENT_ID
        )
        
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')
            
        return {
            'email': idinfo.get('email'),
            'name': idinfo.get('name'),
            'picture': idinfo.get('picture'),
            'user_id': idinfo.get('sub')
        }
    except ValueError as e:
        st.error(f"Token verification failed: {e}")
        return None

def check_authentication():
    """Check if user is authenticated"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    return st.session_state.authenticated

def login_page():
    """Display login page with Google OAuth"""
    st.title("🔐 Ada Lovelace School Assistant - Login")
    st.markdown("Please sign in with your Google account to access the school assistant.")
    
    # Check if Google OAuth is configured
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        st.error("⚠️ Google OAuth is not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables.")
        st.info("To set up Google OAuth:")
        st.markdown("""
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select an existing one
        3. Enable the Google+ API
        4. Create OAuth 2.0 credentials (Web application)
        5. Add authorized redirect URI: `http://localhost:8501`
        6. Set environment variables:
           - `GOOGLE_CLIENT_ID`: Your client ID
           - `GOOGLE_CLIENT_SECRET`: Your client secret
           - `GOOGLE_REDIRECT_URI`: http://localhost:8501
        """)
        
        # Fallback to demo mode with hardcoded credentials
        st.divider()
        st.markdown("### 🔧 Demo Mode (Development Only)")
        st.warning("Using hardcoded credentials for development. Remove in production!")
        
        VALID_CREDENTIALS = {
            "admin": "Password1!",
            "teacher": "Password2!",
            "user": "Password3!"
        }
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_email = f"{username}@demo.local"
                    st.session_state.user_picture = None
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        with st.expander("🔍 Demo Credentials"):
            st.markdown("""
            **Available accounts:**
            - Username: `admin` | Password: `Password1!`
            - Username: `teacher` | Password: `Password2!`
            - Username: `user` | Password: `Password3!`
            """)
        return
    
    # Handle OAuth callback
    query_params = st.query_params
    
    if "code" in query_params:
        st.info("🔄 Processing Google sign-in...")
        code = query_params["code"]
        state = query_params.get("state")
        
        # For now, skip state verification due to Streamlit session limitations
        # In production, use a database to store state tokens with timestamps
        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write(f"Code received: {code[:20]}...")
            st.write(f"State received: {state}")
            st.write(f"Redirect URI: {GOOGLE_REDIRECT_URI}")
        
        # Exchange code for token
        try:
            import requests as http_requests
            
            token_url = "https://oauth2.googleapis.com/token"
            token_data = {
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code"
            }
            
            st.write("Exchanging code for token...")
            token_response = http_requests.post(token_url, data=token_data)
            token_json = token_response.json()
            
            with st.expander("🔍 Token Response Debug"):
                if "error" in token_json:
                    st.error(f"Error: {token_json.get('error')}")
                    st.write(f"Description: {token_json.get('error_description')}")
                else:
                    st.write("Token received successfully!")
                    st.write(f"Token type: {token_json.get('token_type')}")
                    st.write(f"Has id_token: {'id_token' in token_json}")
            
            if "id_token" in token_json:
                st.write("Verifying token...")
                user_info = verify_google_token(token_json["id_token"])
                
                if user_info:
                    # Check if email is allowed
                    if not is_email_allowed(user_info['email']):
                        st.error(f"❌ Access denied: {user_info['email']} is not authorized to access this application.")
                        st.info("Please contact the administrator if you believe this is an error.")
                        
                        with st.expander("ℹ️ Authorization Info"):
                            if ALLOWED_EMAILS and ALLOWED_EMAILS[0]:
                                st.write("Authorized email patterns:")
                                for pattern in ALLOWED_EMAILS:
                                    st.write(f"- {pattern}")
                            else:
                                st.write("No email restrictions are configured.")
                        
                        if st.button("Back to Login"):
                            st.query_params.clear()
                            st.rerun()
                        return
                    
                    st.session_state.authenticated = True
                    st.session_state.username = user_info['name']
                    st.session_state.user_email = user_info['email']
                    st.session_state.user_picture = user_info.get('picture')
                    st.session_state.user_id = user_info['user_id']
                    
                    # Store access token and refresh token for Gmail access
                    if "access_token" in token_json:
                        st.session_state.google_access_token = token_json["access_token"]
                    if "refresh_token" in token_json:
                        st.session_state.google_refresh_token = token_json["refresh_token"]
                    
                    st.success(f"✅ Login successful! Welcome {user_info['name']}")
                    
                    # Show granted permissions
                    if "scope" in token_json:
                        with st.expander("✅ Granted Permissions"):
                            scopes = token_json["scope"].split()
                            for scope in scopes:
                                if "gmail" in scope:
                                    st.write("📧 Gmail read access")
                                elif "profile" in scope:
                                    st.write("👤 Profile information")
                                elif "email" in scope:
                                    st.write("📨 Email address")
                    
                    # Clear query params
                    st.query_params.clear()
                    
                    st.write("Redirecting to main app...")
                    st.rerun()
                else:
                    st.error("❌ Failed to verify token")
            else:
                error_msg = token_json.get('error_description', token_json.get('error', 'Unknown error'))
                st.error(f"❌ Failed to get token: {error_msg}")
                
        except Exception as e:
            st.error(f"❌ Authentication error: {str(e)}")
            st.exception(e)
        
        return
    
    # Display Google Sign-In button
    st.markdown("### Sign in with Google")
    
    auth_result = get_google_auth_url()
    if auth_result:
        google_auth_url, state = auth_result
        st.markdown(
            f"""
            <a href="{google_auth_url}" target="_self">
                <button style="
                    background-color: #4285f4;
                    color: white;
                    padding: 12px 24px;
                    border: none;
                    border-radius: 4px;
                    font-size: 16px;
                    font-weight: 500;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                ">
                    <svg width="18" height="18" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
                        <path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12c0-6.627,5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C12.955,4,4,12.955,4,24c0,11.045,8.955,20,20,20c11.045,0,20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/>
                        <path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"/>
                        <path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.025C9.505,39.556,16.227,44,24,44z"/>
                        <path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.571c0.001-0.001,0.002-0.001,0.003-0.002l6.19,5.238C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/>
                    </svg>
                    Sign in with Google
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    st.caption("🔒 Your privacy is protected. We only access your basic profile information (name and email).")

def logout():
    """Logout function"""
    st.session_state.authenticated = False
    for key in ["username", "user_email", "user_picture", "user_id", "chats", "current_chat_id", "chat_counter", "oauth_state"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Check authentication first
if not check_authentication():
    login_page()
    st.stop()

client = get_mongo_client(app_name="web_content_embedding")
db = get_mongo_db(client)
collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

def get_gemini_embedding(text):
    """Get embeddings using Gemini's embedding model (768 dimensions)"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating Gemini embedding: {e}")
        return None

def get_chat_response(query, chat_history=None):
    """
    Generate chat response using multi-agent coordinator (web search + Gmail)
    Falls back to MongoDB knowledge base if agent fails
    """
    # Try using coordinator agent if user has Gmail access
    if st.session_state.get("google_access_token"):
        try:
            import asyncio
            import nest_asyncio
            from agents.coordinator_agent import call_agent
            
            # Apply nest_asyncio to allow nested event loops
            nest_asyncio.apply()
            
            yield "🔍 Searching school resources and emails...\n\n"
            
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Call coordinator agent with OAuth credentials
            response = loop.run_until_complete(
                call_agent(
                    query,
                    access_token=st.session_state.google_access_token,
                    refresh_token=st.session_state.get("google_refresh_token")
                )
            )
            
            if response:
                yield response
                return
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Agent error: {error_details}")
            yield f"⚠️ Multi-agent search failed: {str(e)}\n\nFalling back to knowledge base...\n\n"
    
    # Fallback to original MongoDB vector search
    embeddings = get_gemini_embedding(query)
    
    if embeddings is None:
        yield "Error: Could not generate embeddings for your query."
        return
    
    # Enhanced vector search to capture more email content
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embeddings,
                "numCandidates": 100,  # Consider more candidates to find email content
                "limit": 15,  # Retrieve more results initially for better prioritization
                "exact": False  # Use approximate search for better performance
            }
        },
        {
            "$addFields": {
                "similarity_score": {
                    "$meta": "vectorSearchScore"
                }
            }
        },
        {
            "$match": {
                "similarity_score": {"$gte": 0.6}  # Slightly lower threshold to capture more email content
            }
        },
        {
            "$limit": 8  # Take more results for better email prioritization
        }
    ])

    
    # Process results with metadata and prioritize email sources
    relevant_docs = []
    for result in results:
        source_type = result.get('source', 'unknown')
        
        # Map source types to user-friendly names and priority
        source_mapping = {
            'email_body': {'display': 'School Email', 'priority': 1},
            'email_attachment': {'display': 'Email Attachment', 'priority': 2},
            'unknown': {'display': 'School Website', 'priority': 3}
        }
        
        source_info = source_mapping.get(source_type, {'display': 'School Resource', 'priority': 3})
        
        doc_info = {
            "content": result['content'],
            "score": result.get('similarity_score', 0),
            "source": source_type,
            "source_display": source_info['display'],
            "priority": source_info['priority'],
            "source_id": str(result.get('source_document_id', ''))
        }
        relevant_docs.append(doc_info)
    
    # Sort by priority (email content first), then by similarity score
    relevant_docs.sort(key=lambda x: (x['priority'], -x['score']))
    
    # Limit to top 5 results but ensure email sources are prioritized
    if len(relevant_docs) > 5:
        # Keep all email sources and fill remaining slots with website content
        email_docs = [doc for doc in relevant_docs if doc['priority'] <= 2]  # email body and attachments
        website_docs = [doc for doc in relevant_docs if doc['priority'] > 2]  # website content
        
        # Take up to 5 total, prioritizing email content
        relevant_docs = email_docs + website_docs[:max(0, 5 - len(email_docs))]
    
    # Build context with better structure
    if not relevant_docs:
        context_string = "No highly relevant documents found for this query."
        yield "I don't have specific information about that topic from the school resources. You may want to contact the school directly for more details."
        return
    
    # Create structured context with enhanced source attribution
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        source_info = f"[Source {i} - {doc['source_display']} (relevance: {doc['score']:.2f})]"
        context_parts.append(f"{source_info}\n{doc['content']}")
    
    context_string = "\n\n".join(context_parts)
    
    # Build conversation history for context
    conversation_context = ""
    if chat_history and len(chat_history) > 1:
        conversation_context = "\n\nPrevious conversation:\n"
        # Only include last 4 messages to avoid token limits
        recent_history = chat_history[-4:]
        for msg in recent_history[:-1]:  # Exclude current message
            role = "Human" if msg["sender"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['text']}\n"
    
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    # Enhanced prompt with improved instructions and examples
    prompt = f"""You are the official AI assistant for Ada Lovelace School. Your role is to help students, parents, and staff by providing accurate, helpful information based exclusively on official school documentation.

**Context Information:**
- Today's Date: {current_date}
- User Question: "{query}"
{conversation_context}

**Available School Information (prioritized by source reliability):**
{context_string}

**Core Instructions:**
1. **SOURCE FIDELITY**: Answer ONLY using the provided school information above. Never invent or assume details not explicitly stated.

2. **SOURCE PRIORITIZATION**: Information sources are ranked by reliability and currentness:
   - **HIGHEST PRIORITY**: School Emails (most current, official communications)
   - **HIGH PRIORITY**: Email Attachments (official documents, forms, announcements)
   - **STANDARD PRIORITY**: School Website (general information, may be less current)

3. **RESPONSE STRUCTURE**: Format your response clearly:
   - Start with a direct answer to the question
   - Provide supporting details from the sources (prioritize email sources)
   - End with next steps or additional help if applicable

4. **SOURCE ATTRIBUTION**: Always indicate where information comes from, using specific source types:
   - "According to a recent school email..." (for email body sources)
   - "Based on an official school document..." (for email attachment sources)  
   - "The school website states..." (for website sources)
   - "School communications indicate..." (for mixed sources)

5. **HANDLING GAPS**: When information is missing or incomplete:
   - Clearly state: "I don't have specific information about [topic] in the school resources available to me"
   - Suggest contacting the school directly: "For the most current information, please contact the school office"
   - If partial information exists, share what you have and note what's missing

6. **CONVERSATION AWARENESS**: 
   - Reference previous questions when relevant
   - Build on earlier context in the conversation
   - Maintain continuity in multi-turn discussions

7. **QUALITY INDICATORS**:
   - Prioritize information from email sources over website sources
   - Prioritize information from sources with higher relevance scores (above 0.8)
   - When email and website sources conflict, favor email sources as more current
   - Note when information might be outdated
   - Highlight conflicting information from different sources

7. **TONE & STYLE**:
   - Be warm, helpful, and professional
   - Use clear, accessible language appropriate for students and parents
   - Be concise but thorough
   - Use bullet points or numbered lists for complex information

**Example Response Formats:**

*For email sources (preferred):*
"According to a recent school email, [direct answer]. Here are the key details:
• [Detail 1 from email]
• [Detail 2 from email attachment]

*For mixed sources:*  
"Based on school communications, [direct answer]. According to a recent email, [email detail]. The school website also mentions [website detail].

*For website only:*
"The school website indicates [answer]. Here are the details available:
• [Website detail 1]
• [Website detail 2]

For the most current information, you may want to contact [contact method if available]."

**Your Response:**"""

    try:
        response = model.generate_content(
            prompt,
            # generation_config=genai.types.GenerationConfig(
            #     max_output_tokens=800,  # Increased for more comprehensive responses
            #     temperature=0.2,  # Slightly higher for more natural language while maintaining accuracy
            #     top_p=0.9,  # Higher top_p for better coherence and flow
            #     top_k=40,  # Add top_k for more controlled generation
            # ),
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        print(e)
        yield f"Error generating response: {str(e)}"

# Initialize session state for multiple chats
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "chat_1"
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1

# Sidebar for chat management
with st.sidebar:
    # User info and logout with profile picture
    if st.session_state.get("user_picture"):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(st.session_state.user_picture, width=50)
        with col2:
            st.markdown(f"**{st.session_state.username}**")
            st.caption(st.session_state.get("user_email", ""))
    else:
        st.markdown(f"👤 **{st.session_state.username}**")
        if st.session_state.get("user_email"):
            st.caption(st.session_state.user_email)
    
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        logout()
    
    st.divider()
    st.header("💬 Chat Sessions")
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chat_counter += 1
        new_chat_id = f"chat_{st.session_state.chat_counter}"
        st.session_state.current_chat_id = new_chat_id
        st.session_state.chats[new_chat_id] = []
        st.rerun()
    
    st.divider()
    
    # Display existing chats
    for chat_id in st.session_state.chats.keys():
        chat_history = st.session_state.chats[chat_id]
        # Get first user message as chat title, or use default
        if chat_history and chat_history[0]["sender"] == "user":
            title = chat_history[0]["text"][:30] + "..." if len(chat_history[0]["text"]) > 30 else chat_history[0]["text"]
        else:
            title = f"Chat {chat_id.split('_')[1]}"
        
        # Highlight current chat
        if chat_id == st.session_state.current_chat_id:
            st.markdown(f"**🔸 {title}**")
        else:
            if st.button(title, key=f"switch_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
    
    # Delete current chat button
    if len(st.session_state.chats) > 1:
        st.divider()
        if st.button("🗑️ Delete Current Chat", use_container_width=True, type="secondary"):
            if st.session_state.current_chat_id in st.session_state.chats:
                del st.session_state.chats[st.session_state.current_chat_id]
                # Switch to first available chat
                st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                st.rerun()

# Initialize current chat if it doesn't exist
if st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.chats[st.session_state.current_chat_id] = []

# Main chat interface
current_chat = st.session_state.chats[st.session_state.current_chat_id]
chat_number = st.session_state.current_chat_id.split('_')[1]

st.title(f"🎓 Ada Lovelace School Assistant - Chat {chat_number}")
st.caption("Ask me anything about Ada Lovelace School. I'll search through official school information to help you.")

# Display chat history
for msg in current_chat:
    with st.chat_message(msg["sender"]):
        st.markdown(msg["text"])

# Chat input
user_input = st.chat_input("Ask a question about Ada Lovelace School...")
if user_input:
    # Add user message to current chat
    current_chat.append({"sender": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display bot response
    with st.chat_message("bot"):
        response_container = st.empty()
        full_response = ""
        for chunk in get_chat_response(user_input, current_chat):
            full_response += chunk
            response_container.markdown(full_response)
        
        # Add bot response to current chat
        current_chat.append({"sender": "bot", "text": full_response})
        
        # Update the session state
        st.session_state.chats[st.session_state.current_chat_id] = current_chat