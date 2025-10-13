import sys
import datetime
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import GEMINI_API_KEY
from agents.coordinator import AgentCoordinator

# Hardcoded credentials
VALID_CREDENTIALS = {
    "admin": "Password1!",
    "teacher": "Password2!",
    "user": "Password3!"
}

def check_authentication():
    """Check if user is authenticated"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    return st.session_state.authenticated

def login_page():
    """Display login page"""
    st.title("🔐 Ada Lovelace School Assistant - Login")
    st.markdown("Please enter your credentials to access the school assistant.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    # Show demo credentials (remove in production)
    with st.expander("🔍 Demo Credentials"):
        st.markdown("""
        **Available accounts:**
        - Username: `admin` | Password: `Password1!`
        - Username: `teacher` | Password: `Password2!`
        - Username: `user` | Password: `Password3!`
        """)

def logout():
    """Logout function"""
    st.session_state.authenticated = False
    if "username" in st.session_state:
        del st.session_state.username
    if "chats" in st.session_state:
        del st.session_state.chats
    if "current_chat_id" in st.session_state:
        del st.session_state.current_chat_id
    if "chat_counter" in st.session_state:
        del st.session_state.chat_counter
    st.rerun()

# Check authentication first
if not check_authentication():
    login_page()
    st.stop()

# Initialize agent coordinator
@st.cache_resource
def get_agent_coordinator():
    """Initialize and cache the agent coordinator"""
    gmail_creds_path = os.getenv('GMAIL_CREDENTIALS_PATH')
    return AgentCoordinator(GEMINI_API_KEY, gmail_credentials_path=gmail_creds_path)

try:
    coordinator = get_agent_coordinator()
except Exception as e:
    st.error(f"Failed to initialize agent system: {str(e)}")
    st.stop()

def get_chat_response(query, chat_history=None):
    """Get response from agent coordinator"""
    try:
        # Use agent coordinator to answer query
        answer = coordinator.answer_query(query)
        
        # Stream the response word by word for better UX
        words = answer.split()
        for i, word in enumerate(words):
            yield word + " "
            
    except Exception as e:
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
    # User info and logout
    st.markdown(f"👤 **Logged in as:** {st.session_state.username}")
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        logout()
    
    st.divider()
    st.header("💬 Chat Sessions")
    
    # Display agent status
    with st.expander("🤖 Agent Status"):
        st.markdown("""
        **Active Agents:**
        - 🌐 Website Search Agent
        - 📧 Gmail Search Agent
        
        *The system automatically selects the best agent(s) for your query.*
        """)
    
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
st.caption("Powered by Multi-Agent System: Website Search + Gmail Search")

# Information box about the new system
with st.expander("ℹ️ About this Assistant"):
    st.markdown("""
    This assistant uses an **agentic architecture** with two specialized agents:
    
    1. **🌐 Website Search Agent**: Performs deep, nested searches on the Ada Lovelace school website (https://adalovelace.org.uk/)
    2. **📧 Gmail Search Agent**: Searches your Gmail for school-related emails from the school domain
    
    The system intelligently routes your query to the appropriate agent(s) and combines results to give you the best answer.
    
    **Tips for best results:**
    - Ask specific questions about school policies, events, or recent communications
    - The system will automatically choose whether to search the website, Gmail, or both
    - For Gmail search to work, you need to configure Gmail OAuth credentials
    """)

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
        with st.spinner("🤖 Agents are working..."):
            response_container = st.empty()
            full_response = ""
            for chunk in get_chat_response(user_input, current_chat):
                full_response += chunk
                response_container.markdown(full_response)
        
        # Add bot response to current chat
        current_chat.append({"sender": "bot", "text": full_response})
        
        # Update the session state
        st.session_state.chats[st.session_state.current_chat_id] = current_chat
