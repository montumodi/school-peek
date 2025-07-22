import sys
import datetime
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mongo_utils import get_mongo_client, get_mongo_db
import google.generativeai as genai

from config.config import GEMINI_API_KEY, MONGODB_VECTOR_COLL_LANGCHAIN

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
        - Username: `admin` | Password: `admin123`
        - Username: `teacher` | Password: `teacher123`
        - Username: `user` | Password: `password123`
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

client = get_mongo_client(app_name="web_content_embedding")
db = get_mongo_db(client)
collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

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
    embeddings = get_gemini_embedding(query)
    
    if embeddings is None:
        yield "Error: Could not generate embeddings for your query."
        return
    
    # Improved vector search with better configuration
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embeddings,
                "numCandidates": 50,  # Consider more candidates
                "limit": 10,  # Retrieve more results initially
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
                "similarity_score": {"$gte": 0.7}  # Filter by relevance threshold
            }
        },
        {
            "$limit": 5  # Take top 5 after filtering
        }
    ])
    
    # Process results with metadata
    relevant_docs = []
    for result in results:
        doc_info = {
            "content": result['content'],
            "score": result.get('similarity_score', 0),
            "source": result.get('source', 'unknown'),
            "source_id": str(result.get('source_document_id', ''))
        }
        relevant_docs.append(doc_info)
    
    # Build context with better structure
    if not relevant_docs:
        context_string = "No highly relevant documents found for this query."
        yield "I don't have specific information about that topic from the school website. You may want to contact the school directly for more details."
        return
    
    # Create structured context with source attribution
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        source_info = f"[Source {i} - {doc['source']} (relevance: {doc['score']:.2f})]"
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
    
    # Improved prompt with better instructions
    prompt = f"""You are a helpful AI assistant for Ada Lovelace School, designed to provide accurate information based on official school documentation and website content.

**Today's Date:** {current_date}

**Relevant School Information (with source attribution):**
{context_string}
{conversation_context}

**Current Question:** {query}

**Instructions:**
• **Primary Source**: Use ONLY the provided school context to answer questions
• **Source Attribution**: When referencing information, mention which source it comes from (e.g., "According to the school website...")
• **Accuracy**: If information isn't in the context, clearly state "I don't have that specific information from the school resources provided"
• **Relevance Check**: Pay attention to similarity scores - higher scores indicate more relevant information
• **Completeness**: Consider the conversation history to provide contextually relevant answers
• **Clarity**: Structure responses with bullet points or numbered lists when appropriate
• **Helpfulness**: If you can't answer fully, suggest specific actions (e.g., "You may want to contact the school office at [phone] or email [email]")
• **Current Information**: When discussing dates, events, or time-sensitive information, reference today's date
• **Conversational**: Be natural and engaging while maintaining professionalism

**Response Guidelines:**
- Prioritize information from higher-scoring sources
- Be concise but comprehensive
- Use clear formatting (bullet points, headings if needed)
- Include relevant contact information when available
- Acknowledge previous questions in the conversation when relevant
- If multiple sources provide conflicting information, mention this
- If information seems outdated, note this limitation

**Answer:**"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=600,  # Increased for more detailed responses
                temperature=0.1,  # Lower for more consistent responses
                top_p=0.8,  # Add top_p for better quality
            ),
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
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