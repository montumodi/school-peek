import sys
import datetime
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mongo_utils import get_mongo_client, get_mongo_db
import google.generativeai as genai

from config.config import GEMINI_API_KEY, MONGODB_VECTOR_COLL_LANGCHAIN

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
    
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embeddings,
                "exact": True,
                "limit": 5  # Get more relevant documents for better context
            }
        }
    ])
    
    documents = [result['content'] for result in results]
    context_string = " ".join(documents) if documents else "No relevant documents found."
    
    # Build conversation history for context
    conversation_context = ""
    if chat_history and len(chat_history) > 1:
        conversation_context = "\n\nPrevious conversation:\n"
        # Only include last 6 messages to avoid token limits
        recent_history = chat_history[-6:]
        for msg in recent_history[:-1]:  # Exclude current message
            role = "Human" if msg["sender"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['text']}\n"
    
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    prompt = f"""You are a helpful AI assistant for Ada Lovelace School, designed to provide accurate information based on official school documentation and website content.

**Today's Date:** {current_date}

**School Context & Information:**
{context_string}
{conversation_context}

**Current Question:** {query}

**Instructions:**
• **Primary Source**: Use ONLY the provided school context to answer questions
• **Accuracy**: If information isn't in the context, clearly state "I don't have that specific information from the school website"
• **Completeness**: Consider the conversation history to provide contextually relevant answers
• **Clarity**: Structure responses with bullet points or numbered lists when appropriate
• **Helpfulness**: If you can't answer fully, suggest what the user should do (e.g., "You may want to contact the school directly at...")
• **Current Information**: When discussing dates, events, or time-sensitive information, reference today's date
• **Conversational**: Be natural and engaging while maintaining professionalism

**Response Guidelines:**
- Be concise but comprehensive
- Use clear formatting (bullet points, headings if needed)
- Include relevant links or contact information when available
- Acknowledge previous questions in the conversation when relevant
- If information is unclear or missing, say so honestly

**Answer:**"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,  # Increased for more detailed responses
                temperature=0.2,  # Slightly higher for more natural responses
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