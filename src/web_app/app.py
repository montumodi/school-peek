import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from web_app.services.auth_service import check_authentication, login_page, logout
from web_app.services.chat_service import get_chat_response

if not check_authentication():
    login_page()
    st.stop()

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "chat_1"
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1

with st.sidebar:
    st.markdown(f"👤 **Logged in as:** {st.session_state.username}")
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        logout()
    
    st.divider()
    st.header("💬 Chat Sessions")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chat_counter += 1
        new_chat_id = f"chat_{st.session_state.chat_counter}"
        st.session_state.current_chat_id = new_chat_id
        st.session_state.chats[new_chat_id] = []
        st.rerun()
    
    st.divider()
    
    for chat_id in st.session_state.chats.keys():
        chat_history = st.session_state.chats[chat_id]
        if chat_history and chat_history[0]["sender"] == "user":
            title = chat_history[0]["text"][:30] + "..." if len(chat_history[0]["text"]) > 30 else chat_history[0]["text"]
        else:
            title = f"Chat {chat_id.split('_')[1]}"
        
        if chat_id == st.session_state.current_chat_id:
            st.markdown(f"**🔸 {title}**")
        else:
            if st.button(title, key=f"switch_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
    
    if len(st.session_state.chats) > 1:
        st.divider()
        if st.button("🗑️ Delete Current Chat", use_container_width=True, type="secondary"):
            if st.session_state.current_chat_id in st.session_state.chats:
                del st.session_state.chats[st.session_state.current_chat_id]
                st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                st.rerun()

if st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.chats[st.session_state.current_chat_id] = []

current_chat = st.session_state.chats[st.session_state.current_chat_id]
chat_number = st.session_state.current_chat_id.split('_')[1]

st.title(f"🎓 Ada Lovelace School Assistant - Chat {chat_number}")
st.caption("Ask me anything about Ada Lovelace School. I'll search through official school information to help you.")

for msg in current_chat:
    with st.chat_message(msg["sender"]):
        st.markdown(msg["text"])

user_input = st.chat_input("Ask a question about Ada Lovelace School...")
if user_input:
    current_chat.append({"sender": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("bot"):
        response_container = st.empty()
        full_response = ""
        for chunk in get_chat_response(user_input, current_chat):
            full_response += chunk
            response_container.markdown(full_response)
        
        current_chat.append({"sender": "bot", "text": full_response})
        
        st.session_state.chats[st.session_state.current_chat_id] = current_chat