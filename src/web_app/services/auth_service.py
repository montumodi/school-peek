import streamlit as st
from src.config.config import VALID_CREDENTIALS

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
    st.session_state.pop('username', None)
    st.session_state.pop('chats', None)
    st.session_state.pop('current_chat_id', None)
    st.session_state.pop('chat_counter', None)
    st.rerun()
