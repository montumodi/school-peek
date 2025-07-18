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

def get_chat_response(query):
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
                "limit": 1
            }
        }
    ])
    
    documents = [result['content'] for result in results]
    context_string = " ".join(documents) if documents else "No relevant documents found."
    
    prompt = f"""
You are an AI assistant trained to provide answers based on official school information. 
Use only the provided context to answer the question factually and concisely. Use today's date ({datetime.datetime.now()}) for any time-related information.

Context:
{context_string}

Question: {query}

Guidelines:
- Respond using **only the given context** (do not guess).
- If the answer is **not in the context**, reply: "I couldn't find relevant information on the school website."
- Provide answers in **bullet points** for clarity.
- If applicable, include **links or references** from the website.

Answer:
"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.1,
            ),
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error generating response: {str(e)}"

st.title("Chat with AI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["sender"]):
        st.markdown(msg["text"])

user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.chat_history.append({"sender": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("bot"):
        response_container = st.empty()
        full_response = ""
        for chunk in get_chat_response(user_input):
            full_response += chunk
            response_container.markdown(full_response)
        st.session_state.chat_history.append({"sender": "bot", "text": full_response})