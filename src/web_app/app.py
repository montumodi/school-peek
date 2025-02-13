import sys
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mongo_utils import get_mongo_client, get_mongo_db
from utils.embedding_utils import get_transformer_embedding
from huggingface_hub import InferenceClient

from config.config import HF_TOKEN, MONGODB_VECTOR_COLL_LANGCHAIN

client = get_mongo_client(app_name="web_content_embedding")
db = get_mongo_db(client)
collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def get_chat_response(query):
    embeddings = get_transformer_embedding(query)
    
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
Use only the provided context to answer the question factually and concisely.

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
    
    llm = InferenceClient(HF_LLM_MODEL, token=HF_TOKEN)
    output = llm.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        stream=True
    )
    
    for response in output:
        yield response.choices[0].delta.content if response.choices else ""

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