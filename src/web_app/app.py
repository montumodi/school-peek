import sys
import os
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
from pymongo import MongoClient
from huggingface_hub import InferenceClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import MONGODB_URI, MONGODB_DATABASE_NAME, MONGODB_VECTOR_COLL_LANGCHAIN, HF_TOKEN

# Load transformer model for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_transformer_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.tolist()

# MongoDB Connection
client = MongoClient(MONGODB_URI, appname="web_content_embedding")
db = client[MONGODB_DATABASE_NAME]
collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

# Hugging Face LLM Configuration
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def get_chat_response(query):
    """Fetches response from MongoDB and Hugging Face LLM."""
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
    
    prompt = f"""Use the following context to answer the question:
    {context_string}
    Question: {query}
    """
    
    llm = InferenceClient(HF_LLM_MODEL, token=HF_TOKEN)
    output = llm.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        stream=True  # Enable streaming
    )
    
    for response in output:
        yield response.choices[0].delta.content if response.choices else ""

# Streamlit UI
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
