# import sys
# import os
# import streamlit as st
# from ollama import Client as OllamaClient
# from typing import Generator

# # Add parent directory to path for utils imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from utils.mongo_utils import get_mongo_client, get_mongo_db
# from utils.embedding_utils import get_transformer_embedding
# from config.config import MONGODB_VECTOR_COLL_LANGCHAIN, MONGODB_URI, OLLAMA_URL

# # Initialize MongoDB connection
# mongo_client = get_mongo_client(app_name="web_content_embedding")
# db = get_mongo_db(mongo_client)
# collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

# # Initialize Ollama client (using the provided host)
# ollama_client = OllamaClient(
#     host=OLLAMA_URL
# )

# def ollama_generator(model_name: str, messages: list) -> Generator:
#     """
#     Yields streamed output from Ollama's chat API.
#     """
#     stream = ollama_client.chat(model=model_name, messages=messages, stream=True)
#     for chunk in stream:
#         yield chunk['message']['content']

# def get_chat_response(query: str) -> Generator:
#     """
#     Retrieves relevant context from MongoDB, builds a prompt, and
#     returns a generator that streams the AI assistant's response from Ollama.
#     """
#     # Get the query embedding and search for similar documents
#     embeddings = get_transformer_embedding(query)
#     results = collection.aggregate([
#         {
#             "$vectorSearch": {
#                 "index": "vector_index",
#                 "path": "embedding",
#                 "queryVector": embeddings,
#                 "exact": True,
#                 "limit": 1
#             }
#         }
#     ])
#     documents = [result['content'] for result in results]
#     context_string = " ".join(documents) if documents else "No relevant documents found."
    
#     # Build the prompt with the retrieved context
#     prompt = f"""
# You are an AI assistant trained to provide answers based on official school information. 
# Use only the provided context to answer the question factually and concisely.

# Context:
# {context_string}

# Question: {query}

# Guidelines:
# - Respond using **only the given context** (do not guess).
# - If the answer is **not in the context**, reply: "I couldn't find relevant information on the school website."
# - Provide answers in **bullet points** for clarity.
# - If applicable, include **links or references** from the website.

# Answer:
# """
#     messages = [{"role": "user", "content": prompt}]
#     return ollama_generator(model_name=st.session_state.selected_model, messages=messages)

# # --- Streamlit Chat App ---

# st.title("Chat with AI using Ollama")

# # Let the user select a model.
# model_options = [m["model"] for m in ollama_client.list()["models"]]
# selected_model = st.selectbox(
#     "Please select the model:",
#     model_options,
#     index=model_options.index(st.session_state.get("selected_model", model_options[0]))
# )
# st.session_state.selected_model = selected_model

# # Initialize chat history if not present
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display the chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Get new user input
# user_input = st.chat_input("Ask a question...")
# if user_input:
#     # Append user message to session state and display it
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)
    
#     # Generate and stream the assistant's response
#     with st.chat_message("assistant"):
#         response_container = st.empty()
#         full_response = ""
#         for chunk in get_chat_response(user_input):
#             full_response += chunk
#             response_container.markdown(full_response)
#         st.session_state.messages.append({"role": "assistant", "content": full_response})
