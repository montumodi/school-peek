import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import MONGODB_URI, MONGODB_DATABASE_NAME, MONGODB_VECTOR_COLL_LANGCHAIN, HF_TOKEN  # Import the MongoDB URI and database name from the config file

from flask import Flask
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from pymongo import MongoClient
import requests
from huggingface_hub import InferenceClient

# Flask Server for Dash
server = Flask(__name__)

# Dash App
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Hugging Face API Configuration
HF_EMBEDDING_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# MongoDB Connection
client = MongoClient(MONGODB_URI, appname="web_content_embedding")
db = client[MONGODB_DATABASE_NAME]
collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

def get_hf_embedding(text):
    """Fetch embeddings from Hugging Face API."""
    response = requests.post(HF_EMBEDDING_API_URL, headers=HEADERS, json={"inputs": text})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch embedding: {response.text}")

def get_chat_response(query):
    """Get response from MongoDB and Hugging Face LLM."""
    embeddings = get_hf_embedding(query)

    # Perform vector search using MongoDB
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

    # Retrieve matching document
    documents = [result['content'] for result in results]
    context_string = " ".join(documents) if documents else "No relevant documents found."

    # Construct prompt for LLM
    prompt = f"""Use the following context to answer the question in bullet points:
    {context_string}
    Question: {query}
    """

    # Call Hugging Face Inference API for LLM response
    llm = InferenceClient(HF_LLM_MODEL, token=HF_TOKEN)
    output = llm.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    return output.choices[0].message.content if output.choices else "No response generated."

# Dash Layout
app.layout = dbc.Container([
    html.H2("Chat with AI", className="text-center my-3"),

    dcc.Store(id="chat-history", data=[]),

    # Chat Box
    html.Div(id="chat-box", style={"height": "400px", "overflowY": "scroll", "border": "1px solid #ddd", "padding": "10px", "backgroundColor": "#f9f9f9"}),

    # User Input
    dbc.InputGroup([
        dbc.Input(id="user-input", placeholder="Ask a question...", type="text"),
        dbc.Button("Send", id="send-btn", n_clicks=0, color="primary"),
    ], className="my-3"),

    html.Div(id="status", className="text-muted")
])

# Callback to handle chat interaction
@app.callback(
    [Output("chat-box", "children"), Output("chat-history", "data"), Output("status", "children")],
    [Input("send-btn", "n_clicks")],
    [State("user-input", "value"), State("chat-history", "data")]
)
def update_chat(n_clicks, user_input, chat_history):
    if not user_input:
        return dash.no_update

    # Append user message
    chat_history.append({"sender": "user", "text": user_input})

    # Get AI response
    try:
        ai_response = get_chat_response(user_input)
    except Exception as e:
        ai_response = "Error: " + str(e)

    # Append AI message
    chat_history.append({"sender": "bot", "text": ai_response})

    # Create chat elements
    chat_elements = []
    for msg in chat_history:
        align_class = "text-right" if msg["sender"] == "user" else "text-left"
        bg_color = "#007bff" if msg["sender"] == "user" else "#e9ecef"
        text_color = "white" if msg["sender"] == "user" else "black"
        chat_elements.append(html.Div(msg["text"], className=f"{align_class} p-2", style={"backgroundColor": bg_color, "color": text_color, "borderRadius": "10px", "margin": "5px", "display": "inline-block"}))

    return chat_elements, chat_history, ""

# Run the app on port 80
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=True)
