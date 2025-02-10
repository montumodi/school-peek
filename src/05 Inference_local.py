from pymongo import MongoClient
import torch
from transformers import AutoModel, AutoTokenizer
import ssl
from huggingface_hub import InferenceClient
from config.config import MONGODB_URI, MONGODB_DATABASE_NAME, MONGODB_VECTOR_COLL_LANGCHAIN, HF_TOKEN

# Disable SSL verification if needed
ssl._create_default_https_context = ssl._create_unverified_context

# Load a transformer model that generates 384-dimensional embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate 384D embeddings
def get_transformer_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling
    return embedding.tolist()  # Convert to list for MongoDB compatibility

# MongoDB connection
client = MongoClient(MONGODB_URI, appname="web_content_embedding")
db = client[MONGODB_DATABASE_NAME]
collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

# User query
query = "What is pastrol policy?"
embeddings = get_transformer_embedding(query)

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

# Retrieve matching documents
documents = [result['content'] for result in results]

# Convert retrieved documents into a string for LLM prompt
context_string = " ".join(documents)
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


# Authenticate and use Hugging Face inference API
llm = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)

# Get response from LLM
output = llm.chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=300
)

# Print response
print(output.choices[0].message.content)
