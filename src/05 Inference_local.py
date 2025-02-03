from pymongo import MongoClient
from langchain.embeddings import SentenceTransformerEmbeddings   # Changed import to HuggingFaceEmbeddings

from langchain.embeddings import SentenceTransformerEmbeddings
from huggingface_hub import InferenceClient

from pymongo import MongoClient
from langchain.schema import Document  # Import the Document class
from config.config import MONGODB_URI, MONGODB_DATABASE_NAME, MONGODB_VECTOR_COLL_LANGCHAIN, HF_TOKEN  # Import the MongoDB URI and database name from the config file

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# MongoDB connection

client = MongoClient(MONGODB_URI, appname="web_content_embedding")  # Replace with your MongoDB URI if needed
db = client[MONGODB_DATABASE_NAME]  # Replace with your database name
collection = client.get_database(MONGODB_DATABASE_NAME).get_collection(MONGODB_VECTOR_COLL_LANGCHAIN)

# User query
query = "what is relevance between behavior incidents (negatives) and green card?"

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_query(query)
# print(query_embedding)
# Perform vector search using MongoDB $near operator
results = collection.aggregate([
    {
    "$vectorSearch": {
      "index": "vector_index",
      "path": "embedding",
      "queryVector": embeddings,
      "exact": True,
      # "numCandidates": 5,
      "limit": 1
    }
  }
])

# Retrieve matching documents
documents = []
for result in results:
    documents.append(result['content'])


# Specify search query, retrieve relevant documents, and convert to string
context_docs = documents
print(context_docs)
context_string = " ".join([doc for doc in context_docs])
# Construct prompt for the LLM using the retrieved documents as the context
prompt = f"""Use the following pieces of context to answer the question at the end. Make sure answer in bullet points
    {context_string}
    Question: {query}
"""

# Authenticate to Hugging Face and access the model
llm = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.3",
    token = HF_TOKEN)
# Prompt the LLM (this code varies depending on the model you use)
output = llm.chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=300
)
print(output.choices[0].message.content)
