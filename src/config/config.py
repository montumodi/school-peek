import os

MONGODB_URI = os.environ["MONGODB_URI"]  # Read the MongoDB URI from the environment variable
MONGODB_DATABASE_NAME = "ada"
MONGODB_VECTOR_COLL_LANGCHAIN = "website_embeddings"
MONGODB_VECTOR_INDEX = "vector_index"
HF_TOKEN = os.environ["HF_TOKEN"]  # Read the Hugging Face token from the environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Read the Gemini API key from the environment variable
PORT = os.environ["PORT"]  # Read the port from the environment variable
