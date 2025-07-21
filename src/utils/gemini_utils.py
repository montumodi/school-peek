import google.generativeai as genai
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import GEMINI_API_KEY

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_embeddings(texts):
    """Get embeddings for multiple texts using Gemini's embedding model (768 dimensions)"""
    embeddings = []
    for text in texts:
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"  # Use retrieval_document for storing documents
            )
            embeddings.append(result['embedding'])
            print(f"Generated embedding with {len(result['embedding'])} dimensions")
        except Exception as e:
            print(f"Error generating embedding for text: {e}")
            # Skip this text if embedding fails
            continue
    return embeddings
