from utils.mongo_utils import get_mongo_client, get_mongo_db
from utils.embedding_utils import get_transformer_embedding
from huggingface_hub import InferenceClient
from config.config import HF_TOKEN

client = get_mongo_client(app_name="web_content_embedding")
db = get_mongo_db(client)
collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

query = "what is time table for year 7? Can you convert minutes in to days"
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

llm = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)
output = llm.chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=300
)

print(output.choices[0].message.content)