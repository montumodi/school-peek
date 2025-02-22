from sentence_transformers import SentenceTransformer
from functools import lru_cache

model_name = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(model_name)

def get_transformer_embedding(text):
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
