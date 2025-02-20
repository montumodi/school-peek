import torch
from transformers import AutoModel, AutoTokenizer
from functools import lru_cache

model_name = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def get_transformer_embedding(text):
    tokenizer, model = get_model_and_tokenizer()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.tolist()
