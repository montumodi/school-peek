from flask import Flask, request
import os
from langchain_community.document_loaders import PyPDFLoader
from PIL import Image
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from langchain.schema import Document
import re
import torch

from transformers import AutoModel, AutoTokenizer

from config.config import MONGODB_URI, MONGODB_DATABASE_NAME, MONGODB_VECTOR_COLL_LANGCHAIN  # Import the MongoDB URI and database name from the config file

# MongoDB connection
client = MongoClient(MONGODB_URI)  # Use the imported MongoDB URI
db = client[MONGODB_DATABASE_NAME]  # Use the imported database name

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "txt"}

import re

def remove_extra_lines_from_string(text):
    # Remove extra lines (blank lines or lines with only whitespace)
    cleaned_text = '\n'.join([line.strip() for line in text.split('\n') if line.strip() != ''])
    # Remove extra spaces (leading, trailing, and multiple spaces between words)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def get_transformer_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.tolist()

def extract_text_from_pdf(file_path):
    """Extract text from a PDF using PyPDFLoader."""
    pdf_loader = PyPDFLoader(file_path)
    return "\n".join(page.page_content for page in pdf_loader.load())

def extract_text_from_image(file_path):
    """Extract text from an image using pytesseract."""
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)

def extract_text_from_txt(file_path):
    """Extract text from a plain text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

@app.route("/upload", methods=["POST"])
def upload_files():
    if "files" not in request.files:
        return {"error": "No files provided"}, 400

    files = request.files.getlist("files")  # Get multiple files
    extracted_data = []

    for file in files:
        filename = file.filename
        file_ext = filename.split(".")[-1].lower()

        if file_ext not in ALLOWED_EXTENSIONS:
            return {"error": f"File type {file_ext} is not supported"}, 400

        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)

        # Extract text based on file type
        if file_ext == "pdf":
            content = extract_text_from_pdf(file_path)
        elif file_ext in {"png", "jpg", "jpeg"}:
            content = extract_text_from_image(file_path)
        elif file_ext == "txt":
            content = extract_text_from_txt(file_path)
        else:
            content = "Unsupported file type"

        total = 0
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=100, add_start_index=True
        )

        document_obj = Document(page_content=remove_extra_lines_from_string(content))

        # Split document into smaller chunks
        chunks = text_splitter.split_documents([document_obj])

        total = total + len(chunks)

        # Generate embeddings for each chunk
        embeddings = get_transformer_embedding([chunk.page_content for chunk in chunks])

        db.scraped_data.insert_one({"email_attachement_url": filename, "combined_text": remove_extra_lines_from_string(content)})

        # Store embeddings in MongoDB
        for chunk, embedding in zip(chunks, embeddings):
            client.get_database(MONGODB_DATABASE_NAME).get_collection(MONGODB_VECTOR_COLL_LANGCHAIN).insert_one({
                "content": chunk.page_content,
                "embedding": embedding
                # "metadata": chunk.metadata  # optional: add source metadata
            })

        extracted_data.append({"email_attachement_url": filename})

    return extracted_data

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
