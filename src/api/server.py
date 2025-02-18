from flask import Flask, request, jsonify
import os
import re
import sys
import torch
from pymongo import MongoClient
from transformers import AutoModel, AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PIL import Image
import pytesseract
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.mongo_utils import get_mongo_client, get_mongo_db
from utils.text_utils import remove_extra_lines_from_string
from utils.embedding_utils import get_transformer_embedding


from config.config import MONGODB_VECTOR_COLL_LANGCHAIN, PORT

app = Flask(__name__)

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "txt"}
os.makedirs(UPLOAD_DIR, exist_ok=True)

client = get_mongo_client()
db = get_mongo_db(client)

def extract_text(file_path, file_ext):
    if file_ext == "pdf":
        return "\n".join(page.page_content for page in PyPDFLoader(file_path).load())
    elif file_ext in {"png", "jpg", "jpeg"}:
        return pytesseract.image_to_string(Image.open(file_path))
    elif file_ext == "txt":
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return None

def process_file(file):
    filename = file.filename
    file_ext = filename.split(".")[-1].lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported file type: {file_ext}"}, 400
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)
    
    content = extract_text(file_path, file_ext)
    if not content:
        return {"error": "Failed to extract text"}, 500
    
    cleaned_text = remove_extra_lines_from_string(content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100, add_start_index=True)
    chunks = text_splitter.split_documents([Document(page_content=cleaned_text)])
    
    db.scraped_data.insert_one({"email_attachment_url": filename, "combined_text": cleaned_text})
    
    for chunk in chunks:
        embedding = get_transformer_embedding(chunk.page_content)
        db[MONGODB_VECTOR_COLL_LANGCHAIN].insert_one({"content": chunk.page_content, "embedding": embedding})
    
    return {"email_attachment_url": filename}, 200

@app.route("/upload", methods=["POST"])
def upload_files():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist("files")
    response = [process_file(file) for file in files]
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)
