from flask import Flask, request, jsonify
import os
import re
import sys
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PIL import Image
import pytesseract
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.mongo_utils import get_mongo_client, get_mongo_db
from utils.text_utils import remove_extra_lines_from_string
from utils.gemini_utils import get_gemini_embeddings

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

def process_text_content(text_content, source_name, source_type="email_body"):
    """Process text content following the scraping logic sequence: clean -> embed -> save"""
    if not text_content or not text_content.strip():
        return {"error": "No text content provided"}, 400
    
    # Step 1: Process - Clean the text (processing.py logic)
    cleaned_text = remove_extra_lines_from_string(text_content)
    
    # Step 2: Combine text (already done since we have single text - combine.py logic)
    combined_text = cleaned_text
    
    # Save the document with combined text to scraped_data collection
    document_id = db.scraped_data.insert_one({
        "email_body_source" if source_type == "email_body" else "email_attachment_url": source_name, 
        "combined_text": combined_text,
        "datetime": datetime.datetime.now(),
        "source": source_type
    }).inserted_id
    
    # Step 3: Generate embeddings using Gemini API (embeddings.py logic)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )
    
    document_obj = Document(page_content=combined_text)
    chunks = text_splitter.split_documents([document_obj])
    
    # Get embeddings for all chunks
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = get_gemini_embeddings(chunk_texts)
    
    # Step 4: Save embeddings to vector collection
    successful_chunks = min(len(chunks), len(embeddings))
    
    for i in range(successful_chunks):
        chunk = chunks[i]
        embedding = embeddings[i]
        
        db[MONGODB_VECTOR_COLL_LANGCHAIN].insert_one({
            "content": chunk.page_content,
            "embedding": embedding,
            "datetime": datetime.datetime.now(),
            "source_document_id": document_id,
            "embedding_model": "gemini-text-embedding-004",
            "embedding_dimensions": len(embedding),
            "source": source_type
        })
    
    print(f"Processed {successful_chunks} chunks for {source_type}: {source_name}")
    
    return {"source": source_name, "chunks_processed": successful_chunks}, 200

def process_file(file):
    """Process file following the scraping logic sequence: extract -> clean -> combine -> embed -> save"""
    filename = file.filename
    file_ext = filename.split(".")[-1].lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported file type: {file_ext}"}, 400
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)
    
    # Step 1: Extract text from file (scrapping logic)
    content = extract_text(file_path, file_ext)
    if not content:
        return {"error": "Failed to extract text"}, 500
    
    # Clean up uploaded file
    try:
        os.remove(file_path)
    except OSError:
        pass
    
    # Use the common text processing function
    return process_text_content(content, filename, "email_attachment")

@app.route("/upload", methods=["POST"])
def upload_files():
    """Handle email content processing - both email body and attachments"""
    # Debug logging
    print("=== Request Debug Info ===")
    print(f"Content-Type: {request.content_type}")
    print(f"Form data keys: {list(request.form.keys())}")
    print(f"Files keys: {list(request.files.keys())}")
    print(f"Raw form data: {dict(request.form)}")
    
    # Check for email body content
    email_body = request.form.get("email_body")
    email_subject = request.form.get("email_subject") or "Email Content"
    
    print(f"Email body: {email_body}")
    print(f"Email subject: {email_subject}")
    
    # Check for file attachments - handle files with unique names
    files = []
    
    # Get all file objects from request.files (each with unique names)
    for key, file_obj in request.files.items():
        if file_obj and file_obj.filename:
            files.append(file_obj)
            print(f"Found file with key '{key}': {file_obj.filename}")
    
    print(f"Total files found: {len(files)}")
    for i, file in enumerate(files):
        print(f"File {i}: {file.filename}, Content-Type: {file.content_type}")
    
    # Check if neither email body nor files are provided
    if not email_body and not files:
        return jsonify({"error": "No email body or files provided"}), 400
    
    dry_run = request.args.get("dry_run", "false").lower() == "true"
    
    if dry_run:
        response = []
        if email_body:
            response.append({"type": "email_body", "subject": email_subject, "content_length": len(email_body)})
        if files:
            file_info = []
            for file in files:
                # Read file content to get size, then reset pointer
                content = file.read()
                file.seek(0)
                file_info.append({
                    "type": "attachment", 
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(content)
                })
            response.extend(file_info)
        return jsonify({"preview": response, "total_files": len(files)})
    
    response = []
    processed_files = 0
    failed_files = 0
    
    # Process email body if provided
    if email_body:
        try:
            result = process_text_content(email_body, email_subject, "email_body")
            response.append({"type": "email_body", "result": result})
        except Exception as e:
            print(f"Error processing email body: {str(e)}")
            response.append({"type": "email_body", "result": {"error": str(e)}, "status": "failed"})
    
    # Process file attachments if provided
    if files:
        for i, file in enumerate(files):
            try:
                print(f"Processing file {i+1}/{len(files)}: {file.filename}")
                result = process_file(file)
                response.append({"type": "attachment", "filename": file.filename, "result": result})
                if result[1] == 200:  # Check if processing was successful
                    processed_files += 1
                else:
                    failed_files += 1
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                response.append({"type": "attachment", "filename": file.filename, "result": {"error": str(e)}, "status": "failed"})
                failed_files += 1
    
    return jsonify({
        "results": response,
        "summary": {
            "total_files": len(files),
            "processed_files": processed_files,
            "failed_files": failed_files,
            "email_body_processed": bool(email_body)
        }
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)
