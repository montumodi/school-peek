import os
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PIL import Image
import pytesseract

from src.utils.mongo_utils import get_mongo_client, get_mongo_db
from src.utils.text_utils import remove_extra_lines_from_string
from src.utils.gemini_utils import get_gemini_embeddings
from src.utils.text_processing_utils import preprocess_text_for_embeddings, improve_chunk_quality, extract_keywords
from src.config.config import MONGODB_VECTOR_COLL_LANGCHAIN

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
    """Process text content following the improved scraping logic sequence: clean -> preprocess -> embed -> save"""
    if not text_content or not text_content.strip():
        return {"error": "No text content provided"}, 400

    cleaned_text = remove_extra_lines_from_string(text_content)
    processed_text = preprocess_text_for_embeddings(cleaned_text)
    combined_text = processed_text

    document_id = db.scraped_data.insert_one({
        "email_body_source" if source_type == "email_body" else "email_attachment_url": source_name,
        "combined_text": combined_text,
        "raw_text": text_content,
        "datetime": datetime.datetime.now(),
        "source": source_type,
        "processing_version": "v2.0"
    }).inserted_id

    text_splitter = improve_chunk_quality(
        combined_text,
        chunk_size=800,
        chunk_overlap=150
    )

    document_obj = Document(page_content=combined_text)
    chunks = text_splitter.split_documents([document_obj])

    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = get_gemini_embeddings(chunk_texts)

    successful_chunks = min(len(chunks), len(embeddings))

    for i in range(successful_chunks):
        chunk = chunks[i]
        embedding = embeddings[i]

        keywords = extract_keywords(chunk.page_content)

        chunk_doc = {
            "content": chunk.page_content,
            "content_raw": text_content[chunk.metadata.get('start_index', 0):chunk.metadata.get('start_index', 0) + len(chunk.page_content)] if chunk.metadata.get('start_index') else chunk.page_content,
            "embedding": embedding,
            "datetime": datetime.datetime.now(),
            "source_document_id": document_id,
            "embedding_model": "gemini-text-embedding-004-improved",
            "embedding_dimensions": len(embedding),
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk.page_content),
            "source": source_type,
            "source_url": source_name,
            "content_preview": chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content,
            "keywords": keywords,
            "processing_version": "v2.0",
            "chunk_metadata": chunk.metadata
        }

        db[MONGODB_VECTOR_COLL_LANGCHAIN].insert_one(chunk_doc)

    print(f"Processed {successful_chunks} improved chunks for {source_type}: {source_name}")

    return {"source": source_name, "chunks_processed": successful_chunks, "processing_version": "v2.0"}, 200

def process_file(file):
    """Process file following the scraping logic sequence: extract -> clean -> combine -> embed -> save"""
    filename = file.filename
    file_ext = filename.split(".")[-1].lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported file type: {file_ext}"}, 400

    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    content = extract_text(file_path, file_ext)
    if not content:
        return {"error": "Failed to extract text"}, 500

    try:
        os.remove(file_path)
    except OSError:
        pass

    return process_text_content(content, filename, "email_attachment")
