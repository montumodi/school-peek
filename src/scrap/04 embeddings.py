from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mongo_utils import get_mongo_client, get_mongo_db
from utils.gemini_utils import get_gemini_embeddings
from config.config import MONGODB_VECTOR_COLL_LANGCHAIN
 
client = get_mongo_client(app_name="web_content_embedding")
db = get_mongo_db(client)
collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

documents = db.scraped_data.find()
total = 0
processed = 0
skipped = 0

print("Starting embedding generation with Gemini text-embedding-004 (768 dimensions)...")

for document in documents:
    processed += 1
    document_id = document.get('_id')
    print(f"Processing document {processed}: {document_id}")
    
    # Check if embeddings already exist for this document
    existing_embeddings = collection.count_documents({
        "source_document_id": document_id,
        "embedding_model": "gemini-text-embedding-004"
    })
    
    if existing_embeddings > 0:
        print(f"Skipping document {document_id} - {existing_embeddings} embeddings already exist")
        skipped += 1
        continue
    
    print(f"Text preview: {document['combined_text'][:200]}...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )

    document_obj = Document(page_content=document["combined_text"])
    chunks = text_splitter.split_documents([document_obj])
    total += len(chunks)
    
    print(f"Created {len(chunks)} chunks from document")

    # Get embeddings for all chunks in this document
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = get_gemini_embeddings(chunk_texts)
    
    # Only process chunks that got embeddings successfully
    successful_chunks = min(len(chunks), len(embeddings))
    
    for i in range(successful_chunks):
        chunk = chunks[i]
        embedding = embeddings[i]
        
        collection.insert_one({
            "content": chunk.page_content,
            "embedding": embedding,
            "datetime": datetime.datetime.now(),
            "source_document_id": document.get("_id"),
            "embedding_model": "gemini-text-embedding-004",
            "embedding_dimensions": len(embedding)
        })
    
    print(f"Inserted {successful_chunks} chunks for document {processed}")

print(f"Total documents processed: {processed}")
print(f"Documents skipped (already had embeddings): {skipped}")
print(f"New chunks processed: {total}")
print("Embedding generation complete!")
print("\nIMPORTANT: Your MongoDB vector index needs to be updated to support 768 dimensions.")
print("Please update your vector search index configuration.")