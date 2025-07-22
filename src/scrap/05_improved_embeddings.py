from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mongo_utils import get_mongo_client, get_mongo_db
from utils.gemini_utils import get_gemini_embeddings
from utils.text_processing_utils import preprocess_text_for_embeddings, improve_chunk_quality, extract_keywords
from config.config import MONGODB_VECTOR_COLL_LANGCHAIN

def create_improved_embeddings():
    """
    Create embeddings with improved text processing and metadata
    """
    client = get_mongo_client(app_name="web_content_embedding")
    db = get_mongo_db(client)
    collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

    documents = db.scraped_data.find()
    total = 0
    processed = 0
    skipped = 0

    print("Starting IMPROVED embedding generation with Gemini text-embedding-004 (768 dimensions)...")

    for document in documents:
        processed += 1
        document_id = document.get('_id')
        print(f"Processing document {processed}: {document_id}")
        
        # Check if embeddings already exist for this document
        existing_embeddings = collection.count_documents({
            "source_document_id": document_id,
            "embedding_model": "gemini-text-embedding-004-improved"
        })
        
        if existing_embeddings > 0:
            print(f"Skipping document {document_id} - {existing_embeddings} embeddings already exist")
            skipped += 1
            continue
        
        # Preprocess text for better embedding quality
        raw_text = document['combined_text']
        processed_text = preprocess_text_for_embeddings(raw_text)
        
        print(f"Text preview: {processed_text[:200]}...")
        
        # Use improved chunking strategy
        text_splitter = improve_chunk_quality(
            processed_text, 
            chunk_size=800,  # Slightly smaller for better semantic coherence
            chunk_overlap=150
        )

        document_obj = Document(page_content=processed_text)
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
            
            # Extract keywords for better searchability
            keywords = extract_keywords(chunk.page_content)
            
            # Enhanced document structure with better metadata
            chunk_doc = {
                "content": chunk.page_content,
                "content_raw": raw_text[chunk.metadata.get('start_index', 0):chunk.metadata.get('start_index', 0) + len(chunk.page_content)] if chunk.metadata.get('start_index') else chunk.page_content,
                "embedding": embedding,
                "datetime": datetime.datetime.now(),
                "source_document_id": document.get("_id"),
                "embedding_model": "gemini-text-embedding-004-improved",
                "embedding_dimensions": len(embedding),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk.page_content),
                "source": document.get("source", "unknown"),
                "source_url": document.get("email_attachment_url") or document.get("email_body_source", ""),
                "content_preview": chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content,
                "keywords": keywords,
                "processing_version": "v2.0",
                "chunk_metadata": chunk.metadata
            }
            
            collection.insert_one(chunk_doc)
        
        print(f"Inserted {successful_chunks} improved chunks for document {processed}")

    print(f"Total documents processed: {processed}")
    print(f"Documents skipped (already had embeddings): {skipped}")
    print(f"New chunks processed: {total}")
    print("IMPROVED embedding generation complete!")
    print("\nIMPROVEMENTS APPLIED:")
    print("✅ Enhanced text preprocessing")
    print("✅ Better chunking strategy")
    print("✅ Keyword extraction")
    print("✅ Improved metadata structure")
    print("✅ Version tracking")

if __name__ == "__main__":
    create_improved_embeddings()
