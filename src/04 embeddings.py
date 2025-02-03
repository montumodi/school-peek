from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from pymongo import MongoClient
from langchain.schema import Document  # Import the Document class
from config.config import MONGODB_URI, MONGODB_DATABASE_NAME, MONGODB_VECTOR_COLL_LANGCHAIN  # Import the MongoDB URI and database name from the config file

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# MongoDB connection
client = MongoClient(MONGODB_URI, appname="web_content_embedding")  # Use the imported MongoDB URI
db = client[MONGODB_DATABASE_NAME]  # Use the imported database name
collection = client.get_database(MONGODB_DATABASE_NAME).get_collection(MONGODB_VECTOR_COLL_LANGCHAIN)

documents = db.scraped_data.find()
total = 0
for document in documents:
    print(document["combined_text"])
    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=100, add_start_index=True
    )

    document_obj = Document(page_content=document["combined_text"])

    # Split document into smaller chunks
    chunks = text_splitter.split_documents([document_obj])

    total = total + len(chunks)

    # Generate embeddings for each chunk
    embedding_model = SentenceTransformerEmbeddings(model_name="all-minilm")
    embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])

    # Store embeddings in MongoDB
    for chunk, embedding in zip(chunks, embeddings):
        collection.insert_one({
            "content": chunk.page_content,
            "embedding": embedding
            # "metadata": chunk.metadata  # optional: add source metadata
        })
print(total)