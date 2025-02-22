from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mongo_utils import get_mongo_client, get_mongo_db
from config.config import MONGODB_VECTOR_COLL_LANGCHAIN
 
client = get_mongo_client(app_name="web_content_embedding")
db = get_mongo_db(client)
collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

documents = db.scraped_data.find()
total = 0
for document in documents:
    print(document["combined_text"])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=100, add_start_index=True
    )

    document_obj = Document(page_content=document["combined_text"])
    chunks = text_splitter.split_documents([document_obj])
    total += len(chunks)

    embedding_model = SentenceTransformerEmbeddings(model_name="all-minilm")
    embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])

    for chunk, embedding in zip(chunks, embeddings):
        collection.insert_one({
            "content": chunk.page_content,
            "embedding": embedding,
            "datetime": datetime.datetime.now()
        })
print(total)