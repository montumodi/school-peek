from pymongo import MongoClient
import re
from config import MONGODB_URI, MONGODB_DATABASE_NAME  # Import the MongoDB URI and database name from the config file

# MongoDB connection
client = MongoClient(MONGODB_URI)  # Use the imported MongoDB URI
db = client[MONGODB_DATABASE_NAME]  # Use the imported database name

documents = db.scraped_data.find()

for document in documents:
    document["combined_text"] = document["web_page"]["content_clean"] + "\n".join(pdf['content_clean'] for pdf in document.get('pdfs', []))
    db.scraped_data.replace_one({"_id": document["_id"]}, document)
    print("Content combined and saved to MongoDB.")