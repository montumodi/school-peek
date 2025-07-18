import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mongo_utils import get_mongo_client, get_mongo_db

client = get_mongo_client()
db = get_mongo_db(client)

documents = db.scraped_data.find()

for document in documents:
    document["combined_text"] = document["web_page"]["content_clean"] + "\n".join(pdf['content_clean'] for pdf in document.get('pdfs', []))
    db.scraped_data.replace_one({"_id": document["_id"]}, document)
    print("Content combined and saved to MongoDB.")