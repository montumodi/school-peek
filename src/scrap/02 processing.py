import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.text_utils import remove_extra_lines_from_string
from utils.mongo_utils import get_mongo_client, get_mongo_db

db = get_mongo_db(get_mongo_client())

documents = db.scraped_data.find()

for document in documents:
    document["web_page"]["content_clean"] = remove_extra_lines_from_string(document["web_page"]["content"])

    if 'pdfs' in document and document['pdfs']:
        for pdf in document['pdfs']:
            pdf["content_clean"] = remove_extra_lines_from_string(pdf['content'])

    db.scraped_data.replace_one({"_id": document["_id"]}, document)
    print("Content cleaned and saved to MongoDB.")