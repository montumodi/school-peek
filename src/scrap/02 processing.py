from utils.mongo_utils import get_mongo_client, get_mongo_db
from utils.text_utils import remove_extra_lines_from_string

client = get_mongo_client()
db = get_mongo_db(client)

documents = db.scraped_data.find()

for document in documents:
    document["web_page"]["content_clean"] = remove_extra_lines_from_string(document["web_page"]["content"])

    if 'pdfs' in document and document['pdfs']:
        for pdf in document['pdfs']:
            pdf["content_clean"] = remove_extra_lines_from_string(pdf['content'])

    db.scraped_data.replace_one({"_id": document["_id"]}, document)
    print("Content cleaned and saved to MongoDB.")