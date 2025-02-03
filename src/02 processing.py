from pymongo import MongoClient
import re
from config import MONGODB_URI, MONGODB_DATABASE_NAME  # Import the MongoDB URI and database name from the config file

# MongoDB connection
client = MongoClient(MONGODB_URI)  # Use the imported MongoDB URI
db = client[MONGODB_DATABASE_NAME]  # Use the imported database name

documents = db.scraped_data.find()

def remove_extra_lines_from_string(text):
    # Remove extra lines (blank lines or lines with only whitespace)
    cleaned_text = '\n'.join([line.strip() for line in text.split('\n') if line.strip() != ''])
    # Remove extra spaces (leading, trailing, and multiple spaces between words)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

for document in documents:
    document["web_page"]["content_clean"] = remove_extra_lines_from_string(document["web_page"]["content"])  # Clean the content of the web page

    if 'pdfs' in document and document['pdfs']:
        for pdf in document['pdfs']:
            pdf["content_clean"] = remove_extra_lines_from_string(pdf['content'])  # Clean the content of the PDF

    db.scraped_data.replace_one({"_id": document["_id"]}, document)
    print("Content cleaned and saved to MongoDB.")