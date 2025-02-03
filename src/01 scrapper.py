from urllib.parse import urljoin
import bs4
import os
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from pymongo import MongoClient
import requests
import tempfile
from config.config import MONGODB_URI, MONGODB_DATABASE_NAME  # Import the MongoDB URI and database name from the config file

# MongoDB connection
client = MongoClient(MONGODB_URI)  # Use the imported MongoDB URI
db = client[MONGODB_DATABASE_NAME]  # Use the imported database name
os.environ["USER_AGENT"] = "dummy user agent"

bs4_strainer = bs4.SoupStrainer(class_="page-content")

def download_pdf_with_timeout(pdf_url, timeout=300):
    response = requests.get(pdf_url, timeout=timeout)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Save the content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name  # Store the path to the temporary file
    return tmp_file_path

def find_content_and_insert_into_mongo(document):
    print(f"Page URL: {document['page_url']}")
    web_page_loader = WebBaseLoader(
        web_paths=(document['page_url'],),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    web_page_doc = web_page_loader.load()[0].page_content
    print(web_page_doc)
    document['web_page'] = {"url": document['page_url'], "content": web_page_doc}
    pdfs = []
    if 'pdf_urls' in document and document['pdf_urls']:
        for pdf_url in document['pdf_urls']:
            print(f"PDF URLs: {pdf_url}")
            pdf_loader = PyPDFLoader(
                file_path=download_pdf_with_timeout(pdf_url)
            )
            pdfs.append({"url": pdf_url, "content": "\n".join(content.page_content for content in pdf_loader.load())})
        document["pdfs"] = pdfs
    del document['pdf_urls']
    del document['page_url']
    db.scraped_data.insert_one(document)
    print("inserted...")

def scrape_website_and_pdfs(base_url, db):
    visited = set()

    def scrape_page(url):
        if url in visited or not url.startswith(base_url):
            return
        print(f"Scraping: {url}")
        visited.add(url)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            return

        # Parse the page content
        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # Collect all PDF links
        pdf_links = []
        for tag in soup.find_all("a", href=True):
            link = urljoin(url, tag["href"])
            if link.endswith(".pdf") and link not in pdf_links:
                pdf_links.append(link)

        # If any PDFs are found, insert them into MongoDB
        find_content_and_insert_into_mongo({"page_url": url, "pdf_urls": pdf_links})

        # Recursively find links to other pages
        for link_tag in soup.find_all("a", href=True):
            next_page = urljoin(url, link_tag["href"])
            next_page = next_page.split("#")[0]  # Remove fragments
            if next_page not in visited and next_page.startswith(base_url):
                scrape_page(next_page)

    scrape_page(base_url)

# Example usage
if __name__ == "__main__":
    base_url = "https://adalovelace.org.uk"  # Replace with your target website
    scrape_website_and_pdfs(base_url, db)
    print("Data has been inserted into MongoDB.")
