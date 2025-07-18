from urllib.parse import urljoin
import bs4
import sys
import os
import datetime
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
import requests
import tempfile
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mongo_utils import get_mongo_client, get_mongo_db

client = get_mongo_client()
db = get_mongo_db(client)
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

bs4_strainer = bs4.SoupStrainer(class_="page-content")

def download_pdf_with_timeout(pdf_url, timeout=300, max_retries=3, retry_delay=1):
    """
    Download a PDF with timeout and retry logic.
    
    Args:
        pdf_url (str): URL of the PDF to download
        timeout (int): Request timeout in seconds
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
    
    Returns:
        str: Path to the downloaded temporary file, or None if skipped
    
    Raises:
        requests.RequestException: If all retry attempts fail for retryable errors
    """
    # HTTP status codes that should not be retried (permanent failures)
    non_retryable_status_codes = {400, 401, 403, 404, 410, 421, 422, 451}
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(pdf_url, timeout=timeout)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            return tmp_file_path
        except requests.HTTPError as e:
            # Check if it's a non-retryable HTTP error
            if hasattr(e.response, 'status_code') and e.response.status_code in non_retryable_status_codes:
                print(f"Skipping PDF due to non-retryable error {e.response.status_code}: {pdf_url}")
                print(f"Error details: {e}")
                return None
            last_exception = e
        except requests.RequestException as e:
            last_exception = e
        
        # Only retry if we haven't hit max attempts and it's a retryable error
        if attempt < max_retries:
            print(f"Failed to download PDF (attempt {attempt + 1}/{max_retries + 1}): {last_exception}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            # Exponential backoff: increase delay for next retry
            retry_delay *= 2
        else:
            print(f"All retry attempts failed for PDF: {pdf_url}")
            raise last_exception

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
            pdf_file_path = download_pdf_with_timeout(pdf_url)
            if pdf_file_path is not None:
                pdf_loader = PyPDFLoader(file_path=pdf_file_path)
                pdfs.append({"url": pdf_url, "content": "\n".join(content.page_content for content in pdf_loader.load())})
            else:
                print(f"Skipped PDF due to error: {pdf_url}")
        document["pdfs"] = pdfs
    del document['pdf_urls']
    del document['page_url']
    document['timestamp'] = datetime.datetime.now()
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

        soup = bs4.BeautifulSoup(response.text, "html.parser")
        pdf_links = []
        for tag in soup.find_all("a", href=True):
            link = urljoin(url, tag["href"])
            if link.endswith(".pdf") and link not in pdf_links:
                pdf_links.append(link)

        find_content_and_insert_into_mongo({"page_url": url, "pdf_urls": pdf_links})

        for link_tag in soup.find_all("a", href=True):
            next_page = urljoin(url, link_tag["href"])
            next_page = next_page.split("#")[0]
            if next_page not in visited and next_page.startswith(base_url):
                scrape_page(next_page)

    scrape_page(base_url)

if __name__ == "__main__":
    base_url = "https://adalovelace.org.uk"
    scrape_website_and_pdfs(base_url, db)
    print("Data has been inserted into MongoDB.")