import re
import string

def preprocess_text_for_embeddings(text):
    """
    Preprocess text to improve embedding quality
    """
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove or normalize special characters that don't add semantic value
    text = re.sub(r'[^\w\s\-.,;:!?()"]', ' ', text)
    
    # Normalize multiple punctuation
    text = re.sub(r'[.]{2,}', '...', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    # Remove URLs that might not be relevant for semantic search
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    
    # Remove email addresses for privacy (replace with placeholder)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove phone numbers (replace with placeholder)
    text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', '[PHONE]', text)
    
    # Normalize common abbreviations
    abbreviations = {
        'etc.': 'etcetera',
        'e.g.': 'for example',
        'i.e.': 'that is',
        'vs.': 'versus',
        'Mr.': 'Mister',
        'Mrs.': 'Missus',
        'Dr.': 'Doctor'
    }
    
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    
    return text.strip()

def improve_chunk_quality(text, chunk_size=1000, chunk_overlap=200):
    """
    Improve text chunking for better semantic coherence
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Use sentence-aware splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " "],
        add_start_index=True,
        strip_whitespace=True
    )
    
    return text_splitter

def extract_keywords(text, max_keywords=10):
    """
    Extract important keywords from text for better retrieval
    """
    # Simple keyword extraction (can be enhanced with NLP libraries)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
        'after', 'above', 'below', 'out', 'off', 'over', 'under', 'again', 
        'further', 'then', 'once', 'this', 'that', 'these', 'those', 'are',
        'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'
    }
    
    # Count word frequency
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Return most frequent words
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
    return [word for word, freq in keywords]
