import os

MONGODB_URI = os.environ["MONGODB_URI"]  # Read the MongoDB URI from the environment variable
MONGODB_DATABASE_NAME = "ada"
MONGODB_VECTOR_COLL_LANGCHAIN = "website_embeddings"
MONGODB_VECTOR_INDEX = "vector_index"
HF_TOKEN = os.environ["HF_TOKEN"]  # Read the Hugging Face token from the environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Read the Gemini API key from the environment variable
PORT = os.environ["PORT"]  # Read the port from the environment variable

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")  # Google OAuth Client ID
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")  # Google OAuth Client Secret
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")  # OAuth redirect URI

# Allowed email addresses/domains for login
# Can be specific emails: ["admin@school.edu", "teacher@school.edu"]
# Or domains: ["@school.edu", "@admin.school.edu"]
# Or mix of both: ["admin@example.com", "@school.edu"]
ALLOWED_EMAILS = os.environ.get("ALLOWED_EMAILS", "").split(",") if os.environ.get("ALLOWED_EMAILS") else []
# If empty, all Google accounts are allowed (not recommended for production)
