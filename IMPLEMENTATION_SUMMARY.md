# Agentic Architecture Implementation Summary

## What Was Changed

The Ada Lovelace School Assistant web application has been completely rewritten to use a **multi-agent architecture** instead of the previous RAG (Retrieval-Augmented Generation) pattern.

## Key Changes

### 1. New Agent System

Created three new agent modules in `src/agents/`:

#### **Website Search Agent** (`website_agent.py`)
- Performs nested deep searches on https://adalovelace.org.uk/
- Uses web scraping with BeautifulSoup
- Implements three tools:
  - `search_page`: Search specific pages
  - `extract_links`: Find all links on a page
  - `search_nested`: Deep nested search with link following
- Configurable search depth and page limits
- Intelligent content extraction from page structure

#### **Gmail Search Agent** (`gmail_agent.py`)
- Searches user's Gmail for school-related emails
- Filters by school domain (@adalovelace.org.uk)
- Implements three tools:
  - `search_emails`: Search by keywords
  - `get_recent_emails`: Get recent emails by date range
  - `search_by_subject`: Search by email subject
- Uses Gmail API with OAuth 2.0 authentication
- Extracts and summarizes email content

#### **Agent Coordinator** (`coordinator.py`)
- Orchestrates the two specialized agents
- Routes queries to appropriate agent(s)
- Implements three tools:
  - `search_website`: Route to website agent
  - `search_gmail`: Route to Gmail agent
  - `search_both`: Query both agents simultaneously
- Uses LangChain's ReAct agent framework
- Combines results from multiple sources

### 2. Updated Web Application

- **New File**: `src/web_app/app.py` - Uses agentic architecture
- **Backup**: `src/web_app/app_rag_old.py` - Original RAG implementation preserved
- Removed dependency on MongoDB embeddings
- Real-time search across multiple sources
- Improved user experience with agent status indicators
- Added information about which agents are available

### 3. Updated Dependencies

Added to `requirements-web.txt`:
- `langchain` - Agent framework
- `langchain-google-genai` - Gemini LLM integration
- `langchain-community` - Additional tools
- `beautifulsoup4` - Web scraping
- `google-auth-oauthlib` - Gmail OAuth
- `google-auth-httplib2` - Gmail API transport
- `google-api-python-client` - Gmail API client

### 4. Configuration Changes

Updated `src/config/config.py`:
- Added `GMAIL_CREDENTIALS_PATH` environment variable for Gmail OAuth

Updated `.gitignore`:
- Excluded Gmail OAuth token files (`token.pickle`, `credentials.json`)

### 5. Documentation

Created comprehensive documentation:
- **AGENTIC_ARCHITECTURE.md** - Overview of new architecture
- **src/agents/README.md** - Detailed agent documentation
- **test_agents.py** - Test script for agents
- **example_usage.py** - Interactive examples

## Architecture Comparison

### Before (RAG Pattern)

```
User Query
    ↓
Embedding Generation
    ↓
Vector Search in MongoDB
    ↓
Context Retrieval
    ↓
LLM Response Generation
    ↓
Response to User
```

**Limitations:**
- Required pre-processing and embedding generation
- Limited to indexed content
- Static information only
- No real-time updates
- Single source (MongoDB)

### After (Agentic Pattern)

```
User Query
    ↓
Agent Coordinator
    ├→ Website Search Agent
    │     ├→ Search pages
    │     ├→ Extract links
    │     └→ Nested search
    │
    └→ Gmail Search Agent
          ├→ Search emails
          ├→ Get recent emails
          └→ Search by subject
    ↓
Result Combination
    ↓
Response to User
```

**Advantages:**
- Real-time search (no pre-processing needed)
- Multiple sources (website + Gmail)
- Dynamic content retrieval
- Intelligent query routing
- More flexible and extensible
- Better for time-sensitive information

## How It Works

1. **User asks a question** in the Streamlit web app
2. **Agent Coordinator analyzes** the query using Gemini LLM
3. **Coordinator decides** which agent(s) to use:
   - Website agent for general school info (policies, timetables, etc.)
   - Gmail agent for recent communications and announcements
   - Both agents for comprehensive answers
4. **Agents execute** their searches using specialized tools
5. **Results are combined** and formatted by coordinator
6. **Response is streamed** back to user

## Setup Instructions

### Basic Setup (Website Agent Only)

```bash
# Install dependencies
pip install -r requirements-web.txt

# Set API key
export GEMINI_API_KEY="your-gemini-api-key"

# Run the app
cd src/web_app
streamlit run app.py
```

### Full Setup (Website + Gmail Agents)

```bash
# Install dependencies
pip install -r requirements-web.txt

# Set API key
export GEMINI_API_KEY="your-gemini-api-key"

# Configure Gmail
# 1. Enable Gmail API in Google Cloud Console
# 2. Create OAuth 2.0 credentials (Desktop app)
# 3. Download credentials JSON file
export GMAIL_CREDENTIALS_PATH="/path/to/credentials.json"

# Run the app (first run will open browser for OAuth)
cd src/web_app
streamlit run app.py
```

## Testing

Run the test script:
```bash
export GEMINI_API_KEY="your-api-key"
python test_agents.py
```

Run the example script:
```bash
export GEMINI_API_KEY="your-api-key"
python example_usage.py
```

## Usage Examples

### Programmatic Usage

```python
from agents.coordinator import AgentCoordinator

# Initialize coordinator
coordinator = AgentCoordinator(
    gemini_api_key="your-key",
    gmail_credentials_path="/path/to/credentials.json"  # Optional
)

# Ask a question
result = coordinator.answer_query("What are the school holidays?")
print(result)
```

### Web App Usage

1. Open the Streamlit app
2. Login with credentials (admin/Password1!, teacher/Password2!, user/Password3!)
3. Ask questions in the chat interface
4. The system automatically selects the best agent(s)
5. Results are streamed back in real-time

## Benefits

1. **Real-time Information**: No need for batch processing or embedding generation
2. **Multiple Sources**: Can search both website and Gmail simultaneously  
3. **Intelligent Routing**: Automatically chooses the best source for each query
4. **Extensible**: Easy to add more agents (calendar, documents, etc.)
5. **More Accurate**: Direct source access vs. similarity matching
6. **Better Context**: Can perform deep nested searches
7. **Time-sensitive**: Accesses latest emails and website updates

## Limitations

1. **Gmail Setup**: Requires manual OAuth configuration
2. **Performance**: May be slower than pre-indexed search for first query
3. **Rate Limits**: Subject to external API rate limits
4. **Dependencies**: More complex dependency chain
5. **Internet Required**: Needs active internet connection

## Future Improvements

- Add caching layer for frequently accessed content
- Implement retry logic and rate limiting
- Add more specialized agents (calendar, document storage, etc.)
- Support for conversation memory across sessions
- Implement result ranking and relevance scoring
- Add parallelization for multi-agent queries
- Support for more LLM providers

## Migration Guide

### Switching Back to RAG System

If needed, you can switch back to the old RAG system:

```bash
cd src/web_app
mv app.py app_agents.py
mv app_rag_old.py app.py
```

### Database Requirements

The new system **does not require** MongoDB for basic functionality:
- Website agent: No database needed
- Gmail agent: No database needed
- Coordinator: No database needed

However, you may still want to keep MongoDB for:
- User session storage
- Query logging
- Analytics

## Files Changed/Added

### New Files
- `src/agents/__init__.py` - Agents module init
- `src/agents/website_agent.py` - Website search agent
- `src/agents/gmail_agent.py` - Gmail search agent
- `src/agents/coordinator.py` - Agent coordinator
- `src/agents/README.md` - Agent documentation
- `src/web_app/app.py` - New agentic web app
- `AGENTIC_ARCHITECTURE.md` - Architecture overview
- `test_agents.py` - Test script
- `example_usage.py` - Usage examples

### Modified Files
- `requirements-web.txt` - Added new dependencies
- `src/config/config.py` - Added Gmail config
- `.gitignore` - Excluded Gmail tokens

### Backed Up Files
- `src/web_app/app_rag_old.py` - Original RAG app (preserved)

## Support

For issues or questions:
1. Check the documentation in `AGENTIC_ARCHITECTURE.md`
2. Review agent-specific docs in `src/agents/README.md`
3. Run test script: `python test_agents.py`
4. Try examples: `python example_usage.py`

## Credits

Built using:
- LangChain - Agent framework
- Google Gemini - LLM
- Streamlit - Web interface
- BeautifulSoup - Web scraping
- Gmail API - Email access
