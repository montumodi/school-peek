# Agentic Architecture for School Peek

## Overview

The web application has been rewritten to use a proper **agentic system** instead of the previous RAG (Retrieval-Augmented Generation) pattern. The new architecture uses multiple specialized agents that work together to answer user queries.

## Architecture

### Multi-Agent System

The system consists of three main components:

1. **Agent 1: Website Search Agent** (`src/agents/website_agent.py`)
   - Performs nested deep search on https://adalovelace.org.uk/
   - Uses web scraping to extract content from school website
   - Supports multiple search strategies (page search, link extraction, nested search)
   - Intelligently navigates through website structure

2. **Agent 2: Gmail Search Agent** (`src/agents/gmail_agent.py`)
   - Searches user's Gmail account for school-related emails
   - Filters emails from school domain (@adalovelace.org.uk)
   - Supports searching by keywords, subject, and date range
   - Requires Gmail OAuth authentication

3. **Agent Coordinator** (`src/agents/coordinator.py`)
   - Orchestrates the two specialized agents
   - Determines which agent(s) to use based on the user query
   - Combines results from multiple agents
   - Uses LangChain's ReAct agent framework

## Key Changes from RAG Architecture

### Before (RAG Pattern)
- Relied on pre-computed embeddings stored in MongoDB
- Required batch processing and embedding generation
- Limited to static, indexed content
- Used vector similarity search

### After (Agentic Architecture)
- Real-time search across multiple sources
- Dynamic content retrieval
- More flexible and extensible
- Intelligent query routing

## Dependencies

New dependencies added to `requirements-web.txt`:
- `langchain` - Agent framework
- `langchain-google-genai` - Gemini LLM integration
- `langchain-community` - Additional tools
- `beautifulsoup4` - Web scraping
- `google-auth-oauthlib` - Gmail OAuth
- `google-auth-httplib2` - Gmail API
- `google-api-python-client` - Gmail API client

## Configuration

### Environment Variables

Required:
- `GEMINI_API_KEY` - API key for Google Gemini LLM

Optional:
- `GMAIL_CREDENTIALS_PATH` - Path to Gmail OAuth credentials JSON file
  - Required for Gmail search functionality
  - Download from Google Cloud Console (OAuth 2.0 credentials)

### Gmail Setup (Optional)

To enable Gmail search:

1. Create a project in Google Cloud Console
2. Enable Gmail API
3. Create OAuth 2.0 credentials (Desktop app)
4. Download credentials JSON file
5. Set `GMAIL_CREDENTIALS_PATH` environment variable
6. First run will open browser for OAuth consent

## Usage

### Running the Web App

```bash
# Set environment variables
export GEMINI_API_KEY="your-api-key"
export GMAIL_CREDENTIALS_PATH="/path/to/credentials.json"  # Optional

# Run Streamlit app
cd src/web_app
streamlit run app.py
```

### How It Works

1. User asks a question
2. Coordinator agent analyzes the query
3. Coordinator decides which agent(s) to invoke:
   - Website agent for general school information
   - Gmail agent for recent communications
   - Both agents for comprehensive answers
4. Agents perform their searches
5. Coordinator combines and formats the results
6. Response is streamed back to the user

### Example Queries

**Website Search:**
- "What is the school timetable for Year 7?"
- "Where can I find the school policies?"
- "When is the next school holiday?"

**Gmail Search:**
- "Show me recent announcements from the school"
- "What did the latest parent letter say?"
- "Any recent emails about school trips?"

**Combined Search:**
- "Tell me about upcoming events"
- "What are the latest updates from school?"

## File Structure

```
src/
├── agents/
│   ├── __init__.py
│   ├── website_agent.py      # Website search agent
│   ├── gmail_agent.py         # Gmail search agent
│   └── coordinator.py         # Agent coordinator
├── web_app/
│   ├── app.py                 # New agentic web app
│   └── app_rag_old.py        # Old RAG-based app (backup)
├── config/
│   └── config.py              # Updated configuration
└── ...
```

## Benefits

1. **Real-time Information**: No need for pre-processing or embedding generation
2. **Multiple Sources**: Can search both website and Gmail simultaneously
3. **Intelligent Routing**: Automatically chooses the best source for each query
4. **Extensible**: Easy to add more agents (e.g., calendar, documents)
5. **More Accurate**: Direct source access vs. similarity matching
6. **Better Context**: Can perform deep nested searches

## Limitations and Future Improvements

### Current Limitations
- Gmail search requires manual OAuth setup
- Website search may be slower than pre-indexed search
- Rate limiting on external API calls

### Future Improvements
- Add caching layer for frequently accessed pages
- Implement rate limiting and retry logic
- Add more agents (calendar, file storage, etc.)
- Support for more sophisticated query understanding
- Add conversation memory across sessions
- Implement result ranking and relevance scoring

## Troubleshooting

### Gmail Agent Not Working
- Ensure `GMAIL_CREDENTIALS_PATH` is set correctly
- Check that Gmail API is enabled in Google Cloud Console
- Verify OAuth credentials are valid
- Run OAuth flow to generate token.pickle file

### Website Agent Timeouts
- Some pages may take longer to load
- Adjust timeout values in `website_agent.py`
- Consider adding caching

### Agent Errors
- Check that all dependencies are installed
- Verify GEMINI_API_KEY is valid
- Check internet connectivity for external API calls

## Migration from Old System

The old RAG-based system is preserved as `app_rag_old.py`. To switch back:

```bash
cd src/web_app
mv app.py app_agents.py
mv app_rag_old.py app.py
```

## Testing

To test individual agents:

```python
import sys
sys.path.append('src')

from agents.website_agent import WebsiteSearchAgent
from agents.gmail_agent import GmailSearchAgent
from agents.coordinator import AgentCoordinator

# Test website agent
website_agent = WebsiteSearchAgent(gemini_api_key="your-key")
result = website_agent.search("school timetable")
print(result)

# Test coordinator
coordinator = AgentCoordinator(gemini_api_key="your-key")
result = coordinator.answer_query("What are the school holidays?")
print(result)
```

## License

Same as the parent project.
