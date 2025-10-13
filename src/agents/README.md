# Agents Module

This module implements the agentic architecture for the Ada Lovelace School Assistant.

## Overview

The agents module consists of three main components:

1. **Website Search Agent** - Searches the school website
2. **Gmail Search Agent** - Searches user's Gmail for school emails
3. **Agent Coordinator** - Orchestrates the two agents

## Architecture

### WebsiteSearchAgent (`website_agent.py`)

Performs nested deep search on https://adalovelace.org.uk/

**Tools:**
- `search_page`: Search a specific page
- `extract_links`: Extract all links from a page
- `search_nested`: Perform deep nested search with link following

**Features:**
- Web scraping with BeautifulSoup
- Intelligent content extraction
- Visited URL tracking
- Configurable depth and page limits

**Usage:**
```python
from agents.website_agent import WebsiteSearchAgent

agent = WebsiteSearchAgent(gemini_api_key="your-key")
result = agent.search("What is the school timetable?")
print(result)
```

### GmailSearchAgent (`gmail_agent.py`)

Searches user's Gmail for emails from the school domain.

**Tools:**
- `search_emails`: Search Gmail by keywords
- `get_recent_emails`: Get recent emails from school
- `search_by_subject`: Search by subject line

**Features:**
- Gmail API integration
- OAuth 2.0 authentication
- Email content extraction
- Domain filtering (@adalovelace.org.uk)

**Setup:**
1. Enable Gmail API in Google Cloud Console
2. Create OAuth 2.0 credentials
3. Download credentials JSON
4. Set `GMAIL_CREDENTIALS_PATH` environment variable

**Usage:**
```python
from agents.gmail_agent import GmailSearchAgent

agent = GmailSearchAgent(
    gemini_api_key="your-key",
    credentials_path="/path/to/credentials.json"
)
result = agent.search("recent announcements")
print(result)
```

### AgentCoordinator (`coordinator.py`)

Coordinates multiple agents to answer queries.

**Tools:**
- `search_website`: Route to website agent
- `search_gmail`: Route to Gmail agent
- `search_both`: Search both sources

**Features:**
- Intelligent query routing
- Multi-agent orchestration
- Result combining
- LangChain ReAct framework

**Usage:**
```python
from agents.coordinator import AgentCoordinator

coordinator = AgentCoordinator(
    gemini_api_key="your-key",
    gmail_credentials_path="/path/to/credentials.json"
)
result = coordinator.answer_query("What are the school holidays?")
print(result)
```

## LangChain Integration

All agents use LangChain's agent framework:

- **ReAct Pattern**: Reasoning + Acting
- **Tools**: Specific functions each agent can use
- **LLM**: Google Gemini 1.5 Flash for decision making
- **AgentExecutor**: Manages tool calls and iterations

## Dependencies

```
langchain
langchain-google-genai
langchain-community
beautifulsoup4
requests
google-auth-oauthlib
google-auth-httplib2
google-api-python-client
```

## Configuration

### Environment Variables

Required:
- `GEMINI_API_KEY` - Google Gemini API key

Optional:
- `GMAIL_CREDENTIALS_PATH` - Path to Gmail OAuth credentials

### Agent Parameters

**WebsiteSearchAgent:**
- `gemini_api_key`: API key for Gemini LLM
- `base_url`: School website URL (default: https://adalovelace.org.uk)
- `max_depth`: Maximum depth for nested search (default: 3)
- `max_pages`: Maximum pages to visit (default: 20)

**GmailSearchAgent:**
- `gemini_api_key`: API key for Gemini LLM
- `credentials_path`: Path to Gmail OAuth credentials
- `token_path`: Path to store access token (default: token.pickle)
- `school_domain`: School email domain (default: adalovelace.org.uk)

**AgentCoordinator:**
- `gemini_api_key`: API key for Gemini LLM
- `gmail_credentials_path`: Path to Gmail OAuth credentials (optional)

## Error Handling

All agents implement error handling:
- Network errors (timeouts, connection issues)
- Authentication errors (Gmail OAuth)
- Parsing errors (malformed HTML, JSON)
- LLM errors (API failures, rate limits)

Errors are returned as descriptive messages rather than raising exceptions.

## Testing

See `test_agents.py` in the root directory:

```bash
export GEMINI_API_KEY="your-key"
python test_agents.py
```

## Examples

See `example_usage.py` in the root directory:

```bash
export GEMINI_API_KEY="your-key"
python example_usage.py
```

## Extending

To add a new agent:

1. Create new agent file (e.g., `calendar_agent.py`)
2. Implement agent class with LangChain tools
3. Add agent to coordinator's tools
4. Update coordinator routing logic

Example structure:
```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

class MyNewAgent:
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(...)
        self.tools = [
            Tool(name="tool1", func=self._tool1, description="..."),
            Tool(name="tool2", func=self._tool2, description="...")
        ]
        self.agent = self._create_agent()
    
    def _create_agent(self):
        # Create ReAct agent
        pass
    
    def _tool1(self, input_str: str) -> str:
        # Implement tool
        pass
    
    def search(self, query: str) -> str:
        # Main search method
        pass
```

## Performance Considerations

- **Caching**: Consider caching frequently accessed pages
- **Rate Limiting**: Implement rate limiting for external APIs
- **Timeouts**: Adjust timeouts based on network conditions
- **Concurrency**: Could parallelize agent calls in coordinator

## Security

- **OAuth Tokens**: Never commit token.pickle or credentials.json
- **API Keys**: Use environment variables, never hardcode
- **User Data**: Gmail agent accesses user's emails - ensure proper consent
- **Input Validation**: Sanitize all user inputs before passing to tools

## Known Limitations

1. Gmail agent requires manual OAuth setup
2. Website search may be slower than pre-indexed search
3. No caching of results (yet)
4. Limited error recovery (no retry logic)
5. Sequential agent execution (no parallelization)

## Future Improvements

- [ ] Add result caching
- [ ] Implement retry logic with exponential backoff
- [ ] Parallelize multi-agent searches
- [ ] Add more specialized agents (calendar, documents, etc.)
- [ ] Implement conversation memory
- [ ] Add relevance ranking
- [ ] Support for more LLM providers

## License

Same as parent project.
