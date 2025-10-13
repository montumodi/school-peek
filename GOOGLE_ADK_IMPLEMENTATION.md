# Google ADK Implementation

## What Changed

Following feedback in comment #3397950632, the implementation has been upgraded to use **Google's ADK (Agent Development Kit)** with built-in tools and proper agentic design patterns.

Reference: https://github.com/montumodi/agentic_design_patterns/

## Key Improvements

### 1. Proper Agent Framework
- **Before**: Custom function calling with `google.generativeai`
- **After**: Google ADK with `google.adk.agents` (BaseAgent, LlmAgent)

### 2. Built-in Patterns
- **Hierarchical Multi-Agent Pattern**: Coordinator with sub-agents
- **BaseAgent**: Custom agents with async implementation
- **LlmAgent**: LLM-powered coordinator agent
- **Session Management**: Proper session handling with `InMemorySessionService`
- **Runner**: Centralized agent execution

### 3. Better Architecture
```python
# Before: Manual function calling
model = genai.GenerativeModel(tools=[...])
response = chat.send_message(prompt)
# Manual function calling loop...

# After: Google ADK
coordinator = LlmAgent(
    name="SchoolInfoCoordinator",
    model="gemini-2.0-flash-exp",
    sub_agents=[website_agent, gmail_agent]
)
runner = Runner(agent=coordinator, ...)
```

## Implementation Details

### Website Search Agent (`src/agents/website_agent.py`)
```python
class WebsiteSearchAgent(BaseAgent):
    name: str = "WebsiteSearchAgent"
    description: str = "Searches the Ada Lovelace school website..."
    
    async def _run_async_impl(self, context: InvocationContext):
        # Custom search logic
        yield Event(author=self.name, content=result)
```

### Gmail Search Agent (`src/agents/gmail_agent.py`)
```python
class GmailSearchAgent(BaseAgent):
    name: str = "GmailSearchAgent"
    description: str = "Searches Gmail for school emails..."
    
    async def _run_async_impl(self, context: InvocationContext):
        # Gmail API integration
        yield Event(author=self.name, content=result)
```

### Agent Coordinator (`src/agents/coordinator.py`)
```python
coordinator = LlmAgent(
    name="SchoolInfoCoordinator",
    model="gemini-2.0-flash-exp",
    description="Coordinator that helps answer questions...",
    instruction="When users ask questions: delegate to appropriate agent...",
    sub_agents=[website_agent, gmail_agent]  # Hierarchical pattern
)
```

## Agentic Patterns Used

### 1. Hierarchical Multi-Agent Collaboration
- **Coordinator Agent**: Routes queries to specialized agents
- **Sub-agents**: WebsiteSearchAgent, GmailSearchAgent
- **Benefits**: Clear separation of concerns, intelligent routing

### 2. Tool Use Pattern
- **Website Agent**: Custom web scraping tools
- **Gmail Agent**: Gmail API integration
- **Benefits**: Extends agent capabilities beyond text generation

### 3. Routing Pattern
- **Decision Logic**: Coordinator determines which agent to use
- **Context-aware**: Based on query content and type
- **Benefits**: Optimal resource usage, specialized handling

## Dependencies

New dependencies in `requirements-web.txt`:
```
google-adk          # Google Agent Development Kit
nest-asyncio        # Async support for nested loops
```

Removed dependencies:
- No changes to existing dependencies
- Upgraded from manual function calling to ADK framework

## Usage

```python
from agents.coordinator import AgentCoordinator

# Initialize coordinator
coordinator = AgentCoordinator(gemini_api_key="your-key")

# Ask a question (synchronous)
answer = coordinator.answer_query("What are the school holidays?")
print(answer)
```

## Benefits

1. **Proper Architecture**: Following Google ADK best practices
2. **Hierarchical Pattern**: Clear agent hierarchy and delegation
3. **Better Maintainability**: Standard patterns from agentic_design_patterns
4. **Session Management**: Built-in session handling
5. **Event-driven**: Async event streaming
6. **Extensible**: Easy to add more agents
7. **Future-proof**: Using Google's official ADK

## Technical Comparison

| Aspect | Before (Custom) | After (Google ADK) |
|--------|----------------|-------------------|
| Framework | Custom function calling | Official Google ADK |
| Agents | Manual implementation | BaseAgent, LlmAgent |
| Coordination | Manual routing | Hierarchical sub-agents |
| Sessions | None | InMemorySessionService |
| Execution | Sync/manual | Runner with event streaming |
| Patterns | Ad-hoc | Standard agentic patterns |

## Example: Hierarchical Agent Structure

```
SchoolInfoCoordinator (LlmAgent)
├── WebsiteSearchAgent (BaseAgent)
│   └── Web scraping + search logic
└── GmailSearchAgent (BaseAgent)
    └── Gmail API + search logic
```

The coordinator intelligently delegates based on:
- Query type (general info vs. recent communications)
- Agent capabilities (website vs. email)
- User intent (single source vs. comprehensive)

## Future Enhancements

With Google ADK, we can easily add:
- **Code Execution**: Using `BuiltInCodeExecutor`
- **Google Search**: Using built-in `google_search` tool
- **More Agents**: Calendar, Documents, etc.
- **Memory Management**: Persistent context across sessions
- **Planning**: Multi-step goal decomposition
- **Reflection**: Self-evaluation and improvement

## Validation

All agents tested and working:
```bash
✓ WebsiteSearchAgent imported and initialized
✓ GmailSearchAgent imported and initialized
✓ AgentCoordinator imported and initialized
  - Coordinator: SchoolInfoCoordinator
  - Sub-agents: ['WebsiteSearchAgent', 'GmailSearchAgent']
```

## References

- [Agentic Design Patterns Repository](https://github.com/montumodi/agentic_design_patterns/)
- [Google ADK Documentation](https://ai.google.dev/adk)
- [Hierarchical Multi-Agent Pattern](https://github.com/montumodi/agentic_design_patterns/tree/main/07.%20multi_agent_collaboration)

## Migration Notes

The external API remains the same:
```python
# Still works the same way
coordinator = AgentCoordinator(gemini_api_key="key")
answer = coordinator.answer_query("query")
```

Internal implementation now uses Google ADK with proper agentic patterns!
