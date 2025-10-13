# Update: Switched to Google's Native Function Calling (ADK)

## What Changed

Following the suggestion in PR comment #3397911949, the implementation has been updated to use **Google's native function calling capabilities** (Agent Development Kit / ADK) instead of LangChain.

## Key Improvements

1. **Reduced Dependencies**
   - Removed: `langchain`, `langchain-google-genai`, `langchain-community`
   - Kept: Core `google-generativeai` library

2. **Native Integration**
   - Uses Google's built-in `FunctionDeclaration` and `Tool` classes
   - Direct integration with Gemini's function calling API
   - Better performance and reliability

3. **Simpler Architecture**
   - No need for LangChain abstractions
   - Direct control over function calling loop
   - Clearer code flow

## Technical Details

### Before (LangChain)
```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", ...)
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

### After (Google Native ADK)
```python
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

genai.configure(api_key=api_key)
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    tools=[Tool(function_declarations=[...])]
)
chat = model.start_chat()
response = chat.send_message(prompt)
```

## Benefits

1. **Fewer Dependencies**: 3 fewer packages to install and maintain
2. **Better Integration**: Native Google APIs are better supported
3. **More Control**: Direct access to function calling mechanisms
4. **Simpler Code**: Less abstraction layers
5. **Future-Proof**: Google's native APIs will be better maintained

## Files Changed

- `src/agents/website_agent.py` - Refactored to use native function calling
- `src/agents/gmail_agent.py` - Refactored to use native function calling  
- `src/agents/coordinator.py` - Refactored to use native function calling
- `requirements-web.txt` - Removed LangChain dependencies

## Compatibility

The external API remains the same:
```python
# Still works exactly the same way
coordinator = AgentCoordinator(gemini_api_key="key")
result = coordinator.answer_query("What are school holidays?")
```

All existing documentation, tests, and usage examples remain valid.
