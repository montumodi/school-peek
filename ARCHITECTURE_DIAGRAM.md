# Agentic Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                    (Streamlit Web App)                           │
│                      src/web_app/app.py                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ User Query
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT COORDINATOR                             │
│                  src/agents/coordinator.py                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Powered by: LangChain ReAct Agent Framework           │    │
│  │  LLM: Google Gemini 1.5 Flash                          │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Routes queries to appropriate agent(s):                        │
│  - Website search for general information                       │
│  - Gmail search for recent communications                       │
│  - Both for comprehensive answers                               │
└─────────────┬──────────────────────────────┬────────────────────┘
              │                              │
              ▼                              ▼
┌────────────────────────────┐  ┌─────────────────────────────────┐
│   WEBSITE SEARCH AGENT     │  │    GMAIL SEARCH AGENT           │
│   src/agents/              │  │    src/agents/gmail_agent.py    │
│   website_agent.py         │  │                                 │
├────────────────────────────┤  ├─────────────────────────────────┤
│                            │  │                                 │
│ Tools:                     │  │ Tools:                          │
│ ┌────────────────────┐     │  │ ┌─────────────────────────┐     │
│ │ • search_page      │     │  │ │ • search_emails         │     │
│ │ • extract_links    │     │  │ │ • get_recent_emails     │     │
│ │ • search_nested    │     │  │ │ • search_by_subject     │     │
│ └────────────────────┘     │  │ └─────────────────────────┘     │
│                            │  │                                 │
│ Features:                  │  │ Features:                       │
│ • Web scraping             │  │ • Gmail API integration         │
│ • BeautifulSoup parsing    │  │ • OAuth 2.0 authentication      │
│ • Nested link following    │  │ • Email content extraction      │
│ • Smart content extraction │  │ • Domain filtering              │
└──────────┬─────────────────┘  └──────────┬──────────────────────┘
           │                               │
           ▼                               ▼
┌────────────────────────────┐  ┌─────────────────────────────────┐
│  https://adalovelace       │  │      Gmail Account              │
│         .org.uk            │  │   (user's email)                │
│                            │  │                                 │
│  • School website          │  │  • School emails                │
│  • Policies                │  │  • Announcements                │
│  • Timetables              │  │  • Parent letters               │
│  • Events                  │  │  • Communications               │
│  • Static information      │  │  • Time-sensitive info          │
└────────────────────────────┘  └─────────────────────────────────┘


DATA FLOW:
==========

1. User Query → Agent Coordinator
2. Coordinator analyzes query using Gemini LLM
3. Coordinator decides which agent(s) to invoke
4. Agent(s) execute their specialized tools
5. Results gathered from external sources
6. Coordinator combines and formats results
7. Response streamed back to user


TECHNOLOGY STACK:
=================

┌─────────────────────────────────────────────────────────────────┐
│                       FRONTEND                                   │
│  • Streamlit - Web interface                                    │
│  • Python 3.12+                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    AGENT FRAMEWORK                               │
│  • LangChain - Agent orchestration                              │
│  • LangChain Google GenAI - Gemini integration                  │
│  • ReAct Pattern - Reasoning + Acting                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         LLM                                      │
│  • Google Gemini 1.5 Flash                                      │
│  • Used for: query analysis, routing, reasoning                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL APIS                                 │
│  • Gmail API - Email access                                     │
│  • HTTP Requests - Website scraping                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    UTILITIES                                     │
│  • BeautifulSoup4 - HTML parsing                                │
│  • Google Auth - OAuth 2.0                                      │
│  • Requests - HTTP client                                       │
└─────────────────────────────────────────────────────────────────┘


COMPARISON WITH OLD ARCHITECTURE:
==================================

OLD (RAG Pattern):                    NEW (Agentic Pattern):
─────────────────                     ──────────────────────

User Query                            User Query
    ↓                                     ↓
Generate Embedding                    Agent Coordinator
    ↓                                     ├→ Website Agent
MongoDB Vector Search                     │  └→ Real-time scraping
    ↓                                     └→ Gmail Agent
Retrieve Context                             └→ Real-time search
    ↓                                     ↓
Generate Response                     Combine Results
    ↓                                     ↓
Return to User                        Return to User

Pros:                                 Pros:
• Fast (pre-indexed)                  • Real-time information
• Predictable performance             • Multiple sources
                                      • More flexible
Cons:                                 • Intelligent routing
• Requires batch processing           • Better for time-sensitive queries
• Static information only             
• Single source                       Cons:
• Stale data                          • Slower for first query
                                      • More complex setup
                                      • Internet dependency


AGENT DECISION LOGIC:
====================

Coordinator receives query and decides:

┌────────────────────────────────────────┐
│ Query Type          | Agent(s) Used    │
├────────────────────────────────────────┤
│ General school info | Website Agent    │
│ Policies            | Website Agent    │
│ Timetables          | Website Agent    │
│ Events              | Website Agent    │
│ Contact info        | Website Agent    │
├────────────────────────────────────────┤
│ Recent emails       | Gmail Agent      │
│ Announcements       | Gmail Agent      │
│ Parent letters      | Gmail Agent      │
│ Class-specific      | Gmail Agent      │
├────────────────────────────────────────┤
│ "Latest updates"    | Both Agents      │
│ "Recent info"       | Both Agents      │
│ "Comprehensive"     | Both Agents      │
└────────────────────────────────────────┘
```
