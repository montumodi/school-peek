# Quick Start Guide - Agentic Architecture

## What Changed?

The School Peek web application now uses **intelligent agents** instead of pre-indexed database search. This means:
- ✅ **Real-time information** from school website and Gmail
- ✅ **Smarter query routing** to the best information source
- ✅ **More up-to-date answers** without needing database updates

## Getting Started (5 Minutes)

### 1. Install Dependencies

```bash
pip install -r requirements-web.txt
```

### 2. Set Your API Key (Required)

```bash
export GEMINI_API_KEY="your-actual-gemini-api-key-here"
```

**Important**: This is a **required** environment variable. The application will not work without it.

Don't have a Gemini API key? Get one free at: https://aistudio.google.com/app/apikey

### 3. Run the App

```bash
cd src/web_app
streamlit run app.py
```

### 4. Open in Browser

The app will automatically open at: http://localhost:8501

### 5. Login

Use one of these demo credentials:
- Username: `admin`, Password: `Password1!`
- Username: `teacher`, Password: `Password2!`
- Username: `user`, Password: `Password3!`

### 6. Start Chatting!

Try questions like:
- "What is the school timetable?"
- "When are the next school holidays?"
- "Tell me about school facilities"

## Optional: Enable Gmail Search

To search your Gmail for school emails:

### 1. Get Gmail Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project
3. Enable Gmail API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download the credentials JSON file

### 2. Set Credentials Path

```bash
export GMAIL_CREDENTIALS_PATH="/path/to/your/credentials.json"
```

### 3. First Run OAuth

The first time you ask about emails, a browser window will open for you to:
1. Choose your Google account
2. Grant permission to read Gmail
3. Allow the app to access your emails

A `token.pickle` file will be created to remember your authorization.

### 4. Try Gmail Queries

- "Show me recent emails from school"
- "What did the latest parent letter say?"
- "Any announcements this week?"

## How It Works

The system has two agents:

### 🌐 Website Search Agent
Searches https://adalovelace.org.uk/ for:
- School policies
- Timetables and calendars
- Facilities information
- General school information

### 📧 Gmail Search Agent (Optional)
Searches your Gmail for:
- Recent announcements
- Parent letters
- School communications
- Time-sensitive information

The **Agent Coordinator** automatically decides which agent(s) to use based on your question!

## Troubleshooting

### "GEMINI_API_KEY not set"
This is a **required** environment variable. Get your API key from https://aistudio.google.com/app/apikey
```bash
export GEMINI_API_KEY="your-actual-api-key-here"
```

### "Gmail authentication required"
```bash
export GMAIL_CREDENTIALS_PATH="/path/to/credentials.json"
```

### "Module not found"
```bash
pip install -r requirements-web.txt
```

### "Connection timeout"
- Check your internet connection
- The website agent needs to access adalovelace.org.uk

### "Agent taking too long"
- First queries may be slower as agents search in real-time
- Subsequent queries benefit from internal caching

## Testing

Test the agents without running the full app:

```bash
# Set API key first
export GEMINI_API_KEY="your-key"

# Run test script
python test_agents.py

# Try examples
python example_usage.py
```

## Command Reference

```bash
# Install dependencies
pip install -r requirements-web.txt

# Set environment variables
export GEMINI_API_KEY="your-key"
export GMAIL_CREDENTIALS_PATH="/path/to/credentials.json"  # Optional

# Run web app
cd src/web_app
streamlit run app.py

# Run tests
python test_agents.py

# Run examples
python example_usage.py

# Switch back to old RAG system (if needed - advanced)
# WARNING: This will replace the current agentic app with the old RAG system
cd src/web_app
cp app.py app_agents_backup.py  # Create backup first
cp app_rag_old.py app.py
# Verify the switch worked:
grep -q "RAG" app.py && echo "Switched to RAG system" || echo "Still using agentic system"
```

## What Questions Work Best?

### Good Questions (Website Agent):
- "What is the school day schedule?"
- "Where can I find the uniform policy?"
- "Tell me about Year 7 curriculum"
- "When are parent-teacher conferences?"
- "What facilities does the school have?"

### Good Questions (Gmail Agent):
- "Show me emails from this week"
- "Any recent announcements?"
- "What was in the latest newsletter?"
- "Recent communications about trips"

### Good Questions (Both):
- "What are the latest school updates?"
- "Tell me about upcoming events"
- "Any recent news from school?"

### Less Effective:
- Very specific questions about content that may not exist
- Questions requiring calculations or analysis
- Questions about non-school topics

## Features

### Multi-Chat Support
- Create multiple chat sessions
- Switch between chats
- Each chat maintains its own history

### Smart Agent Selection
- System automatically chooses the best agent(s)
- No need to specify which source to search

### Real-Time Streaming
- Responses stream as they're generated
- See progress as agents work

### Agent Status
- Sidebar shows which agents are active
- Clear indication of system capabilities

## Learn More

- **Full Documentation**: See `AGENTIC_ARCHITECTURE.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Architecture Diagram**: See `ARCHITECTURE_DIAGRAM.md`
- **Agent Details**: See `src/agents/README.md`

## Need Help?

1. Check error messages in the terminal
2. Review the troubleshooting section above
3. Read the full documentation
4. Try the test scripts to isolate issues

## Next Steps

Once you're comfortable with the basic setup:

1. **Customize** the agents for your specific needs
2. **Add** new agents for other data sources
3. **Integrate** with existing school systems
4. **Deploy** to a production environment

## Tips for Best Results

1. **Be Specific**: "Year 7 timetable" is better than "schedule"
2. **Use Keywords**: Include important terms from your question
3. **Try Both**: Some questions benefit from checking both sources
4. **Check Dates**: For time-sensitive info, mention the timeframe

## Security Notes

- **API Keys**: 
  - Never commit API keys to version control
  - Always use environment variables (never hardcode in files)
  - Use `.env` files with proper `.gitignore` entries for local development
  - In production, use secure secret management (AWS Secrets Manager, Azure Key Vault, etc.)
- **Gmail Access**: 
  - Users must grant explicit permission via OAuth 2.0
  - OAuth tokens are stored in `token.pickle` - keep this file private
- **Credentials**: 
  - Keep `credentials.json` and `token.pickle` private and out of version control
  - These files are already in `.gitignore`
- **Production Deployment**: 
  - Use proper secret management services
  - Never include credentials in Docker images or deployment artifacts
  - Rotate API keys regularly
  - Use least-privilege access principles

## Performance Tips

- First queries may take 5-10 seconds as agents search
- Subsequent similar queries may be faster
- Gmail queries require OAuth setup first
- Website queries depend on network speed

Enjoy your new intelligent school assistant! 🎓
