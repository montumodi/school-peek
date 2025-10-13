#!/usr/bin/env python3
"""
Example usage of the agentic architecture
Demonstrates how to use the agents programmatically
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.website_agent import WebsiteSearchAgent
from agents.gmail_agent import GmailSearchAgent
from agents.coordinator import AgentCoordinator


def example_website_search():
    """Example: Using the Website Search Agent directly"""
    print("\n" + "=" * 60)
    print("Example 1: Direct Website Search")
    print("=" * 60)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    # Initialize website agent
    agent = WebsiteSearchAgent(gemini_api_key=api_key)
    
    # Search for school timetable
    query = "What is the school timetable?"
    print(f"\nQuery: {query}")
    print("Searching...")
    
    result = agent.search(query)
    print(f"\nResult:\n{result}\n")


def example_gmail_search():
    """Example: Using the Gmail Search Agent directly"""
    print("\n" + "=" * 60)
    print("Example 2: Direct Gmail Search")
    print("=" * 60)
    
    api_key = os.getenv('GEMINI_API_KEY')
    gmail_creds = os.getenv('GMAIL_CREDENTIALS_PATH')
    
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    if not gmail_creds:
        print("Note: GMAIL_CREDENTIALS_PATH not set. Gmail search may not work.")
    
    # Initialize Gmail agent
    agent = GmailSearchAgent(
        gemini_api_key=api_key,
        credentials_path=gmail_creds
    )
    
    # Search for recent emails
    query = "Get recent emails from the school"
    print(f"\nQuery: {query}")
    print("Searching Gmail...")
    
    result = agent.search(query)
    print(f"\nResult:\n{result}\n")


def example_coordinator():
    """Example: Using the Agent Coordinator (recommended)"""
    print("\n" + "=" * 60)
    print("Example 3: Using Agent Coordinator (Recommended)")
    print("=" * 60)
    
    api_key = os.getenv('GEMINI_API_KEY')
    gmail_creds = os.getenv('GMAIL_CREDENTIALS_PATH')
    
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    # Initialize coordinator
    coordinator = AgentCoordinator(
        gemini_api_key=api_key,
        gmail_credentials_path=gmail_creds
    )
    
    # Example queries
    queries = [
        "What are the school holidays?",
        "Show me recent announcements",
        "Tell me about the school facilities"
    ]
    
    for query in queries:
        print(f"\n{'=' * 40}")
        print(f"Query: {query}")
        print(f"{'=' * 40}")
        print("Processing...")
        
        result = coordinator.answer_query(query)
        print(f"\nAnswer:\n{result}\n")


def main():
    """Run all examples"""
    print("\n🎓 Agentic Architecture Usage Examples")
    print("=" * 60)
    print("\nThese examples demonstrate how to use the new agentic")
    print("architecture for the Ada Lovelace School Assistant.\n")
    
    # Check prerequisites
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  WARNING: GEMINI_API_KEY not set!")
        print("   Please set it to run these examples:")
        print("   export GEMINI_API_KEY='your-api-key'\n")
        return 1
    
    if not os.getenv('GMAIL_CREDENTIALS_PATH'):
        print("ℹ️  NOTE: GMAIL_CREDENTIALS_PATH not set")
        print("   Gmail search will not work without it.")
        print("   Set it to enable Gmail features:")
        print("   export GMAIL_CREDENTIALS_PATH='/path/to/credentials.json'\n")
    
    try:
        # Run examples
        print("\n" + "=" * 60)
        print("Choose an example to run:")
        print("=" * 60)
        print("1. Website Search Agent")
        print("2. Gmail Search Agent")
        print("3. Agent Coordinator (Recommended)")
        print("4. Run all examples")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-4): ").strip()
        
        if choice == '1':
            example_website_search()
        elif choice == '2':
            example_gmail_search()
        elif choice == '3':
            example_coordinator()
        elif choice == '4':
            example_website_search()
            example_gmail_search()
            example_coordinator()
        elif choice == '0':
            print("Exiting...")
        else:
            print("Invalid choice!")
            return 1
        
        print("\n✅ Examples completed!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
