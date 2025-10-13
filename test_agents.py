#!/usr/bin/env python3
"""
Simple test script for the agentic architecture
Tests the Website Search Agent functionality
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.website_agent import WebsiteSearchAgent
from agents.coordinator import AgentCoordinator

def test_website_agent():
    """Test the website search agent"""
    print("=" * 60)
    print("Testing Website Search Agent")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not set. Skipping test.")
        return False
    
    try:
        agent = WebsiteSearchAgent(gemini_api_key=api_key)
        print("✓ Website agent initialized successfully")
        
        # Test a simple page fetch
        print("\n📝 Testing page fetch...")
        result = agent._fetch_page_content("https://adalovelace.org.uk")
        print(f"✓ Fetched content (length: {len(result)} chars)")
        print(f"Preview: {result[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_coordinator():
    """Test the agent coordinator"""
    print("\n" + "=" * 60)
    print("Testing Agent Coordinator")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not set. Skipping test.")
        return False
    
    try:
        coordinator = AgentCoordinator(gemini_api_key=api_key)
        print("✓ Coordinator initialized successfully")
        print("✓ Available tools:")
        for tool in coordinator.tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n🧪 Starting Agentic Architecture Tests\n")
    
    results = []
    
    # Test website agent
    results.append(("Website Agent", test_website_agent()))
    
    # Test coordinator
    results.append(("Coordinator", test_coordinator()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    # Overall result
    all_passed = all(result[1] for result in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check logs above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
