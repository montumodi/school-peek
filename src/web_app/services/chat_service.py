import datetime
import google.generativeai as genai

from src.utils.mongo_utils import get_mongo_client, get_mongo_db
from src.config.config import GEMINI_API_KEY, MONGODB_VECTOR_COLL_LANGCHAIN

client = get_mongo_client(app_name="web_content_embedding")
db = get_mongo_db(client)
collection = db[MONGODB_VECTOR_COLL_LANGCHAIN]

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_embedding(text):
    """Get embeddings using Gemini's embedding model (768 dimensions)"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating Gemini embedding: {e}")
        return None

def get_chat_response(query, chat_history=None):
    embeddings = get_gemini_embedding(query)

    if embeddings is None:
        yield "Error: Could not generate embeddings for your query."
        return

    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embeddings,
                "numCandidates": 50,
                "limit": 10,
                "exact": False
            }
        },
        {
            "$addFields": {
                "similarity_score": {
                    "$meta": "vectorSearchScore"
                }
            }
        },
        {
            "$match": {
                "similarity_score": {"$gte": 0.7}
            }
        },
        {
            "$limit": 5
        }
    ])

    relevant_docs = []
    for result in results:
        doc_info = {
            "content": result['content'],
            "score": result.get('similarity_score', 0),
            "source": result.get('source', 'unknown'),
            "source_id": str(result.get('source_document_id', ''))
        }
        relevant_docs.append(doc_info)

    if not relevant_docs:
        context_string = "No highly relevant documents found for this query."
        yield "I don't have specific information about that topic from the school website. You may want to contact the school directly for more details."
        return

    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        source_info = f"[Source {i} - {doc['source']} (relevance: {doc['score']:.2f})]"
        context_parts.append(f"{source_info}\n{doc['content']}")

    context_string = "\n\n".join(context_parts)

    conversation_context = ""
    if chat_history and len(chat_history) > 1:
        conversation_context = "\n\nPrevious conversation:\n"
        recent_history = chat_history[-4:]
        for msg in recent_history[:-1]:
            role = "Human" if msg["sender"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['text']}\n"

    current_date = datetime.datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are a helpful AI assistant for Ada Lovelace School, designed to provide accurate information based on official school documentation and website content.

**Today's Date:** {current_date}

**Relevant School Information (with source attribution):**
{context_string}
{conversation_context}

**Current Question:** {query}

**Instructions:**
• **Primary Source**: Use ONLY the provided school context to answer questions
• **Source Attribution**: When referencing information, mention which source it comes from (e.g., "According to the school website...")
• **Accuracy**: If information isn't in the context, clearly state "I don't have that specific information from the school resources provided"
• **Relevance Check**: Pay attention to similarity scores - higher scores indicate more relevant information
• **Completeness**: Consider the conversation history to provide contextually relevant answers
• **Clarity**: Structure responses with bullet points or numbered lists when appropriate
• **Helpfulness**: If you can't answer fully, suggest specific actions (e.g., "You may want to contact the school office at [phone] or email [email]")
• **Current Information**: When discussing dates, events, or time-sensitive information, reference today's date
• **Conversational**: Be natural and engaging while maintaining professionalism

**Response Guidelines:**
- Prioritize information from higher-scoring sources
- Be concise but comprehensive
- Use clear formatting (bullet points, headings if needed)
- Include relevant contact information when available
- Acknowledge previous questions in the conversation when relevant
- If multiple sources provide conflicting information, mention this
- If information seems outdated, note this limitation

**Answer:**"""

    try:
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=600,
                temperature=0.1,
                top_p=0.8,
            ),
            stream=True
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error generating response: {str(e)}"
