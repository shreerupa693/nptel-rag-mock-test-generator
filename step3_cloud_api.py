import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("[SYSTEM ERROR] GEMINI_API_KEY not found. Please check your .env file.")

# Initialize the official client (natively supports AQ... keys)
client = genai.Client(api_key=api_key)

def ask_cloud_llm(query: str, retrieved_context: list[dict]) -> str:
    """
    Step 3 Practical Implementation: Offloads reasoning to Gemini 
    using the official SDK to support modern AQ... credentials.
    """
    # 1. Format the database text snippets cleanly
    formatted_chunks = []
    for c in retrieved_context:
        meta = c.get("metadata", {})
        source = meta.get("source", "N/A")
        page = meta.get("page", 0)
        text = c.get("text", "")
        formatted_chunks.append(f"[Slide Source: {source}, Page: {page}]\n{text}")
        
    context_str = "\n\n".join(formatted_chunks)

    # 2. Construct the prompt
    prompt = (
        "You are an expert, strict NPTEL examiner for the course.\n"
        "Your task is to answer the user's question using ONLY the provided context.\n"
        "CRITICAL RULES:\n"
        "1. Do not use any outside knowledge.\n"
        "2. Always cite the Source and Page number from the context.\n\n"
        f"CONTEXT MATERIAL:\n{context_str}\n\n"
        f"USER REQUEST / QUESTION: {query}\n\n"
        "ANSWER:"
    )

    try:
        # 3. Call Gemini using the official SDK client
        response = client.models.generate_content(
            model="gemini-3.8-flash",  # Or your desired model version
            contents=prompt,
        )
        return response.text
        
    except Exception as e:
        return f"[NETWORK / API ERROR] Failed to fetch from cloud engine: {str(e)}"