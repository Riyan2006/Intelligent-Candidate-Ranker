import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Initialize the Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

try:
    # Using the current 2026 multimodal embedding model
    response = client.models.embed_content(
        model='gemini-embedding-2-preview',
        contents='Vibe coding the Redrob Hackathon.'
    )

    # Extract the vector
    vector = response.embeddings[0].values
    print(f"✅ Success! Generated embedding vector of length: {len(vector)}")

except Exception as e:
    print(f"❌ API Error: {e}")