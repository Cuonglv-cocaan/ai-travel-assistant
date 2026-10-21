
from src.config import GEMINI_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY chưa được tải từ file .env")

print("API key đã được tải.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
    max_retries=3,
)

response = llm.invoke("Reply with exactly: Gemini connection successful.")

print("Gemini response:", response.content)
print("TEST PASSED: Gemini API đang hoạt động.")
