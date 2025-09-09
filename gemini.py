import os
from dotenv import load_dotenv

load_dotenv()  # Đọc các biến từ file .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(GEMINI_API_KEY)