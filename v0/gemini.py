import os
from dotenv import load_dotenv

load_dotenv()  # Đọc các biến từ file .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash-lite")
response = model.generate_content("Hello, how are you?")
print(response.text)