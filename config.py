import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("❌ Missing GEMINI_API_KEY in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# FAST OCR + Classification (SUPPORTED!)
GEMINI_OCR = genai.GenerativeModel("models/gemini-2.0-flash-lite")
GEMINI_CLASSIFIER = genai.GenerativeModel("models/gemini-2.0-flash-lite")

# BEST accuracy for item extraction
GEMINI_ITEMS = genai.GenerativeModel("models/gemini-2.5-flash")
