import google.generativeai as genai

# Your API Key
GEMINI_API_KEY = "AIzaSyCsxUpOJfLvsyXahiIc98x0RkpQfy135iU"

# Configure client
genai.configure(api_key=GEMINI_API_KEY)

# Instantiate Gemini model
GEMINI_MODEL = genai.GenerativeModel("models/gemini-2.5-flash")
