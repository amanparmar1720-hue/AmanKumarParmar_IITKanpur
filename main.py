from fastapi import FastAPI
from pydantic import BaseModel
from app.extractor import extract_from_url

app = FastAPI()

class ExtractRequest(BaseModel):
    document: str   # URL (pdf or image)

@app.post("/extract")
def extract_document(payload: ExtractRequest):
    try:
        return extract_from_url(payload.document)
    except Exception as e:
        return {
            "is_success": False,
            "error": str(e)
        }
