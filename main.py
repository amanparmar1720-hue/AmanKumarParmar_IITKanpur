from fastapi import FastAPI
from pydantic import BaseModel
from app.extractor import extract_document

app = FastAPI()

class ExtractRequest(BaseModel):
    document: str

@app.post("/extract")
def extract_bill(payload: ExtractRequest):
    try:
        # CORRECT: pass only the string URL
        return extract_document(payload.document)
    except Exception as e:
        return {"is_success": False, "error": str(e)}
