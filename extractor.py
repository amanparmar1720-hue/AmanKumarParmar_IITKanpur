from app.config import GEMINI_OCR, GEMINI_CLASSIFIER, GEMINI_ITEMS

import fitz
import json
import requests
import google.generativeai as genai

import logging
import time

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)



# -----------------------------------------------------
# CONFIG GEMINI
# -----------------------------------------------------



# -----------------------------------------------------
# HELPERS FOR FILE TYPE
# -----------------------------------------------------
def is_pdf(url: str):
    return ".pdf" in url.lower()



def is_image(url: str):
    url = url.lower()
    return (".png" in url) or (".jpg" in url) or (".jpeg" in url)


# -----------------------------------------------------
# DOWNLOAD REMOTE FILE
# -----------------------------------------------------
def download_from_url(url: str):
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise Exception("Failed to download document")
    return resp.content


# -----------------------------------------------------
# PDF → IMAGES (old working code)
# -----------------------------------------------------
def convert_pdf_to_images(pdf_bytes: bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=200)
        images.append(pix.tobytes("png"))

    return images


# -----------------------------------------------------
# GEMINI OCR
# -----------------------------------------------------
def gemini_ocr_extract(img_bytes: bytes):
    resp = GEMINI_OCR.generate_content(
        [
            "Extract ONLY visible text from this hospital bill.",
            {"mime_type": "image/png", "data": img_bytes}
        ]
    )
    text = resp.text if hasattr(resp, "text") else ""
    return text, resp.usage_metadata


# -----------------------------------------------------
# PAGE CLASSIFICATION
# -----------------------------------------------------
class DummyUsage:
    def __init__(self):
        self.prompt_token_count = 0
        self.candidates_token_count = 0


def classify_page_text(text: str):
    t = text.lower()
    lines = [l.strip() for l in t.split("\n") if l.strip()]

    # =====================================================
    # 1) FINAL BILL — STRONG SIGNALS (ALWAYS RUN FIRST)
    # =====================================================
    final_bill_signals = [
        "patient name", "bill no", "uhid",
        "admission date", "discharge date",
        "net amount payable", "net bill amount",
        "total bill amount", "grand total", "amount received"
    ]

    final_bill_categories = [
        "consultation", "room rent", "bed charges", "investigation",
        "laboratory", "radiology", "pathology",
        "medical consumable", "drugs", "procedures",
        "surgery", "miscellaneous services"
    ]

    strong_final = sum(1 for k in final_bill_signals if k in t) >= 2
    category_final = sum(1 for k in final_bill_categories if k in t) >= 2

    if strong_final and category_final:
        return "Final Bill", DummyUsage()

    # =====================================================
    # 2) NON-PHARMACY HARD OVERRIDE
    # (If these appear → page CANNOT be pharmacy)
    # =====================================================
    non_pharmacy_hard = [
        "consultation", "consultant",
        "dr.", "doctor", "surgeon", "general surgery",
        "dietician", "dietary",
        "procedure", "surgery",
        "ward", "room rent"
    ]

    if any(k in t for k in non_pharmacy_hard):
        return "Bill Detail", DummyUsage()

    # =====================================================
    # 3) BILL DETAIL — STRONG RULES ONLY
    # =====================================================
    import re
    item_row_pattern = r"\b(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)"
    line_items = len(re.findall(item_row_pattern, t))

    if line_items >= 5:
        return "Bill Detail", DummyUsage()

    bill_detail_keywords = [
        "x-ray", "xray", "usg", "ultrasound", "echo", "2d echo",
        "mri", "ct", "ecg",
        "cbc", "blood", "culture",
        "serum", "lipid", "crp",
        "lft", "kft", "widal", "hba1c",
        "radiology", "pathology", "laboratory"
    ]

    if any(kw in t for kw in bill_detail_keywords):
        return "Bill Detail", DummyUsage()

    # =====================================================
    # 4) PHARMACY — STRICT & CONTEXTUAL (RUN LAST)
    # =====================================================
    pharmacy_keywords = [
        "pharmacy", "drug", "batch", "batch no", "mtrs name",
        "tablet", "tab ", "cap ", "capsule",
        "inj ", "injection", "vial",
        "syp", "syrup", "ointment", "gel",
        "mrp", "gst"
    ]

    strict_pharm_hits = sum(k in t for k in pharmacy_keywords)

    # If there are drug signals AND no doctor/procedure → pharmacy
    if strict_pharm_hits >= 3:
        return "Pharmacy", DummyUsage()

    # handwritten-style bills
    med_indicators = ["tab", "cap", "inj", "syp"]
    med_lines = sum(1 for l in lines if any(m in l for m in med_indicators))

    if med_lines >= 3:
        return "Pharmacy", DummyUsage()

    # =====================================================
    # 5) GEMINI FALLBACK (rare)
    # =====================================================
    prompt = f"""
Classify this page strictly as:
- Final Bill
- Bill Detail
- Pharmacy

Rules:
Final Bill = totals + patient details + category-level charges
Bill Detail = many items/tests/services
Pharmacy = ONLY medicines list (tabs/caps/inj/mrp/batch numbers)

Text:
{text}
"""

    resp = GEMINI_CLASSIFIER.generate_content(prompt)
    label = resp.text.strip().lower()

    if "final" in label:
        return "Final Bill", resp.usage_metadata
    if "pharm" in label:
        return "Pharmacy", resp.usage_metadata
    return "Bill Detail", resp.usage_metadata


# -----------------------------------------------------
# ITEM EXTRACTION
# -----------------------------------------------------
def extract_items_from_text(text: str, page_type: str):

    model = GEMINI_ITEMS

    # --------------------------------------------------------
    # STRICT JSON PROMPT (prevents ANY non-JSON output)
    # --------------------------------------------------------
    prompt = f"""
Extract ONLY the bill line items from this {page_type} page.

Return STRICT JSON ONLY in the format:

[
  {{
    "item_name": "...",
    "item_rate": "...",
    "item_quantity": "...",
    "item_amount": "..."
  }}
]

RULES:
- Do NOT include explanations.
- Do NOT include comments.
- Do NOT add text before or after the JSON.
- If no items found, return [] ONLY.

TEXT:
{text}
"""

    try:
        resp = model.generate_content(prompt)
        usage = resp.usage_metadata

        raw = resp.text.strip()

        # --------------------------------------------------------
        # CLEAN OUTPUT (fix hallucinations, remove garbage)
        # --------------------------------------------------------

        # Remove backticks and language markers
        raw = raw.replace("```", "").replace("json", "").strip()

        # Extract only the JSON array
        start = raw.find("[")
        end = raw.rfind("]") + 1

        if start == -1 or end == -1:
            return [], usage

        items_json = raw[start:end]

        # --------------------------------------------------------
        # FIRST PARSE ATTEMPT
        # --------------------------------------------------------
        try:
            items = json.loads(items_json)
            return items, usage

        except:
            pass  # move to second attempt

        # --------------------------------------------------------
        # SECOND CHANCE: AUTO-FIX COMMON JSON ISSUES
        # --------------------------------------------------------
        import re

        fixed_json = items_json

        # remove trailing commas
        fixed_json = re.sub(r",\s*}", "}", fixed_json)
        fixed_json = re.sub(r",\s*]", "]", fixed_json)

        # convert single quotes to double quotes (Gemini sometimes does this)
        fixed_json = fixed_json.replace("'", '"')

        try:
            items = json.loads(fixed_json)
            return items, usage
        except Exception as e:
            print("❌ final JSON parse error:", e)
            return [], usage

    except Exception as e:
        print("❌ item extraction error:", e)
        return [], {"prompt_token_count": 0, "candidates_token_count": 0}

# -----------------------------------------------------
# MASTER FUNCTION (URL input)
# -----------------------------------------------------
def extract_document(url: str):
    logger.info("====================================================")
    logger.info(f"STARTING EXTRACTION")
    logger.info(f"Document URL: {url}")
    logger.info("====================================================")

    overall_start = time.time()

    # Download file
    file_bytes = download_from_url(url)
    logger.info("✔ File downloaded successfully")

    # Detect PDF or Image
    if is_image(url):
        logger.info("✔ Detected IMAGE file")
        pages = [file_bytes]
    else:
        logger.info("✔ Detected PDF file — converting to images...")
        t_pdf = time.time()
        pages = convert_pdf_to_images(file_bytes)
        logger.info(f"✔ PDF converted to {len(pages)} page(s) in {time.time()-t_pdf:.2f}s")

    final_output = []
    total_items = 0

    total_in = 0
    total_out = 0

    # Process pages one by one
    for page_no, img_bytes in enumerate(pages, start=1):
        logger.info(f"----------------------------------------------------")
        logger.info(f"PROCESSING PAGE {page_no}")
        logger.info(f"----------------------------------------------------")

        # 1) OCR
        t1 = time.time()
        text, u1 = gemini_ocr_extract(img_bytes)
        logger.info(f"✔ OCR completed in {time.time()-t1:.2f}s")

        total_in += u1.prompt_token_count
        total_out += u1.candidates_token_count

        # 2) Page Type classification
        t1 = time.time()
        page_type, u2 = classify_page_text(text)
        logger.info(f"✔ Page classified as: {page_type} (in {time.time()-t1:.2f}s)")

        total_in += u2.prompt_token_count
        total_out += u2.candidates_token_count

        # 3) Item extraction
        t1 = time.time()
        items, u3 = extract_items_from_text(text, page_type)
        logger.info(f"✔ Extracted {len(items)} item(s) in {time.time()-t1:.2f}s")

        total_in += u3.prompt_token_count
        total_out += u3.candidates_token_count
        total_items += len(items)

        # Store final results
        final_output.append({
            "page_no": str(page_no),
            "page_type": page_type,
            "bill_items": items
        })

    # Total execution time
    total_time = time.time() - overall_start

    logger.info("====================================================")
    logger.info("EXTRACTION COMPLETED 🎉")
    logger.info(f"Total Pages        : {len(pages)}")
    logger.info(f"Total Items        : {total_items}")
    logger.info(f"Total Tokens Used  : {total_in + total_out} (in={total_in}, out={total_out})")
    logger.info(f"Total Time Taken   : {total_time:.2f} seconds")
    logger.info("====================================================")

    return {
        "token_usage": {
            "total_tokens": total_in + total_out,
            "input_tokens": total_in,
            "output_tokens": total_out
        },
        "pagewise_line_items": final_output,
        "total_item_count": total_items
    }


    # -------------------------------
    # FINAL RESPONSE (BAJAJ FORMAT)
    # -------------------------------
    return {
        "token_usage": {
            "total_tokens": total_in + total_out,
            "input_tokens": total_in,
            "output_tokens": total_out
        },
        "pagewise_line_items": final_output,
        "total_item_count": total_items
    }
