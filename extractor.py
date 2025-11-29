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
# PDF → IMAGES
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


GLOBAL_PAGE_RESULTS = []


def classify_page_text(text: str, pdf_pages_processed: int, total_pages: int):
    t = text.lower()
    lines = [l.strip() for l in t.split("\n") if l.strip()]

    # ================================================================
    # 0) HARD PHARMACY DETECTION — RETAIL DRUG BILL COLUMN HEADERS
    # ================================================================
    pharmacy_headers = [
        "batch no", "batch", "exp", "expiry", "exp dt",
        "mfr", "manufacturer", "sch", "rs", "ps",
        "name of drug", "particulars"
    ]

    header_hits = sum(1 for h in pharmacy_headers if h in t)

    # If ≥4 retail headers → guaranteed PHARMACY
    if header_hits >= 4:
        GLOBAL_PAGE_RESULTS.append("Pharmacy")
        return "Pharmacy", DummyUsage()

    # ================================================================
    # 1) STRONG NON-PHARMACY OVERRIDE
    # (These metadata appear ONLY in hospital bill details / final bill)
    # ================================================================
    non_pharmacy_forced = [
        "admission date", "discharge date", "patient name",
        "patient regn", "bill date", "bill no", "ip no",
        "uhid", "ward", "room rent", "bed charges"
    ]

    if any(x in t for x in non_pharmacy_forced):
        GLOBAL_PAGE_RESULTS.append("Bill Detail")
        return "Bill Detail", DummyUsage()

    # ================================================================
    # 2) FINAL BILL DETECTION
    # ================================================================
    final_bill_signals = [
        "admission date", "discharge date",
        "patient name", "bill no", "uhid",
        "net amount payable", "total bill amount",
        "grand total", "amount received"
    ]

    final_bill_categories = [
        "consultation", "room rent", "bed charges",
        "investigation", "laboratory", "radiology",
        "pathology", "pharmacy", "procedures",
        "surgery", "medical consumable"
    ]

    has_signals = sum(k in t for k in final_bill_signals) >= 2
    has_cat = sum(k in t for k in final_bill_categories) >= 1

    if has_signals and has_cat:
        GLOBAL_PAGE_RESULTS.append("Final Bill")
        return "Final Bill", DummyUsage()

    # ================================================================
    # 3) BILL DETAIL DETECTION
    # ================================================================
    import re
    numeric_row_pattern = r"\b(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)"
    numeric_rows = len(re.findall(numeric_row_pattern, t))

    if numeric_rows >= 5:
        GLOBAL_PAGE_RESULTS.append("Bill Detail")
        return "Bill Detail", DummyUsage()

    bill_detail_keywords = [
        "x-ray", "xray", "usg", "ultrasound", "echo", "2d echo",
        "mri", "ct", "ecg", "cbc", "blood", "culture", "serum",
        "crp", "lft", "kft", "widal", "hba1c",
        "radiology", "pathology", "laboratory"
    ]

    if any(k in t for k in bill_detail_keywords):
        GLOBAL_PAGE_RESULTS.append("Bill Detail")
        return "Bill Detail", DummyUsage()

    # ================================================================
    # 4) PHARMACY (Soft Logic — Only if no contradiction)
    # ================================================================
    pharmacy_keywords = [
        "pharmacy", "drug", "mrp", "cgst", "sgst", "igst",
        "tablet", "tab", "capsule", "cap", "syrup", "syp",
        "injection", "inj", "vial", "ointment", "gel"
    ]

    pharm_hits = sum(k in t for k in pharmacy_keywords)

    non_pharmacy_markers = [
        "consultation", "procedure", "surgery", "doctor",
        "dr.", "dietician", "admission", "discharge",
        "room rent", "bed charges"
    ]

    non_pharm_hits = sum(k in t for k in non_pharmacy_markers)

    # Soft-pharmacy allowed only if:
    # - at least 3 pharmacy hints
    # - NO non-pharmacy hints
    if pharm_hits >= 3 and non_pharm_hits == 0:
        GLOBAL_PAGE_RESULTS.append("Pharmacy")
        return "Pharmacy", DummyUsage()

    # ================================================================
    # 5) DEFAULT → BILL DETAIL (Safest)
    # ================================================================
    GLOBAL_PAGE_RESULTS.append("Bill Detail")
    return "Bill Detail", DummyUsage()



# ===================================================================
# FINAL GLOBAL FIX (continuation pages)
# ===================================================================
def fix_global_page_classification(pagewise_results):

    # If any page is Bill Detail → pharmacy must be validated
    if any(pg["page_type"] == "Bill Detail" for pg in pagewise_results):

        for pg in pagewise_results:
            txt = pg.get("ocr_text", "").lower()

            # Page marked Pharmacy but missing pharmacy structure = wrong
            missing_pharmacy_structure = (
                not any(x in txt for x in ["batch", "exp", "rs", "ps", "mfr"])
            )

            if pg["page_type"] == "Pharmacy" and missing_pharmacy_structure:
                pg["page_type"] = "Bill Detail"

    return pagewise_results


# -----------------------------------------------------
# ITEM EXTRACTION
# -----------------------------------------------------
def extract_items_from_text(text: str, page_type: str):

    model = GEMINI_ITEMS

    prompt = f"""
Extract ONLY the bill line items from this {page_type} page.

Return STRICT JSON ONLY:

[
  {{
    "item_name": "...",
    "item_rate": "...",
    "item_quantity": "...",
    "item_amount": "..."
  }}
]

If no items exist, return [] ONLY.

TEXT:
{text}
"""

    try:
        resp = model.generate_content(prompt)
        usage = resp.usage_metadata

        raw = resp.text.strip()
        raw = raw.replace("```", "").replace("json", "").strip()

        start = raw.find("[")
        end = raw.rfind("]") + 1

        if start == -1 or end == -1:
            return [], usage

        items_json = raw[start:end]

        try:
            return json.loads(items_json), usage
        except:
            pass

        # Fix common issues
        import re
        fixed = re.sub(r",\s*}", "}", items_json)
        fixed = re.sub(r",\s*]", "]", fixed)
        fixed = fixed.replace("'", '"')

        return json.loads(fixed), usage

    except Exception as e:
        print("❌ item extraction error:", e)
        return [], {"prompt_token_count": 0, "candidates_token_count": 0}


# -----------------------------------------------------
# MASTER EXTRACTION FUNCTION
# -----------------------------------------------------
def extract_document(url: str):
    logger.info("====================================================")
    logger.info(f"STARTING EXTRACTION")
    logger.info(f"Document URL: {url}")
    logger.info("====================================================")

    overall_start = time.time()

    file_bytes = download_from_url(url)
    logger.info("✔ File downloaded successfully")

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

    GLOBAL_PAGE_RESULTS.clear()

    # ---------------- PROCESS EACH PAGE ----------------
    for page_no, img_bytes in enumerate(pages, start=1):

        logger.info("----------------------------------------------------")
        logger.info(f"PROCESSING PAGE {page_no}")
        logger.info("----------------------------------------------------")

        # OCR
        t1 = time.time()
        text, u1 = gemini_ocr_extract(img_bytes)
        logger.info(f"✔ OCR completed in {time.time()-t1:.2f}s")

        total_in += u1.prompt_token_count
        total_out += u1.candidates_token_count

        # CLASSIFICATION
        t1 = time.time()
        page_type, u2 = classify_page_text(
            text,
            pdf_pages_processed=page_no - 1,
            total_pages=len(pages)
        )
        logger.info(f"✔ Page classified as: {page_type} (in {time.time()-t1:.2f}s)")

        total_in += u2.prompt_token_count
        total_out += u2.candidates_token_count

        # ITEM EXTRACTION
        t1 = time.time()
        items, u3 = extract_items_from_text(text, page_type)
        logger.info(f"✔ Extracted {len(items)} item(s) in {time.time()-t1:.2f}s")

        total_in += u3.prompt_token_count
        total_out += u3.candidates_token_count
        total_items += len(items)

        final_output.append({
            "page_no": str(page_no),
            "page_type": page_type,
            "ocr_text": text,     # needed for global override
            "bill_items": items
        })

    # ---------------- APPLY GLOBAL FIX ----------------
    final_output = fix_global_page_classification(final_output)

    # Remove OCR text from final response
    for pg in final_output:
        pg.pop("ocr_text", None)

    # LOG
    total_time = time.time() - overall_start

    logger.info("====================================================")
    logger.info("EXTRACTION COMPLETED 🎉")
    logger.info(f"Total Pages        : {len(pages)}")
    logger.info(f"Total Items        : {total_items}")
    logger.info(f"Total Tokens Used  : {total_in + total_out} (in={total_in}, out={total_out})")
    logger.info(f"Total Time Taken   : {total_time:.2f} seconds")
    logger.info("====================================================")

    # BAJAJ FORMAT
    return {
        "is_success": True,
        "token_usage": {
            "total_tokens": total_in + total_out,
            "input_tokens": total_in,
            "output_tokens": total_out
        },
        "data": {
            "pagewise_line_items": final_output,
            "total_item_count": total_items
        }
    }
