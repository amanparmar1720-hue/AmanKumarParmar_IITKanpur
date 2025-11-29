import io
import re
import requests
from pdf2image import convert_from_bytes
from app.config import GEMINI_MODEL


# ------------------------------
# RULE-BASED PAGE CLASSIFIER
# ------------------------------

FINAL_BILL_KEYWORDS = [
    r"consultation", r"room rent", r"bed charges", r"visiting charges",
    r"investigation", r"procedure", r"medical consumable", r"other charges",
    r"total bill amount", r"net amount", r"grand total"
]

PHARMACY_KEYWORDS = [
    r"batch", r"exp", r"mfg", r"inj", r"injection", r"tablet",
    r"syrup", r"ml", r"mg", r"mrp"
]

def rule_based_classification(text: str):
    t = text.lower()

    if any(re.search(k, t) for k in PHARMACY_KEYWORDS):
        return "Pharmacy"
    if any(re.search(k, t) for k in FINAL_BILL_KEYWORDS):
        return "Final Bill"
    return "Bill Detail"


# ------------------------------
# GEMINI VISION OCR
# ------------------------------

def gemini_extract(image_bytes: bytes):
    """Extract using Gemini Vision."""
    try:
        result = GEMINI_MODEL.generate_content(
            ["Extract all text clearly", image_bytes]
        )
        return result.text
    except Exception as e:
        return ""


# ------------------------------
# HELPERS
# ------------------------------

def download_file(url: str):
    """Downloads image/PDF bytes."""
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise Exception("could not download document")
    return r.content


def is_pdf(url: str):
    return url.lower().endswith(".pdf") or "pdf" in url.lower()


# ------------------------------
# MAIN EXTRACTION LOGIC
# ------------------------------

def extract_from_url(url: str):
    file_bytes = download_file(url)

    results = {
        "is_success": True,
        "data": {
            "pagewise_line_items": [],
            "total_item_count": 0
        }
    }

    page_no_counter = 1
    total_items_count = 0

    # CASE 1: PDF document
    if is_pdf(url):

        # convert each PDF page to image
        try:
            pages = convert_from_bytes(file_bytes)
        except:
            return {"is_success": False, "error": "cannot open pdf"}

        for page_image in pages:

            buf = io.BytesIO()
            page_image.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            page_text = gemini_extract(img_bytes)
            page_type = rule_based_classification(page_text)

            # Each bullet/line item Gemini extracts is one item
            lines = [l.strip() for l in page_text.split("\n") if l.strip()]
            line_items = []
            for l in lines:
                line_items.append({
                    "item_name": l,
                    "item_rate": 0.0,
                    "item_quantity": 0.0,
                    "item_amount": 0.0
                })

            results["data"]["pagewise_line_items"].append({
                "page_no": str(page_no_counter),
                "page_type": page_type,
                "bill_items": line_items
            })

            total_items_count += len(line_items)
            page_no_counter += 1

    else:
        # CASE 2: IMAGE document
        page_text = gemini_extract(file_bytes)
        page_type = rule_based_classification(page_text)

        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        line_items = []
        for l in lines:
            line_items.append({
                "item_name": l,
                "item_rate": 0.0,
                "item_quantity": 0.0,
                "item_amount": 0.0
            })

        results["data"]["pagewise_line_items"].append({
            "page_no": "1",
            "page_type": page_type,
            "bill_items": line_items
        })

        total_items_count += len(line_items)

    results["data"]["total_item_count"] = total_items_count
    return results
