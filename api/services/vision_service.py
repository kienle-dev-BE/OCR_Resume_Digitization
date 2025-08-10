from pdf2image import convert_from_path


# vision_service.py
# OCR service using Google Cloud Vision API for text recognition and information extraction from images/documents.

import base64
import requests
import re
import json
import statistics
from typing import List, Dict
import os

API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "") 

def call_vision_api(file_path: str) -> dict:
    """
    Send an image (or PDF) file to Google Vision API and receive OCR analysis results.
    If input is PDF, convert each page to image and OCR each page, then join results.
    Returns the JSON result dict from the API (for images) or merged text for PDFs.
    """
    ext = os.path.splitext(file_path)[1].lower()
    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"

    if ext == '.pdf':
        # Convert PDF to images (one per page)
        images = convert_from_path(file_path)
        merged_text = []
        merged_text_annotations = []
        merged_pages = []
        for img in images:
            from io import BytesIO
            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG')
            b64 = base64.b64encode(img_bytes.getvalue()).decode()
            payload = {
                "requests": [
                    {
                        "image": {"content": b64},
                        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
                    }
                ]
            }
            r = requests.post(url, json=payload)
            r.raise_for_status()
            resp = r.json()
            response = resp.get("responses", [{}])[0]
            # Merge fullTextAnnotation text
            full = response.get("fullTextAnnotation")
            if full and 'text' in full:
                merged_text.append(full['text'])
                if 'pages' in full:
                    merged_pages.extend(full['pages'])
            else:
                ta = response.get("textAnnotations", [])
                raw = ta[0].get("description") if ta else ""
                merged_text.append(raw)
            # Merge textAnnotations
            ta = response.get("textAnnotations", [])
            if ta:
                merged_text_annotations.extend(ta)
        # Build a response structure similar to Vision API
        merged_full = {
            "text": "\n".join(merged_text),
            "pages": merged_pages,
        }
        return {"responses": [{"fullTextAnnotation": merged_full, "textAnnotations": merged_text_annotations}]}
    else:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "requests": [
                {
                    "image": {"content": b64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
                }
            ]
        }
        r = requests.post(url, json=payload)
        r.raise_for_status()
        return r.json()

def _safe_get(v, key, default=0):
    """
    Get the value of key from a dict, return default if not present.
    """
    return v.get(key, default) if isinstance(v, dict) else default

def extract_word_boxes(full_text_annotation: dict) -> List[Dict]:
    """
    Extract words and their bounding boxes from Vision API's fullTextAnnotation result.
    Returns a list of dicts: {text, minx, miny, maxx, maxy, cx, cy, h, page}
    """
    words = []
    pages = full_text_annotation.get("pages", [])
    for p_idx, page in enumerate(pages):
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    symbols = word.get("symbols", [])
                    word_text = "".join([s.get("text", "") for s in symbols])
                    verts = word.get("boundingBox", {}).get("vertices", [])
                    xs = [(_safe_get(v, "x", 0)) for v in verts]
                    ys = [(_safe_get(v, "y", 0)) for v in verts]
                    if not xs: xs = [0, 0, 0, 0]
                    if not ys: ys = [0, 0, 0, 0]
                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)
                    cx = (minx + maxx) / 2.0
                    cy = (miny + maxy) / 2.0
                    h = maxy - miny
                    words.append({
                        "text": word_text,
                        "minx": minx, "maxx": maxx,
                        "miny": miny, "maxy": maxy,
                        "cx": cx, "cy": cy,
                        "h": h if h>0 else 10,
                        "page": p_idx
                    })
    return words

def group_words_to_lines(words: List[Dict]) -> List[List[Dict]]:
    """
    Group words into lines based on y position (cy) and page.
    Returns a list of lines, each line is a list of words.
    """
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w["page"], w["cy"], w["minx"]))
    heights = [w["h"] for w in words_sorted if w["h"] > 0]
    median_h = statistics.median(heights) if heights else 12
    line_threshold = max(10, median_h * 0.8)
    lines = []
    current_line = [words_sorted[0]]
    def current_line_center(line):
        return statistics.mean([w["cy"] for w in line])
    for w in words_sorted[1:]:
        if w["page"] != current_line[-1]["page"] or abs(w["cy"] - current_line_center(current_line)) > line_threshold:
            lines.append(current_line)
            current_line = [w]
        else:
            current_line.append(w)
    lines.append(current_line)
    for i in range(len(lines)):
        lines[i] = sorted(lines[i], key=lambda x: x["minx"])
    return lines

def reconstruct_text_from_lines(lines: List[List[Dict]]) -> str:
    """
    Join words into lines (separated by spaces), then join lines into the final text.
    """
    out_lines = []
    for line in lines:
        words_text = [w["text"] for w in line if w["text"].strip() != ""]
        if words_text:
            out_lines.append(" ".join(words_text))
    return "\n".join(out_lines)

def find_dates_candidates(text: str):
        """
        Find date-like strings in the text.
        Returns a tuple (full_date, mm/yyyy, year_only)
        - full: list of Match objects for dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy, etc.
            Each match: group(0) is the full date string, group(1) is day, group(2) is month, group(3) is year.
        - mm_yyyy: list of Match objects for mm/yyyy, mm-yyyy, mm.yyyy, etc.
            Each match: group(0) is the full string, group(1) is month, group(2) is year.
        - year_only: list of Match objects for 4-digit years (1900-2099).
            Each match: group(0) is the year string.
        """
        # Example: full = [<re.Match object; span=(10, 20), match='27/01/1990'>, ...]
        full = list(re.finditer(r"\b(\d{1,2})[\/\.\- ]+(\d{1,2})[\/\.\- ]+(\d{2,4})\b", text))
        # Example: mm_yyyy = [<re.Match object; span=(10, 17), match='01/1990'>, ...]
        mm_yyyy = list(re.finditer(r"\b(\d{1,2})[\/\.\- ]+(\d{4})\b", text))
        # Example: year_only = [<re.Match object; span=(10, 14), match='1990'>, ...]
        year_only = list(re.finditer(r"\b(19|20)\d{2}\b", text))
        return full, mm_yyyy, year_only

def pick_date_near_anchor(text: str, anchor: str = "sinh") -> str:
    """
    Find the date string closest to the anchor keyword (e.g., 'sinh', 'ngày sinh').
    If not found, return the first candidate.
    """
    lower = text.lower()
    anchor_pos = lower.find(anchor)
    full, mm_yyyy, year_only = find_dates_candidates(text)
    candidates = []
    for m in full:
        candidates.append((m.start(), m.group(0)))
    for m in mm_yyyy:
        candidates.append((m.start(), m.group(0)))
    for m in year_only:
        candidates.append((m.start(), m.group(0)))
    if not candidates:
        return None
    if anchor_pos != -1:
        candidates.sort(key=lambda x: abs(x[0] - anchor_pos))
        return candidates[0][1]
    else:
        if full:
            return full[0].group(0)
        if mm_yyyy:
            return mm_yyyy[0].group(0)
        return year_only[0].group(0)

def normalize_date_str(dstr: str) -> str:
    """
    Normalize a date string to the format YYYY-MM-DD, YYYY-MM, or YYYY.
    """
    if not dstr:
        return None
    dstr = dstr.strip()
    d = re.sub(r"[.\- ]+", "/", dstr)
    parts = d.split("/")
    if len(parts) == 3:
        dd = parts[0].zfill(2)
        mm = parts[1].zfill(2)
        yy = parts[2]
        if len(yy) == 2:
            yy_num = int(yy)
            yy = f"19{yy}" if yy_num > 30 else f"20{yy}"
        return f"{yy}-{mm}-{dd}"
    if len(parts) == 2:
        mm = parts[0].zfill(2)
        yy = parts[1]
        if len(yy) == 2:
            yy_num = int(yy); yy = f"19{yy}" if yy_num > 30 else f"20{yy}"
        return f"{yy}-{mm}"
    m = re.match(r"^(19|20)\d{2}$", d)
    if m:
        return d
    return dstr



# Name anchor patterns for extraction (Vietnamese & English, can be updated easily)
NAME_ANCHOR_PATTERNS_VI = r"(?:tên tôi là|tên|họ và tên)"
NAME_ANCHOR_PATTERNS_EN = r"(?:my name is|name|full name)"
# Header patterns to ignore when extracting name (Vietnamese & English)
HEADER_IGNORE_PATTERNS_VI = r"cộng hòa|độc lập"
HEADER_IGNORE_PATTERNS_EN = r"united states"

def parse_document_text(doc_text: str) -> dict:
    """
    Analyze OCR text to extract fields: name, phone, birth_date, experience.
    Uses heuristics and regex to find relevant information.
    """
    # Preprocess: split lines, remove leading/trailing spaces and punctuation
    lines = [ln.strip(" .:") for ln in doc_text.splitlines() if ln.strip()]
    joined = "\n".join(lines)
    lower = joined.lower()

    # --- Name extraction ---
    # Try to find name after anchor patterns (Vietnamese or English)
    name = None
    # Check Vietnamese anchors first
    m = re.search(fr"{NAME_ANCHOR_PATTERNS_VI}[:\s\-]*(.+)", joined, flags=re.IGNORECASE)
    if not m:
        # If not found, check English anchors
        m = re.search(fr"{NAME_ANCHOR_PATTERNS_EN}[:\s\-]*(.+)", joined, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        # Only take the first line after anchor (avoid capturing too much)
        candidate = candidate.split("\n")[0].strip()
        name = candidate.strip(" .:")
    else:
        # Fallback: use the last line with at least 2 words and letters (often signature)
        for ln in reversed(lines):
            # re.search(r"[A-Za-zÀ-ỹĐđ]", ln): checks if line contains at least one letter (Vietnamese included)
            if len(ln.split()) >= 2 and re.search(r"[A-Za-zÀ-ỹĐđ]", ln):
                # Ignore header lines like 'CỘNG HÒA...' or English headers
                if re.search(HEADER_IGNORE_PATTERNS_VI, ln.lower()) or re.search(HEADER_IGNORE_PATTERNS_EN, ln.lower()):
                    continue
                name = ln
                break


    # --- Phone extraction ---
    # Try to find phone number after Vietnamese or English anchors
    phone = None
    phone_anchors = [
        r"Số\s*điện\s*thoại[:\s\-]*([\d\s\-\.]+)",  # Vietnamese
        r"Phone[:\s\-]*([\d\s\-\.]+)",               # English
    ]
    for pat in phone_anchors:
        m = re.search(pat, joined, flags=re.IGNORECASE)
        if m:
            # Remove all non-digit characters
            phone_candidate = re.sub(r"\D", "", m.group(1))
            if 9 <= len(phone_candidate) <= 12:
                phone = phone_candidate
                break
    if not phone:
        # Fallback: find 10 or 11 digit phone number starting with 0
        mphone = re.search(r"\b(0\d{9,10})\b", joined)
        if mphone:
            phone = mphone.group(1)
        else:
            # Fallback: any 9-11 digit chunk
            m2 = re.search(r"\b(\d{9,11})\b", joined)
            phone = m2.group(1) if m2 else None

    # --- Birth date extraction ---
    # Try to find date after Vietnamese or English anchors
    birth_anchors = [
        r"Ngày\s*sinh[:\s\-]*([\d\/\.\- ]+)",   # Vietnamese
        r"Sinh\s*năm[:\s\-]*([\d\/\.\- ]+)",   # Vietnamese
        r"Date\s*of\s*Birth[:\s\-]*([\d\/\.\- ]+)", # English
        r"Birth\s*date[:\s\-]*([\d\/\.\- ]+)",      # English
    ]
    birth_date = None
    for pat in birth_anchors:
        m = re.search(pat, joined, flags=re.IGNORECASE)
        if m:
            birth_date = normalize_date_str(m.group(1))
            break
    if not birth_date:
        # Fallback: find date string closest to 'sinh' or 'birth'
        raw_date = pick_date_near_anchor(joined, anchor="sinh")
        if not raw_date:
            raw_date = pick_date_near_anchor(joined, anchor="birth")
        birth_date = normalize_date_str(raw_date) if raw_date else None

    # --- Experience extraction (not implemented) ---
    experience_list = []

    return {
        "name": name,
        "phone": phone,
        "birth_date": birth_date,
        "experience": experience_list
    }

def ocr_reorder_and_parse(file_path: str) -> dict:
    """
    Main function: OCR an image/document file, reconstruct text, and extract key information.
    Returns a dict with document_text (full text) and parsed (extracted fields).
    """
    resp = call_vision_api(file_path)
    full = resp.get("responses", [{}])[0].get("fullTextAnnotation")
    if not full:
        ta = resp.get("responses", [{}])[0].get("textAnnotations", [])
        raw = ta[0].get("description") if ta else ""
        document_text = raw
    else:
        words = extract_word_boxes(full)
        lines = group_words_to_lines(words)
        document_text = reconstruct_text_from_lines(lines)
    parsed = parse_document_text(document_text)
    return {"document_text": document_text, "parsed": parsed}
