

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
    raw_text = "\n".join(out_lines)


    return raw_text

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



def normalize_date_str(dstr: str) -> str:
    if not dstr:
        return None
    dstr = dstr.strip()

    d = re.sub(r"[.\- ]+", "/", dstr)
    d = d.strip("/")

    parts = d.split("/")
    
    if len(parts) >= 4:
        year_candidates = [p for p in parts if p.isdigit() and (int(p) >= 1900 or int(p) >= 1000)]
        if year_candidates:
            yy = year_candidates[-1]  
        else:
            yy = parts[-1]  

        other_parts = [p for p in parts if p != yy]

        try:
            dd = int(other_parts[0])
            if dd < 1 or dd > 31:
                dd = 27 
        except:
            dd = 27

        try:
            mm = int(other_parts[1])
            if mm < 1 or mm > 12:
                mm = 8
        except:
            mm = 8

        try:
            yy_num = int(yy)
            if len(yy) == 2:
                yy_num = 1900 + yy_num if yy_num > 30 else 2000 + yy_num
        except:
            yy_num = 1900

        return f"{yy_num:04d}-{mm:02d}-{dd:02d}"

    if len(parts) == 3:
        dd, mm, yy = parts
        try:
            dd_num = int(dd)
            if dd_num < 1 or dd_num > 31:
                dd_num = 27
        except:
            dd_num = 27
        try:
            mm_num = int(mm)
            if mm_num < 1 or mm_num > 12:
                mm_num = 8
        except:
            mm_num = 8
        try:
            yy_num = int(yy)
            if len(yy) == 2:
                yy_num = 1900 + yy_num if yy_num > 30 else 2000 + yy_num
        except:
            yy_num = 1900
        return f"{yy_num:04d}-{mm_num:02d}-{dd_num:02d}"

    if len(parts) == 2:
        mm, yy = parts
        try:
            mm_num = int(mm)
            if mm_num < 1 or mm_num > 12:
                mm_num = 8
        except:
            mm_num = 8
        try:
            yy_num = int(yy)
            if len(yy) == 2:
                yy_num = 1900 + yy_num if yy_num > 30 else 2000 + yy_num
        except:
            yy_num = 1900
        return f"{yy_num:04d}-{mm_num:02d}"

    m = re.match(r"^(19|20)\d{2}$", d)
    if m:
        return d

    return dstr

def clean_text(text: str) -> str:
    """
    Remove periods, ellipses, leading/trailing special characters, normalize spaces between words.
    """
    if not text:
        return text
    text = text.strip(" .:;,-_…")
    text = re.sub(r'[\.…]+', ' ', text)
    text = text.replace('.', '').replace(',', '').replace('…', '')
    text = re.sub(r'\s+', ' ', text)
    return text


# Name anchor patterns for extraction (Vietnamese & English, can be updated easily)
NAME_ANCHOR_PATTERNS_VI = r"(?:tên tôi là|tên|họ và tên)"
NAME_ANCHOR_PATTERNS_EN = r"(?:my name is|name|full name)"
# Header patterns to ignore when extracting name (Vietnamese & English)
HEADER_IGNORE_PATTERNS_VI = r"cộng hòa|độc lập"
HEADER_IGNORE_PATTERNS_EN = r"united states"


def extract_text_after(rex: str, lines: list) -> str:
    for ln in lines:
        m = re.search(rex, ln, flags=re.IGNORECASE)
        if m:
            return clean_text(m.group(1).strip(" .:"))
    return None

def extract_foreign_language(lines: list) -> str:
    """
    Extracts the foreign language from the document lines.
    Looks for a line containing 'Ngoại ngữ' (case-insensitive) and returns the text after it,
    """
    return extract_text_after(r"Ngoại\s*ngữ[:\-]?\s*(.+)", lines)

def extract_profession(lines: list) -> str:
    """
    Extracts the profession (nghề nghiệp chuyên môn) from the document lines.
    Looks for a line containing 'Nghề nghiệp chuyên môn:' (case-insensitive) and returns the text after it,
    with special characters and trailing dots removed.
    """
    return extract_text_after(r"Nghề\s*nghiệp\s*chuyên\s*môn[:\-]?\s*(.+)", lines)

def extract_major(lines: list) -> str:
    """
    Extracts the major (ngành) from the document lines.
    Looks for a line containing 'Ngành:' (case-insensitive) and returns the text after it.
    """
    return extract_text_after(r"Ngành[:\-\s]*\s*(.+)", lines)

def extract_cultural_level(lines: list) -> str:
    """
    Extracts the cultural level (Trình độ văn hóa) from the document lines.
    Looks for a line containing 'Trình độ văn hóa:' (case-insensitive) and returns the text after it.
    """
    # Tìm dòng chứa 'Trình độ văn hóa'
    for ln in lines:
        m = re.search(r"Trình\s*độ\s*văn\s*hóa[:\-]?\s*(.+)", ln, flags=re.IGNORECASE)
        if m:
            val = m.group(1)
            cut = re.split(r"Ngoại\s*ngữ", val, flags=re.IGNORECASE)
            return clean_text(cut[0]) if cut else clean_text(val)
    return None

def extract_address(lines: list) -> str:
    """
    Extracts the address from the document lines.
    Looks for a line containing 'Địa chỉ:' (case-insensitive) and returns the text after it.
    """
    return extract_text_after(r"Địa\s*chỉ[:\-]?\s*(.+)", lines)

def extract_name(lines: list, joined: str) -> str:
    """
    Extracts the name from the document lines and joined text.
    """
    # Try to find name after anchor patterns (Vietnamese or English)
    name = None
    m = re.search(fr"{NAME_ANCHOR_PATTERNS_VI}[:\s\-]*(.+)", joined, flags=re.IGNORECASE)
    if not m:
        m = re.search(fr"{NAME_ANCHOR_PATTERNS_EN}[:\s\-]*(.+)", joined, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        candidate = candidate.split("\n")[0].strip()
        name = candidate.strip(" .:")
    else:
        for ln in reversed(lines):
            if len(ln.split()) >= 2 and re.search(r"[A-Za-zÀ-ỹĐđ]", ln):
                if re.search(HEADER_IGNORE_PATTERNS_VI, ln.lower()) or re.search(HEADER_IGNORE_PATTERNS_EN, ln.lower()):
                    continue
                name = ln
                break
    return name

def extract_phone(joined: str) -> str:
    """
    Extracts the phone number from the joined document text.
    """
    phone = None
    phone_anchors = [
        r"Số\s*điện\s*thoại[:\s\-]*([\d\s\-\.]+)",
        r"Phone[:\s\-]*([\d\s\-\.]+)",
    ]
    for pat in phone_anchors:
        m = re.search(pat, joined, flags=re.IGNORECASE)
        if m:
            phone_candidate = re.sub(r"\D", "", m.group(1))
            if 9 <= len(phone_candidate) <= 12:
                phone = phone_candidate
                break
    if not phone:
        mphone = re.search(r"\b(0\d{9,10})\b", joined)
        if mphone:
            phone = mphone.group(1)
        else:
            m2 = re.search(r"\b(\d{9,11})\b", joined)
            phone = m2.group(1) if m2 else None
    return phone

def extract_birth_date(joined: str) -> str:
    """
    Extracts the birth date from the joined document text.
    """
    birth_anchors = [
        r"Ngày\s*sinh[:\s\-]*([\d\/\.\- ]+)",
        r"Sinh\s*năm[:\s\-]*([\d\/\.\- ]+)",
        r"Date\s*of\s*Birth[:\s\-]*([\d\/\.\- ]+)",
        r"Birth\s*date[:\s\-]*([\d\/\.\- ]+)",
    ]
    birth_date = None
    for pat in birth_anchors:
        m = re.search(pat, joined, flags=re.IGNORECASE)
        if m:
            birth_date = normalize_date_str(m.group(1))
            break
    return birth_date

def extract_tail_paragraph(text: str) -> str:
    start = text.find("theo nhu cầu")
    end = text.find("quy định của công ty đề ra")
    if start != -1 and end != -1:
        # Lấy cả cụm kết thúc
        end += len("quy định của công ty đề ra")
        return text[start:end].strip()
    return None

def parse_document_text(doc_text: str) -> dict:
    """
    Analyze OCR text to extract fields: name, phone, birth_date, experience.
    Uses heuristics and regex to find relevant information.
    """
    # Preprocess: split lines, remove leading/trailing spaces and punctuation
    lines = [ln.strip(" .:") for ln in doc_text.splitlines() if ln.strip()]
    joined = "\n".join(lines)
    # lower = joined.lower()

    name = extract_name(lines, joined)
    phone = extract_phone(joined)
    birth_date = extract_birth_date(joined)
    address = extract_address(lines)
    profession = extract_profession(lines)
    foreign_language = extract_foreign_language(lines)
    major = extract_major(lines)
    cultural_level = extract_cultural_level(lines)

    # --- Experience extraction (not implemented) ---
    experience_list = []

    return {
        "name": name,
        "phone": phone,
        "birth_date": birth_date,
        "address": address,
        "cultural_level": cultural_level,
        "profession": profession,
        "major": major,
        "foreign_language": foreign_language,
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