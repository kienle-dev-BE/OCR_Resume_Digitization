# OCR Backend Service

This project is a Django-based backend service for extracting structured information (name, phone, birth date, etc.) from images and PDF documents using Google Cloud Vision API.

## Features
- Supports both image files (JPG, PNG, etc.) and multi-page PDF documents.
- Uses Google Vision API for high-accuracy OCR.
- Extracts fields: name, phone, birth date (supports both Vietnamese and English documents).
- Unified API response structure for both images and PDFs.
- Automatically deletes uploaded files after processing.
- Secure API key management via `.env` file.

## Project Structure
```
api/
    services/
        vision_service.py   # Main OCR and extraction logic
    views.py               # API endpoint for file upload and OCR
media/                     # Uploaded files (auto-deleted after processing)
resume_ocr/                # Django project settings
.env                       # Environment variables (not committed)
```

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**
   - Python 3.8+
   - `pip install -r requirements.txt`
   - Install poppler for PDF support (see below)
3. **Set up Google Vision API**
   - Get your API key from Google Cloud Console
   - Create a `.env` file in the project root:
     ```
     GOOGLE_VISION_API_KEY=your_api_key_here
     ```
4. **Run the Django server**
   - `python manage.py runserver`

## API Usage
- **Endpoint:** `/extract-resume/`
- **Method:** `POST`
- **Payload:** Multipart form with file field (image or PDF)
- **Response:**
  ```json
  {
    "name": "Nguyễn Thị Lượn",
    "phone": null,
    "birth_date": "27.18.1.1990",
    "experience": []
  }
  ```

## Notes
- For PDF support, install poppler:
  - Windows: Download from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/), add to PATH
  - Linux: `sudo apt install poppler-utils`
- The `.env` file should **not** be committed to version control.

## License
MIT
