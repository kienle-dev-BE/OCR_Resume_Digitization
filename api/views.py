import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from .services.vision_service import ocr_reorder_and_parse

@api_view(['POST'])
def extract_resume(request):
    file = request.FILES.get('file')
    if not file:
        return Response({"error": "No file provided"}, status=400)
    
    filename = file.name
    filepath = os.path.join(settings.MEDIA_ROOT, filename)
    with open(filepath, 'wb+') as dest:
        for chunk in file.chunks():
            dest.write(chunk)
    
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']:
        # Dùng Google Vision API cho cả ảnh và PDF
        result = ocr_reorder_and_parse(filepath)
        # Xóa file sau khi xử lý xong
        try:
            os.remove(filepath)
        except Exception:
            pass
        return Response(result["parsed"])
    else:
        # Xóa file nếu không đúng định dạng
        try:
            os.remove(filepath)
        except Exception:
            pass
        return Response({"error": "Unsupported file type"}, status=400)
