"""
ASGI config for resume_ocr project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
import dotenv
dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'resume_ocr.settings')

application = get_asgi_application()
