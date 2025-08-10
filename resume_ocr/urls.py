
from django.contrib import admin
from django.urls import path
from api.views import extract_resume

urlpatterns = [
    path('admin/', admin.site.urls),
    path('extract-resume/', extract_resume),
]
