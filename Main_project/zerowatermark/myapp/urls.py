from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from myapp.views import generate_zero_watermark

urlpatterns = [
    path('upload/', views.upload, name='upload'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
