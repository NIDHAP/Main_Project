from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

# urlpatterns = [
#     path('upload/', views.upload, name='upload'),
# ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

from django.urls import path
from myapp.views import main_page

urlpatterns = [  
    path('', main_page, name='main'),
    path('upload/', views.upload, name='upload'),
    path('verify/', views.upload_zerowatermark, name='verify'),
    path('result/', views.result_logo, name='result'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)