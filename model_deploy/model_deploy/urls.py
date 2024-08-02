"""
URL configuration for model_deploy project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
import model_api.views as views
from django.conf import settings
from django.conf.urls.static import static
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
   openapi.Info(
      title="Model BI-Presence",
      default_version="v1",
    #   description="API for Face Recognition Employee in Bank Indonesia",
    #   terms_of_service="https://dummy.com/terms/",
    #   contact=openapi.Contact(email="contact@dummy.com"),
    #   license=openapi.License(name="dummy"),
   ),
   public=True,
)

urlpatterns = [
    path('',views.index, name='index'),
    path('admin/', admin.site.urls),
    path('train-model', views.TrainModelView.as_view(), name='train_model'),
    path('prediction', views.PredFacenetView.as_view(), name = 'prediction'),
    # path('detect-faces', views.detect_faces_camera, name='detect_faces_camera'),
    path('classify-face', views.classify_face, name='classify-face'),
    path('swagger', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
