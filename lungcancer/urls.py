from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('lungcancer/upload/<str:filename>', views.serve_photo),
]