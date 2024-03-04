from django.contrib import admin
from django.urls import path, include
from .views import reviews
urlpatterns = [
    path('', reviews, name='reviews'),
]