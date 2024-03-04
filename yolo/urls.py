from django.contrib import admin
from django.urls import path, include
from .views import video_capture, video_feed

urlpatterns = [
    path('video_feed/', video_feed, name='video_feed'),
    path('video_capture/', video_capture, name='video_capture'),
]