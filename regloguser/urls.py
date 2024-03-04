from django.contrib import admin
from django.urls import path, include

from .views import index, upload_csv, upload_csv_data, popular_products,seasonal_analysis_view, threed_shelf, visualize_shelves,upload, gemini,login,optimization
from .views import optimized_products, spring, autumn, summer,winter
urlpatterns = [
    path('', index, name='index'),
    path('upload_csv', upload_csv, name='upload_csv'),
    path('upload_csv_data', upload_csv_data, name='upload_csv_data'),
    path('popular_products', popular_products, name='popular_products'),
    path('gemini', gemini, name='gemini'),
    path('vs', visualize_shelves, name='visualize_shelves'),
    path('seasonal_products', seasonal_analysis_view, name='seasonal_analysis_view'),
    path('threed_shelf', threed_shelf, name='threed_shelf'),
    path('upload', upload, name='upload'),
    path('index', index, name='index'),
    path('login', login, name='login'),
    path('optimization',optimization,name='optimization'),
    path('optimized_products', optimized_products, name='optimized_products'),
    path('summer', summer, name='summer'),
    path('winter', winter, name='winter'),
    path('autumn', autumn, name='autumn'),
    path('spring', spring, name='spring'),

]