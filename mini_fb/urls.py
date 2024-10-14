#mini_fb/urls.py
# author: Thomas Lee (tlee03@bu.edu), 10/5/2024
#description: URL patterns for the new mini_fb app

from django.urls import path 
from django.conf import settings
from . import views




#all of the URLs that are part of this app

urlpatterns = [
    path(r'', views.ShowAllProfilesView.as_view(), name = "show_all"),
    path('profile/<int:pk>', views.ShowProfilePageView.as_view(), name='profile'),
    path('create_profile', views.CreateProfileView.as_view(), name='create_profile'), ### NEW
    path('profile/<int:pk>/create_status', views.CreateStatusMessageView.as_view(), name='create_status'), ### NEW
    #note that r mean regular expression
]