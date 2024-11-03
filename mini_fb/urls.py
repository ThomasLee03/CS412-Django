#mini_fb/urls.py
# author: Thomas Lee (tlee03@bu.edu), 10/5/2024
#description: URL patterns for the new mini_fb app

from django.urls import path 
from django.conf import settings
from . import views

from django.contrib.auth import views as auth_views    ## gives us generic views for login, log out, change password
from .views import * # our view class definition 


#all of the URLs that are part of this app

urlpatterns = [
    path(r'', views.ShowAllProfilesView.as_view(), name = "show_all"),
    path('profile/<int:pk>', views.ShowProfilePageView.as_view(), name='profile'),
    path('create_profile', views.CreateProfileView.as_view(), name='create_profile'), ### NEW
    path('status/create_status', views.CreateStatusMessageView.as_view(), name='create_status'), ### NEW
    path('profile/update', views.UpdateProfileView.as_view(), name='update_profile'), ### NEW
    path('status/<int:pk>/delete', views.DeleteStatusMessageView.as_view(), name='delete_status'), ### NEW
    path('status/<int:pk>/update', views.UpdateStatusMessageView.as_view(), name='update_status'), ### NEW
    path('profile/add_friend/<int:other_pk>', views.CreateFriendView.as_view(), name='add_friend'), ### NEW
    path('profile/friend_suggestions', views.ShowFriendSuggestionsView.as_view(), name='friend_suggestions'), ### NEW
    path('profile/news_feed', views.ShowNewsFeedView.as_view(), name='news_feed'),
    #note that r mean regular expression
    path('login/', auth_views.LoginView.as_view(template_name = 'mini_fb/login.html'), name = "login"),
    path('logout/', auth_views.LogoutView.as_view(template_name='mini_fb/logged_out.html'), name = "logout"),
    #path('logout/', views.custom_logout, name='logout'),
]