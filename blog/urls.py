#hw/urls.py
#description: URL patterns for the new hw app

from django.urls import path 
from django.conf import settings
from . import views
from django.contrib.auth import views as auth_views    ## gives us generic views for login, log out, change password
from .views import * # our view class definition 



#all of the URLs that are part of this app


urlpatterns = [
    path(r'', views.RandomArticleView.as_view(), name='random'), ## new
    #note that r mean regular expression
    #views.home
  #  path(r'about', views.about, name = "about")
    path(r'show_all', views.ShowAllView.as_view(), name='show_all_one'), 
    path(r'article/<int:pk>', views.ArticlePageView.as_view(), name='article'), 
    #path('create_comment', views.CreateCommentView.as_view(), name='create_comment'), ### FIRST (WITHOUT PK)
    path(r'article/<int:pk>/create_comment', views.CreateCommentView.as_view(), name='create_comment'), ### NEW

    path(r'create_article', views.CreateArticleView.as_view(), name='create_article'), 

    #authentication URLs
    path('login/', auth_views.LoginView.as_view(template_name = 'blog/login.html'), name = "login2"),
    path('logout/', auth_views.LogoutView.as_view(next_page='show_all_one'), name = "logout"),
    path('register/', views.RegistrationView.as_view(), name = 'register'),
]