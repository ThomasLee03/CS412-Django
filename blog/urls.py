#hw/urls.py
#description: URL patterns for the new hw app

from django.urls import path 
from django.conf import settings
from . import views




#all of the URLs that are part of this app

#WHAT IS THE PURPOSE OF R??

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
    
]