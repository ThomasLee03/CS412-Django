#file: urls.py
# author: Thomas Lee (tlee03@bu.edu), 11/10/2024
from django.urls import path
from . import views 
urlpatterns = [
    # map the URL (empty string) to the view
    path(r'', views.ResultsListView.as_view(), name='home'),
    path(r'results', views.ResultsListView.as_view(), name='results_list'),
    path(r'voter/<int:pk>', views.ResultDetailView.as_view(), name='result_detail'),
    path(r'graphs', views.ResultGraphsView.as_view(), name='result_graphs'),
    
]
