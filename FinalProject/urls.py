from django.urls import path
from django.contrib.auth.views import LogoutView
from django.views.generic import TemplateView
from . import views


urlpatterns = [
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path(r'', views.DashboardView.as_view(), name='dashboard'),
    path('paper/create/', views.PaperCreateView.as_view(), name='create_paper'),
    path('paper/<int:pk>/edit/', views.PaperUpdateView.as_view(), name='edit_paper'),
    path('paper/<int:pk>/', views.PaperDetailView.as_view(), name='paper_detail'),
    path('paper/<int:pk>/delete/', views.PaperDeleteView.as_view(), name='delete_paper'),
    path('search/', views.PaperSearchView.as_view(), name='paper_search'),
    path("import-image/", views.ImageImportCreateView.as_view(), name="import_image"),
    path("generated-image/<int:pk>/", views.ImageGeneratedDetailView.as_view(), name="view_generated_image"),
    path('logout/', TemplateView.as_view(template_name='FinalProject/logout_confirm.html'), name='logout_confirm'),
    path('logout/confirm/', LogoutView.as_view(next_page='login2'), name='logoutFP'),
    path('login/', views.login_view, name='login2'),
    path("imputation/create/", views.ImputationCreateView.as_view(), name="create_imputation"),
    path("imputed-image/<int:pk>/", views.ImputedImageDetailView.as_view(), name="view_imputed_image"),
    path("apply-imputation/", views.apply_imputation, name="apply_imputation"),
    path("delete-image/<int:pk>/", views.ImageDeleteView.as_view(), name="delete_image"),
    path("delete-generated-image/<int:pk>/", views.ImageGeneratedDeleteView.as_view(), name="delete_generated_image"),
    path("impute-image/", views.ImputeImageView.as_view(), name="select_impute_image"),
    path('corrupted-images/', views.CorruptedImageListView.as_view(), name='corrupted_image_list'),
    path('corrupted-images/<int:pk>/', views.CorruptedImageDetailView.as_view(), name='corrupted_image_detail'),
    path('masks/', views.MaskListView.as_view(), name='mask_list'),
    path('masks/<int:pk>/', views.MaskDetailView.as_view(), name='mask_detail'),
    path('imputation-comparison/', views.imputation_comparison, name='imputation_comparison'),
]
