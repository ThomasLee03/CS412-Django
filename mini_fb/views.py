from django.shortcuts import render

#file: views.py
# author: Thomas Lee (tlee03@bu.edu), 10/5/2024


# Create your views here.
from django.views.generic import ListView
from .models import * ## import models (e.g., Profile)

# Create your views here.
#class based-view
class ShowAllProfilesView(ListView):
    '''the view to show all articles'''
    model = Profile #the model to display
    template_name = 'mini_fb/show_all_profiles.html'
    context_object_name = 'profiles'
