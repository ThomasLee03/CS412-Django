from django.shortcuts import render

#file: views.py
# author: Thomas Lee (tlee03@bu.edu), 10/5/2024



# Create your views here.
from django.views.generic import ListView, DetailView, CreateView
from .models import * ## import models (e.g., Profile)
from django.urls import reverse
from django import forms
from .forms import CreateProfileForm, CreateStatusMessageForm
from typing import Any

# Create your views here.
#class based-view
class ShowAllProfilesView(ListView):
    '''the view to show all profiles'''
    model = Profile #the model to display
    template_name = 'mini_fb/show_all_profiles.html'
    context_object_name = 'profiles'


class ShowProfilePageView(DetailView):
    '''Show the details for one profile.'''
    model = Profile
    template_name = 'mini_fb/show_profile.html' 
    context_object_name = 'profile'

class CreateProfileView(CreateView):
    '''A view to create a new profile and save it to the database.'''
    model = Profile
    form_class = CreateProfileForm
    template_name = "mini_fb/create_profile_form.html"

class CreateStatusMessageView (CreateView):
    '''A view to create a new message and save it to the database.'''
    form_class = CreateStatusMessageForm
    template_name = "mini_fb/create_status_form.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        #get the context data from the super class
        context = super().get_context_data(**kwargs)
        
        #add the article referred to by the url into this context
        profile = Profile.objects.get(pk = self.kwargs['pk'])
        context['profile'] = profile
        return context
    def form_valid(self, form):
        '''
        Handle the form submission. We need to set the foreign key by 
        attaching the profile to the status message object.
        We can find the profile PK in the URL (self.kwargs).
        '''
      
        print(form.cleaned_data)

        #self.kwargs stands for key word arguments 
        #find the profile identified by the pk (primary key) from the url pattern
        #attach this profile to the instance of the comment to set its FK
        profile = Profile.objects.get(pk=self.kwargs['pk'])
        #now attach to instance of the status message 
        form.instance.profile = profile
        #delegate work to superclass version from method
        return super().form_valid(form)
        
    def get_success_url(self) -> str:
        '''Return the URL to redirect to after successfully submitting form.'''
        return reverse('profile', kwargs = self.kwargs)
