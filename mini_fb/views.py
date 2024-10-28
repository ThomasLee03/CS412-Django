from django.shortcuts import render

#file: views.py
# author: Thomas Lee (tlee03@bu.edu), 10/5/2024



# Create your views here.
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView, View
from .models import * ## import models (e.g., Profile)
from django.urls import reverse
from django import forms
from .forms import CreateProfileForm, CreateStatusMessageForm, UpdateProfileForm, UpdateStatusMessageForm
from django.shortcuts import get_object_or_404, redirect
from typing import Any

# Create your views here.
#class based-view
class ShowAllProfilesView(ListView):
    '''the view to show all profiles'''
    model = Profile #the model to display
    template_name = 'mini_fb/show_all_profiles.html'
    context_object_name = 'profiles'

class ShowNewsFeedView(DetailView):
    model = Profile
    template_name = 'mini_fb/news_feed.html'
    context_object_name = 'profile'

class ShowFriendSuggestionsView(DetailView):
    model = Profile
    template_name = 'mini_fb/friend_suggestions.html'
    context_object_name = 'profile'
    

class CreateFriendView(View):
    def dispatch(self, request, *args, **kwargs):
        # Get the "self" profile from the URL parameters (e.g., user initiating the friendship)
        profile_id = self.kwargs.get('pk')
        other_profile_id = self.kwargs.get('other_pk')

        # Use get_object_or_404 to fetch the Profile objects or return a 404 if they don't exist
        profile = get_object_or_404(Profile, pk=profile_id)
        other_profile = get_object_or_404(Profile, pk=other_profile_id)

        # Call the add_friend method on the profile object
        result = profile.add_friend(other_profile)

        # Redirect the user back to the profile page with the result
        return redirect('profile', pk=profile_id)

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

class UpdateProfileView(UpdateView):
    model = Profile
    form_class = UpdateProfileForm
    template_name = "mini_fb/update_profile_form.html"

class DeleteStatusMessageView(DeleteView):
    model = StatusMessage 
    context_object_name = 'status_message'
    template_name = "mini_fb/delete_status_form.html"
    def get_success_url(self) -> str:
        '''Return the URL to redirect to after successfully submitting form.'''
        # Get the profile related to the status message being deleted
        profile = self.object.profile
        # Use reverse to generate the URL for the profile page
        return reverse('profile', kwargs={'pk': profile.pk})
    
class UpdateStatusMessageView(UpdateView):
    model = StatusMessage 
    form_class = UpdateStatusMessageForm
    context_object_name = 'status_message'
    template_name = "mini_fb/update_status_form.html"
    def get_success_url(self) -> str:
        '''Return the URL to redirect to after successfully submitting form.'''
        # Get the profile related to the status message being deleted
        profile = self.object.profile
        # Use reverse to generate the URL for the profile page
        return reverse('profile', kwargs={'pk': profile.pk})

class CreateStatusMessageView (CreateView):
    '''A view to create a new message and save it to the database.'''
    form_class = CreateStatusMessageForm
    template_name = "mini_fb/create_status_form.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        #get the context data from the super class
        context = super().get_context_data(**kwargs)
        
        #add the message referred to by the url into this context
        profile = Profile.objects.get(pk = self.kwargs['pk'])
        context['profile'] = profile
        return context
    def form_valid(self, form):
        '''
        Handle the form submission. We need to set the foreign key by 
        attaching the profile to the status message object.
        We can find the profile PK in the URL (self.kwargs).        '''
      
        print(form.cleaned_data)

        #self.kwargs stands for key word arguments 
        #find the profile identified by the pk (primary key) from the url pattern
        #attach this profile to the instance of the comment to set its FK
        profile = Profile.objects.get(pk=self.kwargs['pk'])
        
        #now attach to instance of the status message 
        form.instance.profile = profile
        # save the status message to database
        sm = form.save()

        # read the file from the form:
        files = self.request.FILES.getlist('files')
        for file in files:
            # Create an image instance for each uploaded file
            image = Image()
            image.status_message = sm  # Set the foreign key (status message)
            image.image_file = file  # Set the file into the ImageField
            image.save()  # Save the Image object to the database
        #delegate work to superclass version from method
        return super().form_valid(form)
        
    def get_success_url(self) -> str:
        '''Return the URL to redirect to after successfully submitting form.'''
        return reverse('profile', kwargs = self.kwargs)
