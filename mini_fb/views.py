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
from django.contrib.auth.mixins import LoginRequiredMixin 
from django.contrib.auth import logout

def custom_logout(request):
    logout(request)
    return redirect('show_all')  # Redirect to the homepage or another page

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
    def get_object(self):
        '''Return the Profile object associated with that user.'''
        user = self.request.user
        try:
            # Retrieve the Profile object for the logged-in user
            profile = Profile.objects.get(user=user)
            return profile
        except Profile.DoesNotExist:
            # Optionally handle the case where no Profile exists
            # Return None or redirect, depending on your app's needs
            return None

class ShowFriendSuggestionsView(DetailView):
    model = Profile
    template_name = 'mini_fb/friend_suggestions.html'
    context_object_name = 'profile'
    def get_object(self):
        '''Return the Profile object associated with that user.'''
        user = self.request.user
        try:
            # Retrieve the Profile object for the logged-in user
            profile = Profile.objects.get(user=user)
            return profile
        except Profile.DoesNotExist:
            # Optionally handle the case where no Profile exists
            # Return None or redirect, depending on your app's needs
            return None
    

class CreateFriendView(LoginRequiredMixin, View):
    def dispatch(self, request, *args, **kwargs):
        # Get the "self" profile from the URL parameters (e.g., user initiating the friendship)
        other_profile_id = self.kwargs.get('other_pk')

        # Use get_object_or_404 to fetch the Profile objects or return a 404 if they don't exist
        profile = get_object_or_404(Profile, user = request.user)
        other_profile = get_object_or_404(Profile, pk=other_profile_id)

        # Call the add_friend method on the profile object
        result = profile.add_friend(other_profile)

        # Redirect the user back to the profile page with the result
        return redirect(reverse('profile', kwargs={'pk': profile.pk}))
    def form_valid(self, form):
        '''this method is called as part of the form processing'''
        print(f'CreateArticleView.form_valid(): form.cleaned_data={form.cleaned_data}')

        #find the user who is logged in
        user = self.request.user

        #attach that user as a FK(foreign key) to the new Article instance
        form.instance.user = user
        #this is now setting the user to the user attribute in the model 

        #let the superclas do the real work to see what is happening with the form
        return super().form_valid(form)
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login')
    def get_object(self):
        '''Return the Profile object associated with that user.'''
        user = self.request.user
        try:
            # Retrieve the Profile object for the logged-in user
            profile = Profile.objects.get(user=user)
            return profile
        except Profile.DoesNotExist:
            # Optionally handle the case where no Profile exists
            # Return None or redirect, depending on your app's needs
            return None

class ShowProfilePageView(DetailView):
    '''Show the details for one profile.'''
    model = Profile
    template_name = 'mini_fb/show_profile.html' 
    context_object_name = 'profile'

class UpdateProfileView(LoginRequiredMixin, UpdateView):
    model = Profile
    form_class = UpdateProfileForm
    template_name = "mini_fb/update_profile_form.html"
    def form_valid(self, form):
        '''this method is called as part of the form processing'''
        print(f'CreateArticleView.form_valid(): form.cleaned_data={form.cleaned_data}')

        #find the user who is logged in
        user = self.request.user

        #attach that user as a FK(foreign key) to the new Article instance
        form.instance.user = user
        #this is now setting the user to the user attribute in the model 

        #let the superclas do the real work to see what is happening with the form
        return super().form_valid(form)
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login')
    def get_object(self):
        '''Return the Profile object associated with that user.'''
        user = self.request.user
        try:
            # Retrieve the Profile object for the logged-in user
            profile = Profile.objects.get(user=user)
            return profile
        except Profile.DoesNotExist:
            # Optionally handle the case where no Profile exists
            # Return None or redirect, depending on your app's needs
            return None

class DeleteStatusMessageView(LoginRequiredMixin, DeleteView):
    model = StatusMessage 
    context_object_name = 'status_message'
    template_name = "mini_fb/delete_status_form.html"
    def get_success_url(self) -> str:
        '''Return the URL to redirect to after successfully submitting form.'''
        # Get the profile related to the status message being deleted
        profile = self.object.profile
        # Use reverse to generate the URL for the profile page
        return reverse('profile', kwargs={'pk': profile.pk})
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login')
    
class UpdateStatusMessageView(LoginRequiredMixin, UpdateView):
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
    def form_valid(self, form):
        '''this method is called as part of the form processing'''
        print(f'CreateArticleView.form_valid(): form.cleaned_data={form.cleaned_data}')

        #find the user who is logged in
        user = self.request.user

        #attach that user as a FK(foreign key) to the new Article instance
        form.instance.user = user
        #this is now setting the user to the user attribute in the model 

        #let the superclas do the real work to see what is happening with the form
        return super().form_valid(form)
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login')

class CreateStatusMessageView (LoginRequiredMixin, CreateView):
    '''A view to create a new message and save it to the database.'''
    form_class = CreateStatusMessageForm
    template_name = "mini_fb/create_status_form.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        #get the context data from the super class
        context = super().get_context_data(**kwargs)
        
        #add the message referred to by the url into this context
        profile = Profile.objects.get(user= self.request.user)
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
        profile = Profile.objects.get(user= self.request.user)
        #find the user who is logged in
        user = self.request.user

        #attach that user as a FK(foreign key) to the new Article instance
        form.instance.user = user
        
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
        profile = Profile.objects.get(user=self.request.user)
        return reverse('profile', kwargs={'pk': profile.pk})
    def get_login_url(self) -> str:
        '''return the url of the login page'''
        return reverse('login')
    def get_object(self):
        '''Return the Profile object associated with that user.'''
        user = self.request.user
        try:
            # Retrieve the Profile object for the logged-in user
            profile = Profile.objects.get(user=user)
            return profile
        except Profile.DoesNotExist:
            # Optionally handle the case where no Profile exists
            # Return None or redirect, depending on your app's needs
            return None

class CreateProfileView(CreateView):
    '''A view to create a new profile and save it to the database.'''
    model = Profile
    form_class = CreateProfileForm
    template_name = "mini_fb/create_profile_form.html"
    def form_valid(self, form):
        '''This method is called as part of the form processing.'''
        print(f'CreateProfileView.form_valid(): form.cleaned_data={form.cleaned_data}')
        
        # Reconstruct the UserCreationForm with POST data
        user_form = UserCreationForm(self.request.POST)
        
        # Validate and save the UserCreationForm to create a new User
        if user_form.is_valid():
            user = user_form.save()  # Save the new User instance
            print(f"CreateProfileView.form_valid: created user {user}")
            
            # Log the new user in
            login(self.request, user)
            print(f"CreateProfileView.form_valid: {user} is logged in.")
            
            # Attach the user to the Profile instance
            form.instance.user = user
            
            # Delegate the rest to the superclass' form_valid method
            return super().form_valid(form)
        else:
            # If the UserCreationForm is invalid, re-render the form with errors
            print(f"CreateProfileView.form_valid: user_form errors: {user_form.errors}")
            return self.form_invalid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        #create new form and add the information and add the form itself to the context 
        context['UCForm'] = UserCreationForm()  # Create an instance of the form
        return context
    def get_object(self):
        '''Return the Profile object associated with that user.'''
        user = self.request.user
        try:
            # Retrieve the Profile object for the logged-in user
            profile = Profile.objects.get(user=user)
            return profile
        except Profile.DoesNotExist:
            # Optionally handle the case where no Profile exists
            # Return None or redirect, depending on your app's needs
            return None


#stuff to create a user
from django.contrib.auth.forms import UserCreationForm 
from django.contrib.auth.models import User
from django.contrib.auth import login
from django.shortcuts import redirect

class RegistrationView(CreateView):
    '''Handle registration of new Users'''

    template_name = 'mini_fb/register.html'
    form_class = UserCreationForm #built-in from django.contrib.auth.forms
    def dispatch(self, request, *args, **kwargs):
        '''Handle the User creation form submission'''

        #if we received an HTTP POST, we handle it
        if self.request.POST:
            #we handle it
            print(f"RegistrationView.patch: self.request.POST{self.request.POST}")

            #reconstruct the UserCreateForm from the POST data
            form = UserCreationForm(self.request.POST)
            
            #save the form, which creates a new User
            
            if not form.is_valid():
                print(f"form.errors = {form.errors}")
                #let CreateView.dispatch handle the problem 
                return super().dispatch(request, *args, **kwargs)
            
            
            user = form.save() #this will commit the insert to the database
            print(f"RegistrationView.dispatch: created user {user}")
            #log the User in
            login(self.request,user)
            
            print(f"RegistrationView.dispatch: {user} is logged in.")

            #note for mini_fb: attach the FK user to the Profile form instance 


            #return a reponse:
            return redirect(reverse('show_all_one'))

        #let CreateView.dispathch handle the HTTP GET request
        return super().dispatch(request, *args, **kwargs)
