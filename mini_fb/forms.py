#file: forms.py
# author: Thomas Lee (tlee03@bu.edu), 10/13/2024

from django import forms
from .models import Profile, StatusMessage

class CreateProfileForm(forms.ModelForm):
    '''A form to add a profile to the database.'''
    class Meta:
        '''associate this form with the profile model; select fields.'''
        model = Profile 
        fields = ['first_name', 'last_name', 'city', 'email_address', 'profile_image_url']  # which fields from model should we use

class CreateStatusMessageForm(forms.ModelForm):
    '''A form to add a message to the database.'''
    class Meta:
        '''associate this form with the status message model; select fields.'''
        model = StatusMessage 
        fields = ['message']  # which fields from model should we use


class UpdateProfileForm(forms.ModelForm):
    class Meta:
        '''associate this form with the profile model; select fields.'''
        model = Profile 
        fields = ['city', 'email_address', 'profile_image_url']  # which fields from model should we use

class UpdateStatusMessageForm(forms.ModelForm):
    class Meta:
        '''associate this form with the profile model; select fields.'''
        model = StatusMessage 
        fields = ['message']  # which fields from model should we use

