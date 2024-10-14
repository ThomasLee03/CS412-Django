from django.db import models

#file: models.py
# author: Thomas Lee (tlee03@bu.edu), 10/5/2024
from django.urls import reverse
# Create your models here.
class Profile(models.Model):
    '''Encapsulate the idea of an Profile by some author.'''
    # data attributes of a Profile:
    first_name = models.TextField(blank=False)
    last_name = models.TextField(blank=False)
    city = models.TextField(blank=False)
    email_address = models.TextField(blank=False)
    profile_image_url = models.URLField(blank = True)

    def get_messages(self):
        '''Return all of the comments about this article.'''
        comments = StatusMessage.objects.filter(profile=self)
        return comments

    def get_absolute_url(self):
        '''Return the URL to view this profile.'''
        return reverse('profile', kwargs={'pk': self.pk})
    
    
class StatusMessage(models.Model):
    '''Encapsulate the idea of a status message on a Profile.'''
    
    # data attributes of a StatusMessage:
    profile  = models.ForeignKey("Profile", on_delete=models.CASCADE)
    message = models.TextField(blank=False)
    timestamp  = models.DateTimeField(auto_now=True)
   
    def __str__(self):
        '''Return a string representation of this Comment object.'''
        return f'{self.message}'
  