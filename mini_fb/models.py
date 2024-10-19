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
        '''Return a string representation of this status message object.'''
        return f'{self.message}'
    
    def get_images(self):
        '''Return all of the images about this status message.'''
        images = Image.objects.filter(status_message=self)
        return images
    
class Image(models.Model):
    '''Encapsulate the idea of an Image stored in the media directory.'''
    
    # data attributes of an Image:
    image_file = models.ImageField(upload_to='images/', blank=True) 
    uploaded_at = models.DateTimeField(auto_now=True)
    status_message = models.ForeignKey(StatusMessage, on_delete=models.CASCADE, related_name='images')

    def __str__(self):
        '''Return a string representation of this Image object.'''
        return f"Image {self.id} for StatusMessage {self.status_message.id}"
  