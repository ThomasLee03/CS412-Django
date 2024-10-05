from django.db import models

#file: modells.py
# author: Thomas Lee (tlee03@bu.edu), 10/5/2024

# Create your models here.
class Profile(models.Model):
    '''Encapsulate the idea of an Profile by some author.'''
    # data attributes of a Profile:
    first_name = models.TextField(blank=False)
    last_name = models.TextField(blank=False)
    city = models.TextField(blank=False)
    email_address = models.TextField(blank=False)
    profile_image_url = models.URLField(blank = True)
    
    

  