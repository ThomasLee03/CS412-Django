from django.contrib import admin
#file: admin.py
# author: Thomas Lee (tlee03@bu.edu), 10/5/2024


from .models import Profile, StatusMessage, Image, Friend
# Register your models here.

admin.site.register(Profile)

admin.site.register(StatusMessage)

admin.site.register(Image)

admin.site.register(Friend)