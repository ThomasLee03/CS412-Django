from django.db import models

#file: models.py
# author: Thomas Lee (tlee03@bu.edu), 10/5/2024
from django.urls import reverse
from django.db.models import Q
from django.contrib.auth.models import User ##this imports the user accounts
# Create your models here.

class Friend(models.Model):
    timestamp = models.DateTimeField(auto_now=True)
    profile1 = models.ForeignKey("Profile", on_delete=models.CASCADE, related_name="profile1")
    profile2 = models.ForeignKey("Profile", on_delete=models.CASCADE, related_name="profile2")
    def __str__(self):
        return f'{self.profile1.first_name} {self.profile1.last_name} & {self.profile2.first_name} {self.profile2.last_name}'
class Profile(models.Model):
    '''Encapsulate the idea of an Profile by some author.'''
    # data attributes of a Profile:
    first_name = models.TextField(blank=False)
    last_name = models.TextField(blank=False)
    city = models.TextField(blank=False)
    email_address = models.TextField(blank=False)
    profile_image_url = models.URLField(blank = True)

    #now each profile is associated with a user 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def get_messages(self):
        '''Return all of the comments about this article.'''
        comments = StatusMessage.objects.filter(profile=self)
        return comments

    def get_absolute_url(self):
        '''Return the URL to view this profile.'''
        return reverse('profile', kwargs={'pk': self.pk})
    
    def get_friends(self):
        # Get all Friend instances where this profile is profile1 or profile2
        friends_as_profile1 = Friend.objects.filter(profile1=self)
        friends_as_profile2 = Friend.objects.filter(profile2=self)
        
        # Create a list of friends (as profiles)
        friends = [friend.profile2 for friend in friends_as_profile1]  # Friends where this profile is profile1
        friends += [friend.profile1 for friend in friends_as_profile2]  # Friends where this profile is profile2
        
        return friends
    
    def get_news_feed(self):
        friends = self.get_friends()
        status_messages = StatusMessage.objects.filter(profile__in = friends).order_by("-timestamp")
        return status_messages
    
    def get_friend_suggestions(self):
        
        friends = self.get_friends()
        not_friends = []
        for friend in friends:
            friends_of_friend  = Friend.objects.filter(~Q(profile1 = self) & ~Q(profile2 = self) & (Q(profile1 = friend) | Q(profile2 = friend)))

            not_friends += [fof.profile1 if fof.profile2 == friend else fof.profile2 for fof in friends_of_friend]
            
        not_friends = list(set(not_friends))

        # Filter out any profiles that are already direct friends of self
        not_friends = [fof for fof in not_friends if fof not in friends and fof != self]

        return not_friends
    def add_friend(self, other):
        if self == other:
            return "You cannot add yourself as a friend."
        
        # Check for friendship where self is profile1 and other is profile2
        friendship_1 = Friend.objects.filter(profile1=self, profile2=other).first()

        # Check for friendship where self is profile2 and other is profile1
        friendship_2 = Friend.objects.filter(profile1=other, profile2=self).first()

        # If neither friendship exists, create a new one
        if friendship_1 is None and friendship_2 is None:
            new_friendship = Friend(profile1=self, profile2=other)
            new_friendship.save()
            return "Friendship created successfully!"
        
        # If either friendship exists, do not create a duplicate
        return "This friendship already exists."

    
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
  