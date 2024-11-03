from django.db import models

from django.urls import reverse
from django.contrib.auth.models import User ##this imports the user accounts

class Article(models.Model):
    '''Encapsulate the idea of a Article by some author.'''

    #every article has one User:
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # data attributes of a Article:
    title = models.TextField(blank=False)
    author = models.TextField(blank=False)
    text = models.TextField(blank=False)
    published = models.DateTimeField(auto_now=True)
    # image_url = models.URLField(blank=True) ## new

    #image now can take in a image file
    image_file = models.ImageField(blank = True) 
    
    def __str__(self):
        '''Return a string representation of this Article object.'''
        return f'{self.title} by {self.author}'
    
    def get_comments(self):
        '''Return all of the comments about this article.'''
        comments = Comment.objects.filter(article=self)
        return comments
    
    def get_absolute_url(self):
        '''return the URL that will display an instance of this object'''
        #self.pk is the primary to this article instance kwargs = key word arguments
        return reverse('article', kwargs={'pk': self.pk})
    
class Comment(models.Model):
    '''Encapsulate the idea of a Comment on an Article.'''
    
    # data attributes of a Comment:
    article = models.ForeignKey("Article", on_delete=models.CASCADE)
    author = models.TextField(blank=False)
    text = models.TextField(blank=False)
    published = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        '''Return a string representation of this Comment object.'''
        return f'{self.text}'