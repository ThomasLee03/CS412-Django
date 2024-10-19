#blogs/views.py
#define the views for the blog app
from django.shortcuts import render

# Create your views here.
from django.views.generic import ListView, DetailView
from .models import * ## import models (e.g., Article)

from django import forms
from .models import Comment
class CreateCommentForm(forms.ModelForm):
    '''A form to add a Comment to the database.'''
    class Meta:
        '''associate this form with the Comment model; select fields.'''
        model = Comment
        fields = ['article', 'author', 'text', ]  # which fields from model should we use

#class based-view
class ShowAllView(ListView):
    '''the view to show all articles'''
    model = Article #the model to display
    template_name = 'blog/show_all.html'
    context_object_name = 'articles'

from django.views.generic.edit import CreateView
from django.urls import reverse
from .forms import CreateArticleForm, CreateCommentForm
from typing import Any

class CreateCommentView(CreateView):
    '''A view to create a new comment and save it to the database.'''
    form_class = CreateCommentForm
    template_name = "blog/create_comment_form.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        #get the context data from the super class
        context = super().get_context_data(**kwargs)
        
        #add the article referred to by the url into this context
        article = Article.objects.get(pk = self.kwargs['pk'])
        context['article'] = article
        return context
    def form_valid(self, form):
        '''
        Handle the form submission. We need to set the foreign key by 
        attaching the Article to the Comment object.
        We can find the article PK in the URL (self.kwargs).
        '''
      
        print(form.cleaned_data)

        #self.kwargs stands for key word arguments 
        #find the article identified by the pk (primary key) from the url pattern
        #attach this article to the instance of the comment to set its FK
        article = Article.objects.get(pk=self.kwargs['pk'])
        # print(article)
        #now attach to instance of the comment 
        form.instance.article = article
        #delegate work to superclass version from method
        return super().form_valid(form)
        
## also:  revise the get_success_url
    def get_success_url(self) -> str:
        '''Return the URL to redirect to after successfully submitting form.'''
        #return reverse('show_all')
       # article = Article.objects.get(pk = self.kwargs['pk'])
       # return reverse('article', kwargs={'pk': article.pk})
        return reverse('article', kwargs = self.kwargs)


from .models import Article
from django.views.generic import ListView, DetailView
import random
### ... 
# new view class only
class RandomArticleView(DetailView):
    '''Show the details for one article.'''
    model = Article
    template_name = 'blog/article.html'
    context_object_name = 'article'
    # pick one article at random:
    def get_object(self):
        '''Return one Article object chosen at random.'''
        all_articles = Article.objects.all()
        return random.choice(all_articles)
    

class ArticlePageView(DetailView):
    '''Show the details for one article.'''
    model = Article
    template_name = 'blog/article.html' ## reusing same template!!
    context_object_name = 'article'


class CreateArticleView(CreateView):
    '''a view class to create a new article instance.'''
    form_class = CreateArticleForm
    template_name = 'blog/create_article_form.html'

    def form_valid(self, form):
        '''this method is called as part of the form processing'''
        print(f'CreateArticleView.form_valid(): form.cleaned_data={form.cleaned_data}')
        #let the superclas do the real work to see what is happening with the form
        return super().form_valid(form)