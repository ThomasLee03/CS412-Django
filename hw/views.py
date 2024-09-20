#hw/views.py
#description: write view functions to handle URL requests for the hw app

from django.shortcuts import render
from django.http import HttpRequest, HttpResponse

import time 
import random
# Create your views here.
def home (request):
    '''Handle the main URL for the hw app.'''

    response_text = f'''
    <html>
    
    <h1>Hello, world!</h1>
    <p> this is our first django web app</p>

    This page is generated at  {time.ctime()}.
    </html>
    '''

    #create and return a response to the client

    return HttpResponse(response_text)

def home(request):
    '''
    Function to handle the URL request for /hw (main page)
    Delegate rendering to the template hw/home.html'''

    #use this template to render the response
    template_name = 'hw/home.html'

    #create a dictionary of context variables for the template: 
    context = {
        "current_time" : time.ctime(),
        "letter1": chr(random.randint(65,90)), # a letter from A...Z
        "letter2": chr(random.randint(65,90)), # a letter from A...Z
        "number":random.randint(1,10), # a number fronm 1 to 10
    }

    #delegate rendering work to the template 

    #render renders a response to a template
    return render(request, template_name, context)

def about(request):
    '''
    Function to handle the URL request for /hw/about (main page)
    Delegate rendering to the template hw/home.html'''

    #use this template to render the response
    template_name = 'hw/about.html'

    #create a dictionary of context variables for the template: 
    context = {
        "current_time" : time.ctime(),
        "letter1": chr(random.randint(65,90)), # a letter from A...Z
        "letter2": chr(random.randint(65,90)), # a letter from A...Z
        "number":random.randint(1,10), # a number fronm 1 to 10
    }

    #delegate rendering work to the template 

    #render renders a response to a template
    return render(request, template_name, context)
