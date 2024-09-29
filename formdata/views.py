## formdata/views.py 
## get the form to display as "main" web page for this app:
from django.shortcuts import render, HttpResponse, redirect
# Create your views here.
def show_form(request):
    '''Show the web page with the form.'''
    # template_name = 'django/formdata/templates/formdata/form.html'
    template_name = 'formdata/form.html'
    return render(request, template_name)


def submit(request):
    '''
    Handle the form submission
    Read the form data from the request,
    and send it back to a template
    '''

    template_name = 'formdata/confirmation.html'
     # read the form data into python variables:
    if request.POST:
        #print(request.POST)
        name = request.POST['name']
        favorite_color = request.POST['favorite_color']
        context = {
            'name': name,
            'favorite_color':  favorite_color,
            
        }
        return render(request, template_name, context=context)
    #an ok solution, will just have a webpage that says nope
    #return HttpResponse("Nope.")

    ##a better solution: however when it is clicked on it will show the url of the submit html, which we don't want
    #template_name = 'formdata/form.html'
    #return render(request, template_name)

    #a better solution: redirect to the correct URL: this sends them to the form.html
    return redirect("show_form")
    