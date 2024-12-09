from django.shortcuts import render

# Create your views here.
from django.db.models.query import QuerySet
from django.views.generic import ListView, DetailView


import plotly
import plotly.graph_objects as go

from . models import Result
class ResultsListView(ListView):
    '''View to display marathon results'''
    template_name = 'marathon_analytics/results.html'
    model = Result
    context_object_name = 'results'
    #makes sure that only 50 are on each page
    paginate_by = 50
    
    def get_queryset(self):
        
        # limit results to first 25 records (for now)
        qs = super().get_queryset()
        #return qs[:25]
        return qs
    
class ResultsListView(ListView):
    '''View to display marathon results'''
    template_name = 'marathon_analytics/results.html'
    model = Result
    context_object_name = 'results'
    #make sure only 50 are on each page
    paginate_by = 50
    def get_queryset(self):
        
        # start with entire queryset
        qs = super().get_queryset().order_by('place_overall')
        # filter results by these field(s):
        if 'city' in self.request.GET:
            city = self.request.GET['city']
            qs = qs.filter(city=city)
            #qt = Restuls.object.filter(city_icontains = city)
 
        return qs
    
class RestulDetailView(DetailView):
    '''display a single result on it's own page'''
    template_name = 'marathon_analytics/result_detail.html'
    model = Result
    #what does r do?
    context_object_name = 'r'

    #implement some methods
    def get_context_data(self, **kwargs):

        #get superclass version of context
        context = super().get_context_data(**kwargs)
        r = context['r'] #obtain the single Result instance 

        #get data: half-marathon splits
        first_half_seconds = (r.time_half1.hour*3600 + r.time_half1.minute*60 +r.time_half1.second)
        second_half_seconds = (r.time_half2.hour*3600 + r.time_half2.minute*60 +r.time_half2.second)
        #build a pie chart
        x = ['first half time', 'second half time']
        y = [first_half_seconds, second_half_seconds]

        #add the pie chart to the context
        fig = go.Pie(labels=x, values=y)
        pie_div = plotly.offline.plot({'data': [fig]}, auto_open = False, output_type = 'div')
        context['pie_div'] = pie_div

        #create a bar chart with the number of runners passed and who passed by
        x = [f'runners passed by {r.first_name}', f'runner who passed {r.first_name}']
        y = [r.get_runners_passed(), r.get_runners_was_passed()]
        fig = go.Bar(x=x, y=y)
        bar_div = plotly.offline.plot({'data': [fig]}, auto_open = False, output_type = 'div')
        context['bar_div'] = bar_div
        
        return context

