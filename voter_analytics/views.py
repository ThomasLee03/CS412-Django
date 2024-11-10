#file: views.py
# author: Thomas Lee (tlee03@bu.edu), 11/10/2024
from django.shortcuts import render
from django.views.generic import ListView, DetailView
from . models import Voter
from datetime import datetime
import plotly
import plotly.graph_objects as go
from django.db.models import Count, F
# Create your views here.
class ResultsListView(ListView):
    '''View to display marathon results'''
    template_name = 'voter_analytics/results.html'
    model = Voter
    context_object_name = 'voter'
    #make sure only 100 are on each page
    paginate_by = 100
    def get_context_data(self, **kwargs):
            context = super().get_context_data(**kwargs)
            current_year = datetime.now().year
            context['years'] = list(range(1900, current_year + 1))  # List of years for dobmin and dobmax
            return context
    def get_queryset(self):
        
        # start with entire queryset
        qs = super().get_queryset().order_by('partyAffiliation')
        # filter results by these field(s):
        partyAffiliation = self.request.GET.get('partyAffiliation')
        if partyAffiliation:
            qs = qs.filter(partyAffiliation=partyAffiliation)
        dobmin = self.request.GET.get('dobmin')
        if dobmin:
            try:
                # Attempt to parse dobmin only if it's non-empty
                qs = qs.filter(dob__year__gte=int(dobmin))
            except ValueError:
                pass  # Ignore invalid date format

        # Filter by maximum date of birth if 'dobmax' is not empty
        dobmax = self.request.GET.get('dobmax')
        if dobmax:
            try:
                # Attempt to parse dobmax only if it's non-empty
                qs = qs.filter(dob__year__lte=int(dobmax))
            except ValueError:
                pass  # Ignore invalid date format

        # Filter by 'voterScore' if provided and not empty
        voter_score = self.request.GET.get('voterScore')
        if voter_score:
            try:
                qs = qs.filter(voter_score=int(voter_score))
            except ValueError:
                pass  # Handle any invalid integer input gracefully

        if 'election' in self.request.GET:
            selected_elections = self.request.GET.getlist('election')
            print("Selected Elections:", selected_elections)
            print("Query Before Filtering Elections:", qs.query)
            # Assuming each election corresponds to a field in your model, such as v20state, v21town, etc.
            # Use Q objects for complex OR filtering across multiple fields
            if '2020 State Election' in selected_elections:
                qs = qs.filter(v20state='TRUE')
            if '2021 Town Election' in selected_elections:
                qs = qs.filter(v21town='TRUE')
            if 'Primary Election' in selected_elections:
                qs = qs.filter(v21primary='TRUE')
            if '2022 General Election' in selected_elections:
                qs = qs.filter(v22general='TRUE')
            if '2023 Town Election' in selected_elections:
                qs = qs.filter(v23town='TRUE')

        return qs
        
class ResultDetailView(DetailView):
    print(Voter.objects.filter(pk=3).exists())


    '''display a single result on it's own page'''
    template_name = 'voter_analytics/result_detail.html'
    model = Voter
    context_object_name = 'r'

class ResultGraphsView(ListView):

    template_name = 'voter_analytics/result_graphs.html'
    model = Voter
    context_object_name = 'r'
    def get_context_data(self, **kwargs):

        #get superclass version of context
        context = super().get_context_data(**kwargs)
 
        # start with entire queryset
        qs = Voter.objects.all()
        # filter results by these field(s):
        partyAffiliation = self.request.GET.get('partyAffiliation')
        if partyAffiliation:
            qs = qs.filter(partyAffiliation=partyAffiliation)
        dobmin = self.request.GET.get('dobmin')
        if dobmin:
            try:
                # Attempt to parse dobmin only if it's non-empty
                qs = qs.filter(dob__year__gte=int(dobmin))
            except ValueError:
                pass  # Ignore invalid date format

        # Filter by maximum date of birth if 'dobmax' is not empty
        dobmax = self.request.GET.get('dobmax')
        if dobmax:
            try:
                # Attempt to parse dobmax only if it's non-empty
                qs = qs.filter(dob__year__lte=int(dobmax))
            except ValueError:
                pass  # Ignore invalid date format

        # Filter by 'voterScore' if provided and not empty
        voter_score = self.request.GET.get('voterScore')
        if voter_score:
            try:
                qs = qs.filter(voter_score=int(voter_score))
            except ValueError:
                pass  # Handle any invalid integer input gracefully

        if 'election' in self.request.GET:
            selected_elections = self.request.GET.getlist('election')
            print("Selected Elections:", selected_elections)
            print("Query Before Filtering Elections:", qs.query)
            # Assuming each election corresponds to a field in your model, such as v20state, v21town, etc.
            # Use Q objects for complex OR filtering across multiple fields
            if '2020 State Election' in selected_elections:
                qs = qs.filter(v20state='TRUE')
            if '2021 Town Election' in selected_elections:
                qs = qs.filter(v21town='TRUE')
            if 'Primary Election' in selected_elections:
                qs = qs.filter(v21primary='TRUE')
            if '2022 General Election' in selected_elections:
                qs = qs.filter(v22general='TRUE')
            if '2023 Town Election' in selected_elections:
                qs = qs.filter(v23town='TRUE')

        # Get the count of voters by each party affiliation
        party_counts = qs.values('partyAffiliation').annotate(count=Count('partyAffiliation'))

        # Separate labels and values for the chart
        labels = [item['partyAffiliation'] for item in party_counts]
        values = [item['count'] for item in party_counts]

        # Create a Plotly pie chart
        fig = go.Pie(labels=labels, values=values)
        pie_div = plotly.offline.plot({'data': [fig]}, auto_open = False, output_type = 'div')

        context['pie_div'] = pie_div

        # Aggregate the count of voters by year of birth
        birth_year_counts = qs.annotate(year_of_birth=F('dob__year')).values('year_of_birth').annotate(count=Count('year_of_birth')).order_by('year_of_birth')

        # Extract years and counts for plotting
        x = [str(item['year_of_birth']) for item in birth_year_counts]
        y = [item['count'] for item in birth_year_counts]

        fig = go.Bar(x=x, y=y)
        bar_div = plotly.offline.plot({'data': [fig]}, auto_open = False, output_type = 'div')
        context['bar_div'] = bar_div

        election_counts = {
        '2020 State Election': qs.filter(v20state='TRUE').count(),
        '2021 Town Election': qs.filter(v21town='TRUE').count(),
        'Primary Election': qs.filter(v21primary='TRUE').count(),
        '2022 General Election': qs.filter(v22general='TRUE').count(),
        '2023 Town Election': qs.filter(v23town='TRUE').count()
        }

        # Prepare data for the bar chart
        x = list(election_counts.keys())
        y = list(election_counts.values())

        # Create the bar chart
        fig = go.Bar(x=x, y=y)
        bar_div2 = plotly.offline.plot({'data': [fig]}, auto_open = False, output_type = 'div')
        context['bar_div2'] = bar_div2
        
        return context