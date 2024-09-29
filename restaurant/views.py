#hw/views.py
#description: write view functions to handle URL requests for the hw app

import datetime
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse

import time 
import random
# Create your views here.
def main(request):

    #use this template to render the response
    template_name = 'restaurant/main.html'


    #delegate rendering work to the template 

    #render renders a response to a template
    return render(request, template_name)

def order(request):
    # Use this template to render the response
    template_name = 'restaurant/order.html'

    # Create a list of dictionaries for the daily specials
    food = [
        {
            "name": "Namburger",
            "details": "A delicious beef burger with lettuce, tomato, and our special sauce.",
            "price": 9.99
        },
        {
            "name": "Cheeseburger",
            "details": "Juicy grilled beef patty with melted cheddar cheese, served with fries.",
            "price": 10.99
        },
        {
            "name": "Sparkling Water",
            "details": "Refreshing sparkling water, served chilled.",
            "price": 250
        }
    ]

    # Select a random item from the food list as the daily special
    daily_special = food[random.randint(0, len(food)-1)]

    # Create a dictionary of context variables for the template
    context = {
        "daily_special": daily_special,
    }

    # Render the response to the template
    return render(request, template_name, context)


def confirmation(request):
    '''Process the form submission, and generate a result.'''
    template_name = "restaurant/confirmation.html"

    # Menu items and their prices (corresponding to the items in the form)
    menu_items = {
        'burger': {'name': 'Burger', 'price': 8.99},
        'pasta': {'name': 'Pasta', 'price': 12.99},
        'pizza': {'name': 'Pizza', 'price': 10.99},
        'salad': {'name': 'Salad', 'price': 7.99}
    }

    # If POST request, read the form data
    if request.method == 'POST':
        customer_name = request.POST.get('customer_name')
        customer_phone = request.POST.get('customer_phone')
        customer_email = request.POST.get('customer_email')
        special_instructions = request.POST.get('special_instructions', '')

        # Capture ordered items and calculate the total price
        ordered_items = []
        total_price = 0

        # Get the list of selected menu items
        selected_items = request.POST.getlist('menu_item')

        # Process the pizza separately to avoid duplicates
        pizza_ordered = False

        # Add selected items to the ordered_items list and calculate the total price
        for item_key in selected_items:
            if item_key == 'pizza':
                pizza_ordered = True
            elif item_key in menu_items:
                ordered_items.append(menu_items[item_key])
                total_price += menu_items[item_key]['price']

        # If pizza is ordered, check for toppings and handle them
        if pizza_ordered:
            pizza_toppings = request.POST.getlist('pizza_toppings')
            if pizza_toppings:
                topping_cost = 0.50 * len(pizza_toppings)
                total_price += menu_items['pizza']['price'] + topping_cost
                ordered_items.append({
                    'name': f"Pizza with {', '.join(pizza_toppings)}",
                    'price': menu_items['pizza']['price'] + topping_cost
                })
            else:
                # If no toppings were selected, add plain pizza
                ordered_items.append(menu_items['pizza'])
                total_price += menu_items['pizza']['price']

        # Handle the daily special (if selected)
        daily_special_name = request.POST.get('daily_special_name')
        daily_special_price = request.POST.get('daily_special_price')
        if daily_special_name and daily_special_name in selected_items:
            ordered_items.append({'name': daily_special_name, 'price': float(daily_special_price)})
            total_price += float(daily_special_price)

        # Calculate the ready time (random between 30 to 60 minutes from current time)
        current_time = datetime.datetime.now()
        ready_time_delta = datetime.timedelta(minutes=random.randint(30, 60))
        ready_time = (current_time + ready_time_delta).strftime('%I:%M %p')

        # Context for the template
        context = {
            'customer_name': customer_name,
            'customer_phone': customer_phone,
            'customer_email': customer_email,
            'special_instructions': special_instructions,
            'ordered_items': ordered_items,
            'total_price': round(total_price, 2),
            'ready_time': ready_time
        }

        # Render the response to the template
        return render(request, template_name, context)
