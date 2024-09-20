
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse

import time 
import random

quotes = [
    "I'm ready, I'm ready, I'm ready!",
    "The Krusty Krab pizza, is the pizza, for you and me!",
    "Is mayonnaise an instrument?",
    "I'm ugly and I'm proud!",
    "F is for friends who do stuff together!",
    "I don't need it. I don't need it. I definitely don’t need it.",
    "It took us three days to make that potato salad. Three days!",
    "I wumbo, you wumbo, he, she, me wumbo!",
    "The inner machinations of my mind are an enigma.",
    "Firmly grasp it in your hand!",
    "Ravioli, ravioli, give me the formuoli!",
    "This is not your average, everyday darkness. This is... advanced darkness.",
    "I’m a Goofy Goober, yeah!",
    "I can’t see my forehead.",
    "I don't get it.",
    "CHOCOLATE! CHOCOLATE!",
    "I can't, it's for the customer.",
    "Oh well, I guess I'm not wearing any pants today.",
    "Patrick, you’re a genius!",
    "You'll never guess what I found in my sock last night.",
    "It's a giraffe!",
    "Is this the Krusty Krab? No, this is Patrick!",
    "You don't need a license to drive a sandwich.",
    "I’m ready! Promotion!",
    "Well, it may be stupid, but it's also dumb.",
    "Goodbye everyone, I’ll remember you all in therapy.",
    "Look at it! Look at it! I want all of you to look at it!",
    "Too bad SpongeBob's not here to enjoy SpongeBob not being here.",
    "I can't believe I'm saying this, but poor Squidward.",
    "Licking doorknobs is illegal on other planets.",
    "Patrick, don't you have to be stupid somewhere else?",
    "I was born with glass bones and paper skin. Every morning I break my legs, and every afternoon I break my arms.",
    "No, this is Patrick!",
    "It’s called the ugly barnacle. Once there was an ugly barnacle. He was so ugly that everyone died. The end.",
    "You’re good. You’re good. You’re good.",
    "Oh, tartar sauce!",
    "I’ll have you know I stubbed my toe last week while watering my spice garden, and I only cried for 20 minutes.",
    "I can’t believe it. I’ve been betrayed by my own brain.",
    "The best time to wear a striped sweater is all the time.",
    "Oh, so this is the thanks I get for working overtime.",
    "Squidward, we don’t need television. Not as long as we have our… imagination.",
    "I'm a Goofy Goober, ROCK!",
    "You're good. You're good.",
    "Oh no, I broke it!",
    "Hi Kevin.",
    "You’re good, you’re good.",
    "Sandy’s a girl?",
    "Do you smell it? That smell. A kind of smelly smell. The smelly smell that smells… smelly.",
    "I’m ready! I’m ready! I’m ready!",
    "Can I be excused for the rest of my life?",
    "I knew I shouldn’t have gotten out of bed today.",
    "At first, we didn't know what to do with all the money. We tried burying it, shredding it, and burning it.",
    "I'm not a Krusty Krab.",
    "I can't hear you, it's too dark in here.",
    "I made it with my tears.",
    "Patrick, you genius!",
    "I'm ugly and I'm proud!",
    "Patrick, I don't think Wumbo is a real word.",
    "I don’t get it.",
    "That’s not disturbing, this is disturbing.",
    "Leedle-leedle-leedle-lee!",
    "It's not about winning, it's about fun!",
    "Patrick, what am I now? Stupid? No, I’m Texas!",
    "We should take Bikini Bottom and push it somewhere else!",
    "You’ll never guess what I found in my sock last night.",
    "Oh, I can’t read.",
    "Once upon a time, there was an ugly barnacle. He was so ugly that everyone died. The end.",
    "This isn’t your average, everyday darkness. This is… advanced darkness.",
    "No one says ‘cool’ anymore. That’s such an old person thing. Now we say ‘coral.’",
    "Oh, barnacles.",
    "You're good. You're good.",
    "Is mayonnaise an instrument?",
    "You don't need a license to drive a sandwich.",
    "I love money!",
    "You know, it’s not all about fun and games. Sometimes you have to get a little work done.",
    "I can't see my forehead.",
    "I love my job!",
    "Another day, another nickel.",
    "This isn’t your average, everyday stupid.",
    "Don't worry, Patrick, I'll save you!",
    "A five-letter word for happiness: money.",
    "We're not talking about some ordinary sandwich here. This is the deluxe.",
    "Who you callin' pinhead?",
    "Does this look unsure to you?",
    "It's not just a boulder. It's a rock!",
    "No, Patrick, mayonnaise is not an instrument.",
    "Dumb people are always blissfully unaware of how dumb they really are.",
    "You don’t need a license to drive a sandwich.",
    "Look at me! I'm naked!",
    "Firmly grasp it!",
    "What’s that? I should kill everyone and escape?",
    "Once upon a time, there was an ugly barnacle. He was so ugly that everyone died. The end.",
    "Don't worry Squidward, I've got a plan.",
    "This is the best day ever!",
    "We did it, Patrick! We saved the city!",
    "The Krusty Krab pizza, is the pizza, for you and me!",
    "The best time to wear a striped sweater is all the time.",
    "Remember, licking doorknobs is illegal on other planets."
]


images = [
    "https://www.hdwallpaper.nu/wp-content/uploads/2015/11/Spongebob_Squarepants_wallpaper_009.jpg",
    "https://pngimg.com/uploads/spongebob/spongebob_PNG38.png",
    "https://media.tenor.com/DWol7-qAlsoAAAAC/spongebob-dying.gif",
    "https://i.ytimg.com/vi/y1DakJCG1ro/hqdefault.jpg",
    "https://images5.fanpop.com/image/photos/31200000/Spongebob-Squarepants-spongebob-squarepants-31281685-1280-1024.jpg",
    "https://www.hdwallpaper.nu/wp-content/uploads/2015/11/Spongebob_Squarepants_wallpaper_017.jpg",
    "https://images6.fanpop.com/image/photos/33200000/Spongebob-spongebob-squarepants-33210737-2392-2187.jpg",
    "https://images6.fanpop.com/image/photos/33100000/Spongebob-Wallpaper-spongebob-squarepants-33184546-1024-768.jpg"
]

# Create your views here.

def quote(request):
    '''
    Function to handle the URL request for /quotes (main page)
    Delegate rendering to the template quotes/quotes.html'''

    #use this template to render the response
    template_name = 'quotes/quote.html'

    #create a dictionary of context variables for the template: 
    context = {
        "quote": quotes[random.randint(0,len(quotes)-1)], #one of the quotes
        "image": images[random.randint(0,len(images)-1)], #one of the images links
    }


    return render(request, template_name, context)

def show_all(request):
    '''
    Function to handle the URL request for /quotes/about 
    Delegate rendering to the template quotes/about.html'''

    #use this template to render the response
    template_name = 'quotes/show_all.html'

    #create a dictionary of context variables for the template: 
    context = {
        "quotes": quotes,
        "images": images,
    }

    return render(request, template_name, context)

def about(request):
    '''
    Function to handle the URL request for /hw/about (main page)
    Delegate rendering to the template hw/home.html'''

    #use this template to render the response
    template_name = 'quotes/about.html'

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