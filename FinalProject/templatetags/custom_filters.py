from django import template
import base64
register = template.Library()
from django.core.files.base import ContentFile

@register.filter
def dict_contains(dictionary, key):
    """Check if the dictionary contains the given key."""
    if dictionary is None:
        return False
    return str(key) in dictionary

@register.filter
def starts_with(text, starts):
    """
    Returns True if the text starts with the given substring.
    """
    if isinstance(text, str) and isinstance(starts, str):
        return text.startswith(starts)
    return False

@register.filter
def contains_img(text):
    """
    Returns True if the text contains an <img> tag.
    """
    return '<img' in text


@register.filter
def has_key(dictionary, key):
    return key in dictionary

@register.filter
def filter_by_image(generated_images, image):
    return generated_images.filter(researcher=image.researcher)


@register.filter(name='base64encode')
def base64encode(value):
    """Encodes an image to base64"""
    if isinstance(value, ContentFile):
        # Handle ContentFile object
        img_data = value.read()
    elif value:
        # If it's a file-like object with a path (e.g., FileField)
        with open(value.path, "rb") as img_file:
            img_data = img_file.read()
    else:
        return None
    
    return base64.b64encode(img_data).decode('utf-8')