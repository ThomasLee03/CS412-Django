from django.contrib import admin
# file: admin.py
# author: Thomas Lee (tlee03@bu.edu), 11/22/2024

from .models import (
    Researcher,
    Image,
    ImageGenerated,
    ImageWithGenerated,
    Mask,
    CorruptedImage,
    Paper,
    PaperWithResearcher,
    PaperImage,
    PaperWithGeneratedImage,
    PaperWithMask, 
    PaperWithCorruptedImage
)
# Register your models here.
class PaperAdmin(admin.ModelAdmin):
    list_display = ['title', 'time_created']
    filter_horizontal = ['images', 'generated_images', 'masks', 'corrupted_images']

admin.site.register(Researcher)
admin.site.register(Image)
admin.site.register(ImageGenerated)
admin.site.register(ImageWithGenerated)
admin.site.register(Paper)
admin.site.register(PaperWithResearcher)
admin.site.register(PaperImage)
admin.site.register(PaperWithGeneratedImage)
admin.site.register(Mask)
admin.site.register(CorruptedImage)
admin.site.register(PaperWithMask)
admin.site.register(PaperWithCorruptedImage)