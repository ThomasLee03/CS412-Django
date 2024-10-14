#tell admin we want to administer these models

from django.contrib import admin

#tell admin we want to administer these models

from .models import Article, Comment

#regist models
admin.site.register(Article)
admin.site.register(Comment)



