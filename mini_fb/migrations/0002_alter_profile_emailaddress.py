# Generated by Django 5.1.1 on 2024-10-04 19:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mini_fb', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='emailaddress',
            field=models.TextField(),
        ),
    ]
