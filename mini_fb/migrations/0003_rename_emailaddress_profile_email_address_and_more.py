# Generated by Django 5.1.1 on 2024-10-05 17:17

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mini_fb', '0002_alter_profile_emailaddress'),
    ]

    operations = [
        migrations.RenameField(
            model_name='profile',
            old_name='emailaddress',
            new_name='email_address',
        ),
        migrations.RenameField(
            model_name='profile',
            old_name='firstname',
            new_name='first_name',
        ),
        migrations.RenameField(
            model_name='profile',
            old_name='lastname',
            new_name='last_name',
        ),
        migrations.RenameField(
            model_name='profile',
            old_name='image_url',
            new_name='profile_image_url',
        ),
    ]
