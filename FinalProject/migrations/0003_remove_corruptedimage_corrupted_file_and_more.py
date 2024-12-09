# Generated by Django 5.1.3 on 2024-12-06 23:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('FinalProject', '0002_alter_paper_options_image_uploaded_at_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='corruptedimage',
            name='corrupted_file',
        ),
        migrations.RemoveField(
            model_name='mask',
            name='mask_file',
        ),
        migrations.AddField(
            model_name='corruptedimage',
            name='image_file',
            field=models.ImageField(default=1, upload_to='images/'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='mask',
            name='image_file',
            field=models.ImageField(default=1, upload_to='images/'),
            preserve_default=False,
        ),
    ]