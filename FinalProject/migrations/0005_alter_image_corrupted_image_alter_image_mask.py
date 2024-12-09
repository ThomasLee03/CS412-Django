# Generated by Django 5.1.3 on 2024-12-06 23:59

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('FinalProject', '0004_remove_corruptedimage_image_file_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='corrupted_image',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='corrupted_images', to='FinalProject.corruptedimage'),
        ),
        migrations.AlterField(
            model_name='image',
            name='mask',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='masked_images', to='FinalProject.mask'),
        ),
    ]
