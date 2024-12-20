# Generated by Django 5.1.3 on 2024-12-07 11:00

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('FinalProject', '0013_remove_imagegenerated_image_file_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagegenerated',
            name='image_file',
            field=models.ImageField(default=20, upload_to='images/generated/'),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='imagegenerated',
            name='image',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='FinalProject.image'),
        ),
    ]
