# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2016-09-07 22:16
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cards', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='deck',
            name='digest',
            field=models.CharField(db_index=True, max_length=32, unique=True),
        ),
    ]
