# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2016-08-18 23:30
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0004_auto_20160812_1752'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='locale',
            field=models.CharField(default='enUS', help_text="The user's preferred Hearthstone locale for display", max_length=8),
        ),
    ]
