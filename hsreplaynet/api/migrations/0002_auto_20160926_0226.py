# -*- coding: utf-8 -*-
# Generated by Django 1.10.1 on 2016-09-26 02:26
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='apikey',
            name='tokens',
        ),
        migrations.AlterField(
            model_name='authtoken',
            name='creation_apikey',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='tokens', to='api.APIKey'),
        ),
    ]