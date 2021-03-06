# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2016-08-10 07:36
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='APIKey',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('full_name', models.CharField(max_length=254)),
                ('email', models.EmailField(max_length=254)),
                ('website', models.URLField(blank=True)),
                ('api_key', models.UUIDField(blank=True)),
                ('enabled', models.BooleanField(default=True)),
            ],
        ),
        migrations.CreateModel(
            name='AuthToken',
            fields=[
                ('key', models.UUIDField(primary_key=True, serialize=False, verbose_name='Key')),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='Created')),
                ('test_data', models.BooleanField(default=False)),
                ('creation_apikey', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='api.APIKey')),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='auth_tokens', to=settings.AUTH_USER_MODEL))
            ],
        ),
        migrations.AddField(
            model_name='apikey',
            name='tokens',
            field=models.ManyToManyField(to='api.AuthToken'),
        ),
    ]
