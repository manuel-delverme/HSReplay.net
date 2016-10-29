"""
Django settings for hsreplay.net project.
"""

import os
import platform
from django.urls import reverse_lazy


##
# Environments
# ENV_LIVE: True if running on *.hsreplay.net
# ENV_LAMBDA: True if running on AWS Lambda
# ENV_AWS: True if running on AWS (ENV_LIVE or ENV_LAMBDA)
# ENV_DEV: True if running on dev.hsreplay.net, or not on LIVE/LAMBDA

HOSTNAME = platform.node()
ENV_LIVE = HOSTNAME.endswith("hsreplay.net")
ENV_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
ENV_AWS = ENV_LIVE or ENV_LAMBDA
ENV_DEV = not ENV_AWS or HOSTNAME == "dev.hsreplay.net"


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(BASE_DIR, "build")

SITE_ID = 1
ROOT_URLCONF = "hsreplaynet.urls"
WSGI_APPLICATION = "wsgi.application"
SECRET_KEY = "be8^qa&f2fut7_1%q@x2%nkw5u=-r6-rwj8c^+)5m-6e^!zags"


if ENV_DEV:
	DEBUG = True
else:
	# Set DEBUG mode only on dev.hsreplay.net
	ALLOWED_HOSTS = [".hsreplay.net"]
	SECRET_KEY = None


# These apps are used on both Lambda and Web
INSTALLED_APPS_CORE = [
	"django.contrib.auth",
	"django.contrib.contenttypes",
	"django.contrib.sessions",
	"django.contrib.messages",
	"django.contrib.staticfiles",
	"django.contrib.sites",
	"raven.contrib.django.raven_compat",
	"rest_framework",
	"hsreplaynet.accounts",
	"hsreplaynet.api",
	"hsreplaynet.cards",
	"hsreplaynet.features",
	"hsreplaynet.games",
	"hsreplaynet.lambdas",
	"hsreplaynet.scenarios",
	"hsreplaynet.uploads",
	"hsreplaynet.utils",
]

# The following apps are not needed on Lambda
INSTALLED_APPS_WEB = [
	"django.contrib.admin",
	"django.contrib.flatpages",
	"django.contrib.humanize",
	"django_comments",
	"allauth",
	"allauth.account",
	"allauth.socialaccount",
	"allauth.socialaccount.providers.battlenet",
	"django_rq",
	"django_rq_dashboard",
	"loginas",
	"webpack_loader",
	"hsreplaynet.admin",
	"hsreplaynet.packs",
]

INSTALLED_APPS = INSTALLED_APPS_CORE
if not ENV_LAMBDA:
	INSTALLED_APPS += INSTALLED_APPS_WEB


MIDDLEWARE_CLASSES = [
	"django.contrib.sessions.middleware.SessionMiddleware",
	"django.middleware.common.CommonMiddleware",
	"django.middleware.csrf.CsrfViewMiddleware",
	"django.contrib.auth.middleware.AuthenticationMiddleware",
	"django.contrib.messages.middleware.MessageMiddleware",
	"django.middleware.clickjacking.XFrameOptionsMiddleware",
	"django.middleware.security.SecurityMiddleware",
	"django.middleware.gzip.GZipMiddleware",
	"hsreplaynet.utils.middleware.DoNotTrackMiddleware",
	"hsreplaynet.utils.middleware.SetRemoteAddrFromForwardedFor",
]


TEMPLATES = [{
	"BACKEND": "django.template.backends.django.DjangoTemplates",
	"DIRS": [
		os.path.join(BASE_DIR, "hsreplaynet", "templates")
	],
	"APP_DIRS": True,
	"OPTIONS": {
		"context_processors": [
			"django.template.context_processors.debug",
			"django.template.context_processors.request",
			"django.contrib.auth.context_processors.auth",
			"django.contrib.messages.context_processors.messages",
		],
	},
}]


##
# Email

SERVER_EMAIL = "admin@hsreplay.net"
DEFAULT_FROM_EMAIL = "contact@hsreplay.net"
EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"


##
# Internationalization

USE_I18N = False
USE_L10N = True
LANGUAGE_CODE = "en-us"

USE_TZ = True
TIME_ZONE = "UTC"


##
# Static files (CSS, JavaScript, Images)

MEDIA_ROOT = os.path.join(BUILD_DIR, "media")
MEDIA_URL = "/media/"

STATIC_ROOT = os.path.join(BASE_DIR, "static")
STATIC_URL = "/static/"

STATICFILES_DIRS = [
	os.path.join(BASE_DIR, "hsreplaynet", "static"),
]
STATICFILES_STORAGE = "django.contrib.staticfiles.storage.ManifestStaticFilesStorage"

WEBPACK_LOADER = {
	"DEFAULT": {
		"BUNDLE_DIR_NAME": "bundles/",
		"STATS_FILE": os.path.join(BUILD_DIR, "webpack-stats.json"),
	}
}

if ENV_AWS:
	DEFAULT_FILE_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"
	# STATIC_URL = "https://static.hsreplay.net/static/"
	AWS_STORAGE_BUCKET_NAME = "hsreplaynet-replays"
else:
	DEFAULT_FILE_STORAGE = "django.core.files.storage.FileSystemStorage"
	AWS_STORAGE_BUCKET_NAME = None


# S3
AWS_S3_USE_SSL = True
AWS_DEFAULT_ACL = "private"

AWS_IS_GZIPPED = True
GZIP_CONTENT_TYPES = [
	"text/css",
	"text/xml",
	"text/plain",
	"application/xml",
	"application/octet-stream",
]


##
# Account/Allauth settings

AUTH_USER_MODEL = "accounts.User"

AUTHENTICATION_BACKENDS = [
	# Needed to login by username in Django admin, regardless of `allauth`
	"django.contrib.auth.backends.ModelBackend",
	# `allauth` specific authentication methods, such as login by e-mail
	"allauth.account.auth_backends.AuthenticationBackend",
]

LOGIN_REDIRECT_URL = reverse_lazy("my_replays")
LOGIN_URL = reverse_lazy("account_login")

ACCOUNT_DEFAULT_HTTP_PROTOCOL = "https"
SOCIALACCOUNT_ADAPTER = "allauth.socialaccount.providers.battlenet.provider.BattleNetSocialAccountAdapter"
SOCIALACCOUNT_PROVIDERS = {"battlenet": {"SCOPE": []}}


##
# Cache (django-redis-cache)
# https://django-redis-cache.readthedocs.io/en/latest/intro_quick_start.html
CACHES = {
	"default": {
		"BACKEND": "redis_cache.RedisCache",
		"LOCATION": "localhost:6379",
	}
}


##
# RQ
# https://github.com/ui/django-rq

RQ_QUEUES = {
	"default": {
		"HOST": "localhost",
		"PORT": 6379,
		"DB": 0,
		"PASSWORD": "",
		"DEFAULT_TIMEOUT": 360,
	}
}


##
# API
# http://www.django-rest-framework.org/api-guide/settings/

REST_FRAMEWORK = {
	# Use Django's standard `django.contrib.auth` permissions,
	# or allow read-only access for unauthenticated users.
	"DEFAULT_PERMISSION_CLASSES": [
		"rest_framework.permissions.IsAuthenticatedOrReadOnly",
	],
	"DEFAULT_PAGINATION_CLASS": "hsreplaynet.api.pagination.DefaultPagination",
}


##
# Django Debug Toolbar (Only on dev)
# https://github.com/jazzband/django-debug-toolbar
# NOTE: This won't work is DEBUG is set to True from local_settings.py

if ENV_DEV:
	INSTALLED_APPS += [
		"debug_toolbar",
	]
	MIDDLEWARE_CLASSES += [
		"debug_toolbar.middleware.DebugToolbarMiddleware",
	]


##
# Custom site settings

HSREPLAY_CAMPAIGN = "utm_source=hsreplay.net&utm_medium=referral&utm_campaign=download"
HDT_DOWNLOAD_URL = "https://hsdecktracker.net/download/?%s" % (HSREPLAY_CAMPAIGN)
HSTRACKER_DOWNLOAD_URL = "https://hsdecktracker.net/hstracker/download/?%s" % (HSREPLAY_CAMPAIGN)
INFLUX_ENABLED = True

# WARNING: To change this it must also be updated in isolated.uploaders.py
S3_RAW_LOG_UPLOAD_BUCKET = "hsreplaynet-uploads"

KINESIS_UPLOAD_PROCESSING_STREAM_NAME = "replay-upload-processing-stream"
KINESIS_UPLOAD_PROCESSING_STREAM_MIN_SHARDS = 1
KINESIS_UPLOAD_PROCESSING_STREAM_MAX_SHARDS = 256

# The target maximum seconds it should take for kinesis to process a backlog of raw uploads
# This value is used to periodically dynamically resize the stream capacity
KINESIS_STREAM_PROCESSING_THROUGHPUT_SLA_SECONDS = 600

LAMBDA_DEFAULT_EXECUTION_ROLE_NAME = "iam_lambda_execution_role"
# Orphan descriptor.json files created this many days previously will be automatically reaped.
LAMBDA_ORPHAN_REAPING_DELAY_DAYS = 3

SUCCESSFUL_UPLOAD_EVENT_REAPING_DELAY_DAYS = 5
UNSUCCESSFUL_UPLOAD_EVENT_REAPING_DELAY_DAYS = 30

JOUST_STATIC_URL = "https://joust.hearthsim.net/branches/master/"
HEARTHSTONEJSON_URL = "https://api.hearthstonejson.com/v1/%(build)s/%(locale)s/cards.json"
HEARTHSTONE_ART_URL = "https://art.hearthstonejson.com/v1/256x/"

HSREPLAY_TWITTER_HANDLE="HSReplayNet"
HSREPLAY_FACEBOOK_APP_ID="1278788528798942"

# This setting controls whether utils.aws.clients are initialized.
# Add `CONNECT_TO_AWS = True` in local_settings.py if you need to use those locally.
CONNECT_TO_AWS = ENV_AWS

ARCHETYPE_CLASSIFICATION_ENABLED = True

# Monkeypatch default collectstatic ignore patterns
from django.contrib.staticfiles.apps import StaticFilesConfig
StaticFilesConfig.ignore_patterns += ["*.scss", "*.ts", "*.tsx", "typings.json"]


try:
	from hsreplaynet.local_settings import *
except ImportError as e:
	# Make sure you have a `local_settings.py` file in the same directory as `settings.py`.
	# We raise a verbose error because the file is *required* in production.
	raise RuntimeError("A `local_settings.py` file could not be found or imported. (%s)" % e)


if __name__ == "__main__":
	# Invoke `python settings.py` to get a JSON dump of all settings
	import json

	print(json.dumps({
		k: v for k, v in globals().items() if (
			k.isupper() and not k.startswith("_") and not k.endswith("_URL")
		)
	}, sort_keys=True, indent="\t"))
