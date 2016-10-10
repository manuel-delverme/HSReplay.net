from functools import wraps
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.views.generic import View
from .models import Feature
from hsreplaynet.utils.instrumentation import error_handler


def view_requires_feature_access(feature_name):
	"""A decorator for view objects that enforces the feature access policies."""
	def decorator(view_func):
		@wraps(view_func)
		def wrapper(arg1, *args, **kwargs):

			if settings.DEBUG:
				# Feature policies are not enforced in development mode
				return view_func(arg1, *args, **kwargs)

			try:
				feature = Feature.objects.get(name=feature_name)

				if issubclass(arg1.__class__, View):
					# If we are using class based views the request is in args
					request = args[0]
				else:
					request = arg1

				is_enabled = feature.enabled_for_user(request.user)

			except Feature.DoesNotExist as e:
				error_handler(e)
				# Missing features are treated as if they are set to
				# FeatureStatus.STAFF_ONLY. This occurs when new feature code is deployed
				# before the DB is updated
				is_enabled = arg1.user.is_staff

			if is_enabled:
				return view_func(arg1, *args, **kwargs)
			else:
				raise PermissionDenied()

		return wrapper

	return decorator
