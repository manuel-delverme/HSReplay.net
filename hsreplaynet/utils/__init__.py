import datetime
import logging
from dateutil.relativedelta import relativedelta
from uuid import UUID
from django.http import Http404
from django.shortcuts import get_object_or_404
from . import checks  # noqa (needed to register the checks)


def get_logger(name="hsreplaynet", level=logging.DEBUG):
	logger = logging.getLogger(name)
	logger.setLevel(level)

	return logger

log = get_logger()


def delete_file(name):
	"""
	Delete a file by name from the default storage
	"""
	from django.core.files.storage import default_storage
	if default_storage.exists(name):
		default_storage.delete(name)


def delete_file_async(name):
	"""
	Enqueue a RQ job to delete a file by name from the default storage
	"""
	from hsreplaynet.utils.redis import job_queue
	job_queue.enqueue(delete_file, name)


def get_client_ip(request):
	"""
	Get the IP of a client from the request
	"""
	x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
	if x_forwarded_for:
		return x_forwarded_for.split(",")[0]
	return request.META.get("REMOTE_ADDR")


def guess_ladder_season(timestamp):
	epoch = datetime.datetime(2014, 1, 1, tzinfo=timestamp.tzinfo)
	epoch_season = 1
	delta = relativedelta(timestamp, epoch)
	months = (delta.years * 12) + delta.months
	return epoch_season + months


def get_uuid_object_or_404(cls, version=4, **kwargs):
	"""
	Helper that validates every kwarg as a valid UUID
	This avoids ValueError when passing a bad hex string
	to get_object_or_404().
	"""
	for k, v in kwargs.items():
		try:
			kwargs[k] = UUID(v, version=version)
		except ValueError:
			raise Http404("Object not found")
	return get_object_or_404(cls, **kwargs)
