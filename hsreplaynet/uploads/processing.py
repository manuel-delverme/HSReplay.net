"""
A module for scheduling UploadEvents to be processed or reprocessed.
"""
import logging
from django.conf import settings
from hsreplaynet.uploads.models import RawUpload
from hsreplaynet.utils import aws


logger = logging.getLogger(__file__)


def queue_raw_uploads_for_processing(attempt_reprocessing, limit=None):
	"""
	Queue all raw logs to attempt processing them into UploadEvents.

	The primary use for this is for when we deploy code. The intended deploy process is:
		- Notify S3 to suspend triggering lambda upon log upload
		- Perform the Deploy
		- Notify S3 to resume triggering lambda upon log upload
		- Invoke this function to queue for processing any logs uploaded during the deploy

	This method is not intended to requeue uploads that have previously failed.
	For that see the requeue_failed_* family of methods.
	"""
	from hsreplaynet.utils.aws.streams import fill_stream_from_iterable
	logger.info("Starting - Queue all raw uploads for processing")

	publisher_func = aws.publish_raw_upload_batch_to_processing_stream
	iterable = generate_raw_uploads_for_processing(attempt_reprocessing, limit)
	stream_name = settings.KINESIS_UPLOAD_PROCESSING_STREAM_NAME
	fill_stream_from_iterable(stream_name, iterable, publisher_func)


def generate_raw_uploads_for_processing(attempt_reprocessing, limit=None):
	count = 0
	for object in aws.list_all_objects_in(settings.S3_RAW_LOG_UPLOAD_BUCKET, prefix="raw"):
		key = object["Key"]
		if key.endswith(".log"):  # Don't queue the descriptor files, just the .logs
			raw_upload = RawUpload(settings.S3_RAW_LOG_UPLOAD_BUCKET, key)
			raw_upload.attempt_reprocessing = attempt_reprocessing
			yield raw_upload
			count += 1
			if limit and limit >= count:
				return


def current_raw_upload_bucket_size():
	return sum(1 for upload in _list_raw_uploads_by_prefix("raw"))


def _list_raw_uploads_by_prefix(prefix):
	for object in aws.list_all_objects_in(settings.S3_RAW_LOG_UPLOAD_BUCKET, prefix=prefix):
		key = object["Key"]
		if key.endswith(".log"):  # Just emit one message per power.log / canary.log
			yield RawUpload(settings.S3_RAW_LOG_UPLOAD_BUCKET, key)


def _generate_raw_uploads_from_events(events):
	for event in events:
		raw_upload = RawUpload.from_upload_event(event)
		raw_upload.attempt_reprocessing = True
		yield raw_upload


def queue_upload_events_for_reprocessing(events, use_kinesis=False):
	if settings.ENV_AWS or use_kinesis:
		from hsreplaynet.utils.aws.streams import fill_stream_from_iterable
		iterable = _generate_raw_uploads_from_events(events)
		publisher_func = aws.publish_raw_upload_batch_to_processing_stream
		stream_name = settings.KINESIS_UPLOAD_PROCESSING_STREAM_NAME
		fill_stream_from_iterable(stream_name, iterable, publisher_func)
	else:
		for event in events:
			logger.info("Processing UploadEvent %r locally", event)
			event.process()


def queue_upload_event_for_reprocessing(event):
	if settings.ENV_AWS:
		raw_upload = RawUpload.from_upload_event(event)
		raw_upload.attempt_reprocessing = True
		aws.publish_raw_upload_to_processing_stream(raw_upload)
	else:
		logger.info("Processing UploadEvent %r locally", event)
		event.process()
