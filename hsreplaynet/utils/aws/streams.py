import time
import logging
from math import ceil, log, pow
from django.conf import settings
from hsreplaynet.utils.influx import get_avg_upload_processing_seconds
from hsreplaynet.uploads.processing import current_raw_upload_bucket_size
from .clients import KINESIS


logger = logging.getLogger("hsreplaynet")

KINESIS_WRITES_PER_SEC = 1000
KINESIS_MAX_BATCH_WRITE_SIZE = 500
MAX_WRITES_SAFETY_LIMIT = .8


def publish_from_iterable_at_fixed_speed(
	iterable,
	publisher_func,
	max_records_per_second,
	publish_batch_size=1
):
	if max_records_per_second == 0:
		raise ValueError("times_per_second must be greater than 0!")

	finished = False
	while not finished:
		try:
			start_time = time.time()
			records_this_second = 0
			while records_this_second < max_records_per_second:
				# NOTE: If the aggregate data published exceeds 1MB / second there will
				# be write throughput failures.
				# As Of 9-20-16 the average record was ~ 180 Bytes
				# 180 Bytes * 1000 = 180KB (which is well below the threshold)
				batch = next_record_batch_of_size(iterable, publish_batch_size)
				if batch:
					records_this_second += len(batch)
					publisher_func(batch)
				else:
					finished = True

			if not finished:
				elapsed_time = time.time() - start_time
				sleep_duration = 1 - elapsed_time
				if sleep_duration > 0:
					time.sleep(sleep_duration)
		except StopIteration:
			finished = True


def next_record_batch_of_size(iterable, max_batch_size):
	result = []
	count = 0
	record = next(iterable, None)
	while record and count < max_batch_size:
		result.append(record)
		count += 1
		record = next(iterable, None)


def fill_stream_from_iterable(stream_name, iterable, publisher_func):
	"""
	Invoke func on the next item from iter at the maximum throughput the stream supports.
	"""
	stream_size = current_stream_size(stream_name)
	max_transactions_per_sec = stream_size * KINESIS_WRITES_PER_SEC
	target_writes_per_sec = ceil(max_transactions_per_sec * MAX_WRITES_SAFETY_LIMIT)
	logger.info(
		"About to fill stream %s at a target of %s writes per second" %
		(stream_name, target_writes_per_sec)
	)

	publish_from_iterable_at_fixed_speed(
		iterable,
		publisher_func,
		target_writes_per_sec,
		publish_batch_size=KINESIS_MAX_BATCH_WRITE_SIZE
	)


def resize_upload_processing_stream(num_shards=None):
	"""Entry point for periodic job to tune the upload processing stream size.

	If num_shards is not provided this method will use the settings.SLA value to
	calculate an appropriate number of shards.
	"""
	min_shards = settings.KINESIS_UPLOAD_PROCESSING_STREAM_MIN_SHARDS
	max_shards = settings.KINESIS_UPLOAD_PROCESSING_STREAM_MAX_SHARDS
	stream_name = settings.KINESIS_UPLOAD_PROCESSING_STREAM_NAME

	if num_shards:
		if num_shards == 0:
			raise ValueError("Will not resize. num_shards must be larger than 0.")

		if not is_base_two_compatible(num_shards):
			raise ValueError("Will not resize. num_shards must be a power of 2.")

		new_shards_number = min(max_shards, max(min_shards, num_shards))
		resize_stream_to_size(stream_name, new_shards_number)
	else:
		sla_seconds = settings.KINESIS_STREAM_PROCESSING_THROUGHPUT_SLA_SECONDS
		num_records = current_raw_upload_bucket_size()
		processing_duration = get_avg_upload_processing_seconds()

		resize_stream(
			stream_name,
			num_records,
			processing_duration,
			sla_seconds,
			min_shards,
			max_shards
		)


def resize_stream(
	stream_name, backlog_size, processing_duration,
	sla_seconds=600, min_shards=2, max_shards=32
):
	"""Generic logic for dynamically resizing a kinesis stream"""

	logger.info("Resize initiated for stream: %s", stream_name)
	logger.info(
		"Backlog Size: %s, Processing Duration: %s, SLA Seconds: %s",
		backlog_size,
		processing_duration,
		sla_seconds
	)

	minimum_target_shards = shards_required_for_sla(
		backlog_size,
		processing_duration,
		sla_seconds
	)
	logger.info("Minimum required shards to hit SLA is: %s", minimum_target_shards)

	# increase target shards to the next power of 2
	# so that our split and merge operations are easy
	shard_target = base_two_shard_target(minimum_target_shards)
	logger.info("Base two shards required to hit SLA is: %s", shard_target)

	new_shards_number = min(max_shards, max(min_shards, shard_target))
	logger.info("Final new shard number will be: %s", new_shards_number)

	resize_stream_to_size(stream_name, new_shards_number)


def shards_required_for_sla(num_records, processing_duration, sla_seconds):
	"""Calculate how many shards are required to hit the target SLA"""
	# We make sure the inputs are at least 1 to prevent this from returning 0
	safe_num_records = max(1, num_records)
	safe_processing_duration = max(1, processing_duration)
	safe_sla = max(1, sla_seconds)
	return ceil((1.0 * safe_num_records * safe_processing_duration) / safe_sla)


def base_two_shard_target(target):
	assert target >= 1, "Target must be greater than or equal to 1"
	return pow(2, ceil(log(target, 2)))


def wait_for_stream_ready(stream_name):
	# We will wait for up to 1 minute for the stream to become active
	stream_status = current_stream_status(stream_name)
	logger.info("The current stream status is: %s", stream_status)

	attempts = 0
	max_attempts = 15
	while stream_status != "ACTIVE" and attempts < max_attempts:
		time.sleep(4)
		stream_status = current_stream_status(stream_name)
		logger.info("The current stream status is: %s", stream_status)

	if stream_status != "ACTIVE":
		raise Exception("The stream %s never became active!" % stream_name)


def get_open_shards(stream_name):
	shards = generate_shard_list(stream_name)
	open_shards = list(filter(shard_is_open, shards))
	return open_shards


def generate_shard_list(stream_name):
	wait_for_stream_ready(stream_name)
	sinfo = KINESIS.describe_stream(StreamName=stream_name)
	shards = sinfo["StreamDescription"]["Shards"]

	while len(shards) > 0:
		shard = shards.pop(0)
		yield shard
		if len(shards) == 0 and sinfo["StreamDescription"]["HasMoreShards"]:
			sinfo = KINESIS.describe_stream(
				StreamName=stream_name,
				ExclusiveStartShardId=shard["ShardId"]
			)
			shards += sinfo["StreamDescription"]["Shards"]


def current_stream_size(stream_name):
	return len(get_open_shards(stream_name))


def current_stream_status(stream_name):
	sinfo = KINESIS.describe_stream(StreamName=stream_name)
	return sinfo["StreamDescription"]["StreamStatus"]


def is_base_two_compatible(num):
	return log(num, 2).is_integer()


def resize_stream_to_size(stream_name, target_num_shards):
	# We only resize along powers of 2 to make programmatic merging and splitting
	# of the PartitionKey space easy to keep balanced.
	assert is_base_two_compatible(target_num_shards)

	current_size = current_stream_size(stream_name)
	logger.info("The current size is: %s", current_size)

	resizing_iterations = 1
	while target_num_shards != current_size:
		logger.info("Starting resizing iteration %s", resizing_iterations)

		if target_num_shards > current_size:
			logger.info("The shards will be split.")
			split_shards(stream_name)
		else:
			logger.info("The shards will be merged.")
			merge_shards(stream_name)

		resizing_iterations += 1
		current_size = current_stream_size(stream_name)
		logger.info("The current size is: %s", current_size)

	logger.info("The stream is the correct target size. Finished.")


def split_shards(stream_name):
	shards = get_open_shards(stream_name)

	logger.info("There are %s shards to split", len(shards))
	while len(shards):
		next_shard_to_split = shards.pop(0)
		logger.info("The next shard to split is: %s", next_shard_to_split["ShardId"])
		wait_for_stream_ready(stream_name)

		starting_hash_key = int(next_shard_to_split["HashKeyRange"]["StartingHashKey"])
		logger.info("Shard starting hash key: %s", str(starting_hash_key))

		ending_hash_key = int(next_shard_to_split["HashKeyRange"]["EndingHashKey"])
		logger.info("Shard ending hash key: %s", str(ending_hash_key))

		combined_hash_key = starting_hash_key + ending_hash_key
		logger.info("The combined hash key: %s", combined_hash_key)

		split_point_hash_key = '{:.0f}'.format((combined_hash_key / 2))
		logger.info("Shard split point hash key: %s", split_point_hash_key)

		KINESIS.split_shard(
			StreamName=stream_name,
			ShardToSplit=next_shard_to_split["ShardId"],
			NewStartingHashKey=str(split_point_hash_key)
		)
	logger.info("Splitting is complete")


def merge_shards(stream_name):
	shards = get_open_shards(stream_name)
	mergable_shard_tuples = prepare_shards_for_merging(shards)

	logger.info("There are %s merges to perform", len(mergable_shard_tuples))
	while len(mergable_shard_tuples):
		first_shard, second_shard = mergable_shard_tuples.pop(0)
		logger.info(
			"The next two shards to be merged are: (%s, %s)",
			first_shard["ShardId"],
			second_shard["ShardId"]
		)
		wait_for_stream_ready(stream_name)

		KINESIS.merge_shards(
			StreamName=stream_name,
			ShardToMerge=first_shard["ShardId"],
			AdjacentShardToMerge=second_shard["ShardId"]
		)
	logger.info("The merging is complete")


def shard_is_open(s):
	return "EndingSequenceNumber" not in s["SequenceNumberRange"]


def shards_are_mergable(first, second):
	first_end_range = int(first["HashKeyRange"]["EndingHashKey"])
	second_start_range = int(second["HashKeyRange"]["StartingHashKey"])
	return (first_end_range + 1) == second_start_range


def prepare_shards_for_merging(shards):
	logger.info("The shards are being prepared for merging")
	sorted_shards = sorted(shards, key=lambda s: int(s["HashKeyRange"]["EndingHashKey"]))

	less_than_two_shards = len(sorted_shards) < 2
	uneven_number_of_shards = len(sorted_shards) % 2 != 0
	if less_than_two_shards or uneven_number_of_shards:
		# Don't attempt to return a result if we have less than 2 shards
		# Or if we don't have an event number of shards

		if less_than_two_shards:
			logger.info("There are less than 2 shards. No merging will be done.")

		if uneven_number_of_shards:
			logger.info("There are an uneven number of shards. No merging will be done.")

		return []

	result = []
	it = iter(sorted_shards)
	for x in it:
		first_shard = x
		second_shard = next(it)
		assert shards_are_mergable(first_shard, second_shard)
		result.append((first_shard, second_shard))

	return result
