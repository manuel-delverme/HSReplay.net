from django.conf import settings
from .clients import LAMBDA, KINESIS, S3


def get_kinesis_stream_arn_from_name(name):
	stream = KINESIS.describe_stream(
		StreamName=name,
	)
	if stream:
		return stream["StreamDescription"]["StreamARN"]


def publish_raw_upload_to_processing_stream(raw_upload):
	return KINESIS.put_record(
		StreamName=settings.KINESIS_UPLOAD_PROCESSING_STREAM_NAME,
		Data=raw_upload.kinesis_data,
		PartitionKey=raw_upload.kinesis_partition_key,
	)


def publish_raw_upload_batch_to_processing_stream(raw_uploads):
	records = [dict(Data=r.kinesis_data, PartitionKey=r.kinesis_partition_key)
			   for r in raw_uploads]

	return KINESIS.put_records(
		Records=records,
		StreamName=settings.KINESIS_UPLOAD_PROCESSING_STREAM_NAME,
	)


def get_processing_stream_max_writes_per_second():
	stream = KINESIS.describe_stream(
		StreamName=settings.KINESIS_UPLOAD_PROCESSING_STREAM_NAME,
	)
	num_shards = len(stream["StreamDescription"]["Shards"])
	best_case_throughput = (num_shards * 1000)
	safety_limit = .9
	return best_case_throughput * safety_limit


def is_processing_disabled():
	current_configuration = S3.get_bucket_notification_configuration(
		Bucket=settings.S3_RAW_LOG_UPLOAD_BUCKET
	)

	has_notification_configs = "LambdaFunctionConfigurations" in current_configuration
	if not has_notification_configs:
		return False

	lambda_notifications = current_configuration["LambdaFunctionConfigurations"]
	return (len(lambda_notifications) == 0)


def enable_processing_raw_uploads():
	prod_processing_lambda = LAMBDA.get_function(
		FunctionName="ProcessS3CreateObjectV1",
		Qualifier="PROD"
	)
	prod_notification_config = {
		"LambdaFunctionArn": prod_processing_lambda["Configuration"]["FunctionArn"],
		"Events": ["s3:ObjectCreated:*"],
		"Id": "TriggerProdLambdaOnLogCreate",
		"Filter": {
			"Key": {
				"FilterRules": [
					{"Name": "suffix", "Value": "power.log"},
					{"Name": "prefix", "Value": "raw"},
				]
			}
		}
	}

	canary_processing_lambda = LAMBDA.get_function(
		FunctionName="ProcessS3CreateObjectV1",
		Qualifier="CANARY"
	)
	canary_notification_config = {
		"LambdaFunctionArn": canary_processing_lambda["Configuration"]["FunctionArn"],
		"Events": ["s3:ObjectCreated:*"],
		"Id": "TriggerCanaryLambdaOnLogCreate",
		"Filter": {
			"Key": {
				"FilterRules": [
					{"Name": "suffix", "Value": "canary.log"},
					{"Name": "prefix", "Value": "raw"},
				]
			}
		}
	}

	S3.put_bucket_notification_configuration(
		Bucket=settings.S3_RAW_LOG_UPLOAD_BUCKET,
		NotificationConfiguration={
			"LambdaFunctionConfigurations": [
				prod_notification_config,
				canary_notification_config
			]
		}
	)


def disable_processing_raw_uploads():
	# Remove any existing event notification rules by
	# putting an empty configuration on the bucket
	S3.put_bucket_notification_configuration(
		Bucket=settings.S3_RAW_LOG_UPLOAD_BUCKET,
		NotificationConfiguration={}
	)


def list_all_objects_in(bucket, prefix=None):
	list_response = S3.list_objects_v2(Bucket=bucket, Prefix=prefix)
	if list_response["KeyCount"] > 0:
		objects = list_response["Contents"]
		while objects:
			yield objects.pop(0)
			if list_response["IsTruncated"] and not objects:
				list_response = S3.list_objects_v2(
					Bucket=bucket,
					Prefix=prefix,
					ContinuationToken=list_response["NextContinuationToken"]
				)
				objects += list_response["Contents"]
