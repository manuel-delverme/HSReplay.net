import pytest
import os
import json
from datetime import datetime
from django.core.files.storage import default_storage
from hsreplaynet.uploads.models import _generate_upload_key
from hsreplaynet.api.models import APIKey, AuthToken
from hsreplaynet.lambdas.uploads import process_raw_upload
from hsreplaynet.uploads.models import UploadEvent


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(BASE_DIR, "build", "hsreplay-test-data")
UPLOAD_SUITE = os.path.join(DATA_DIR, "hsreplaynet-tests", "uploads")


class MockRawUpload(object):
	def __init__(self, path, storage=None):
		self._descriptor_path = os.path.join(path, "descriptor.json")
		self._descriptor = json.load(open(self._descriptor_path))

		self._powerlog_path = os.path.join(path, "power.log")
		self._log = open(self._powerlog_path).read()

		api_key_str = self._descriptor["gateway_headers"]["X-Api-Key"]
		self._api_key = APIKey.objects.get_or_create(api_key=api_key_str, defaults={
			"full_name": "Test Client",
			"email": "test@example.org",
			"website": "https://example.org",
		})[0]

		auth_token_str = self._descriptor["gateway_headers"]["Authorization"].split()[1]
		self._auth_token = AuthToken.objects.get_or_create(
			key=auth_token_str,
			creation_apikey=self._api_key
		)[0]
		self._api_key.tokens.add(self._auth_token)

		timestamp_str = self._descriptor["upload_metadata"]["match_start"][0:16]
		self._timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M")

		self._shortid = self._descriptor["shortid"]

		if storage:
			key = _generate_upload_key(self._timestamp, self._shortid)
			os.makedirs(os.path.dirname(storage.path(key)), exist_ok=True)
			with storage.open(key, mode="w") as log_file:
				log_file.write(self._log)

		self._upload_event_log_bucket = None
		self._upload_event_log_key = None
		self._reason = None
		self._delete_was_called = False

	@property
	def log(self):
		return self._log

	@property
	def descriptor(self):
		return self._descriptor

	@property
	def api_key(self):
		return self._api_key

	@property
	def auth_token(self):
		return self._auth_token

	@property
	def shortid(self):
		return self._shortid

	@property
	def source_ip(self):
		return self._descriptor["source_ip"]

	@property
	def timestamp(self):
		return self._timestamp

	def player(self, number):
		key = "player%s" % number
		if key in self.descriptor:
			return self.descriptor[key]

		return None

	# Stubs to abstract S3 interactions
	@property
	def bucket(self):
		return "BUCKET"

	@property
	def log_key(self):
		return "LOG_KEY"

	@property
	def upload_http_method(self):
		return "put"

	def prepare_upload_event_log_location(self, bucket, key, descriptor):
		self._upload_event_log_bucket = bucket
		self._upload_event_log_key = key
		self._upload_event_descriptor = descriptor

	def make_failed(self, reason):
		self._reason = reason

	def delete(self):
		self._delete_was_called = True


upload_regression_suite = pytest.mark.skipif(
	not pytest.config.getoption("--all"),
	reason="need --all option to run"
)


@upload_regression_suite
@pytest.mark.django_db
def test_upload_regression_suite():
	if os.path.exists(UPLOAD_SUITE):
		for shortid in os.listdir(UPLOAD_SUITE):
			raw_upload = MockRawUpload(os.path.join(UPLOAD_SUITE, shortid), default_storage)

			# Run first as a create
			do_process_raw_upload(raw_upload, is_reprocessing=False)

			# Then run as a reprocess
			do_process_raw_upload(raw_upload, is_reprocessing=True)
	else:

		assert False, "Upload Suite Does Not Exist On Disk"


def do_process_raw_upload(raw_upload, is_reprocessing):
	process_raw_upload(raw_upload, is_reprocessing)

	# Begin asserting correctness
	created_upload_event = UploadEvent.objects.get(shortid=raw_upload.shortid)
	assert str(created_upload_event.token.key) == str(raw_upload.auth_token.key)
	assert created_upload_event.upload_ip == raw_upload.source_ip

	replay = created_upload_event.game

	validate_fuzzy_date_match(raw_upload.timestamp, replay.global_game.match_start)
	validate_player_data(raw_upload, replay, 1)
	validate_player_data(raw_upload, replay, 2)


def validate_fuzzy_date_match(upload_date, replay_date):
	assert upload_date.year == replay_date.year
	assert upload_date.month == replay_date.month
	assert upload_date.day == replay_date.day


def validate_player_data(raw_upload, replay, number):
	upload_player = raw_upload.player(number)
	if upload_player:
		replay_player = replay.player(number)
		if "rank" in upload_player:
			assert upload_player["rank"] == replay_player.rank

		if "deck" in upload_player:
			assert replay_player.deck_list is not None
			assert len(upload_player["deck"]) == replay_player.deck_list.size()


def test_validate_upload_date():
	"""
	Verifies the upload date / match start validation algorithm.
	The match start is never supposed to be any later than the upload date, so
	if it is, the match start is set to the upload date -- but the timezone
	remains untouched.
	"""

	from aniso8601 import parse_datetime
	from hsreplaynet.games.processing import get_valid_match_start

	values = ((
		# MS greater than UD, same timezone, expecting UD
		"2016-01-01T10:00:00Z",  # Match start
		"2016-01-01T01:01:01Z",  # Upload date
		"2016-01-01T01:01:01Z",  # Expected result
	), (
		# MS lesser than UD, expecting MS
		"2016-01-01T10:00:00+0200",
		"2016-01-01T10:00:00+0100",
		"2016-01-01T10:00:00+0200"
	), (
		# MS greater than UD, different timezone, expecting modified UD
		"2016-01-01T10:00:00+0300",
		"2016-01-01T10:00:00+0400",
		"2016-01-01T09:00:00+0300"
	), (
		# MS greater than UD, different timezone, expecting modified UD
		"2018-01-01T10:00:00-0500",
		"2016-01-01T10:00:00+0500",
		"2016-01-01T00:00:00-0500"
	))

	for match_start, upload_date, expected in values:
		match_start = parse_datetime(match_start)
		upload_date = parse_datetime(upload_date)
		expected = parse_datetime(expected)

		ret = get_valid_match_start(match_start, upload_date)
		# assert expected.tzinfo == match_start.tzinfo
		assert ret.tzinfo == match_start.tzinfo
		assert ret == expected
