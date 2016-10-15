"""A Management Command That Expects An InfluxQL Expression.

The output can be piped into an inputs.txt file for easy processing via EMR. E.g.

SELECT replay_xml FROM cards_played WHERE time > now() - 1m AND card_id = 'OG_101';
"""
from django.conf import settings
from django.core.management.base import BaseCommand
from hsreplaynet.utils.influx import influx


class Command(BaseCommand):
	def add_arguments(self, parser):
		parser.add_argument(
			"query",
			help="An InfluxQL query that must contain a replay_xml field."
		)

	def handle(self, *args, **options):
		query = options["query"]
		rs = influx.query(query)

		for point in rs.get_points():
			print("%s:%s" % (settings.AWS_STORAGE_BUCKET_NAME, point["replay_xml"]))
