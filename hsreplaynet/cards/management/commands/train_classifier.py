"""
A management command for training classifiers intended to be run on a period interval.
Whenever this command is run it will dynamically determine how long it's been since the
previous classifier was trained. It will then determine the correct lookback window
such that the classifier trained during this invocation is exposed to --percent_new %
of new decks.
"""
from math import ceil
from django.core.management.base import BaseCommand
from django.db import connection
from hsreplaynet.cards.models import Classifier

# The included game types are: Friendly, Ranked Standard, Casual Standard
# Zero length decks and AI decks are filtered out
QUERY = """
	SELECT
		c.card_class AS "player_class", (
			SELECT string_agg(
				CASE WHEN ci.count > 1 THEN
					left(repeat(concat(ci.card_id, ', '), ci.count), -2)
				ELSE
					ci.card_id END, ', '
				) AS "deck_list"
			FROM cards_deck cd
			JOIN cards_include ci ON ci.deck_id = cd.id
			WHERE cd.id = gp.deck_list_id
		) AS "deck_list"
	FROM games_globalgameplayer gp
	JOIN games_globalgame gg ON gp.game_id = gg.id
	JOIN card c ON gp.hero_id = c.id
	WHERE gp.is_ai = FALSE
	AND gg.game_type IN (1, 2, 7)
	AND (SELECT count(*) FROM cards_include WHERE deck_id = gp.deck_list_id) > 0
	AND gg.match_start > now()- INTERVAL '%s minutes';
"""

MIN_TRAINING_INTERVAL = 20


class Command(BaseCommand):
	def add_arguments(self, parser):
		parser.add_argument(
			"--percent_new", default=.1, type=float,
			help="The % of new decks which were not provided to the previous classifier"
		)

	def lookback_minutes(self, elapsed, percent_new):
		return ceil(elapsed / percent_new)

	def handle(self, *args, **options):

		elapsed = Classifier.objects.elapsed_minutes_from_previous_training()
		if not elapsed:
			# This is the first time training the classifier
			elapsed = 100

		if elapsed < MIN_TRAINING_INTERVAL:
			self.stdout.write(
				"Cannot train classifier more than every %s minutes" % elapsed
			)
			return

		lookback = self.lookback_minutes(elapsed, options["percent_new"])

		cursor = connection.cursor()
		cursor.execute(QUERY % lookback)
		iter = iter(cursor.fetchall())
		Classifier.objects.train_classifier(iter)
