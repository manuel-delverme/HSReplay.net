from enum import IntEnum
from math import ceil
from django.conf import settings
from django.db import models
from django.dispatch.dispatcher import receiver
from django.urls import reverse
from hearthstone.enums import BnetGameType, FormatType, PlayState
from hsreplaynet.api.models import AuthToken
from hsreplaynet.cards.models import Card, Deck
from hsreplaynet.utils.fields import IntEnumField, PlayerIDField, ShortUUIDField


def _generate_upload_path(timestamp, shortid):
	ts = timestamp.strftime("%Y/%m/%d/%H/%M")
	return "replays/%s/%s.hsreplay.xml" % (ts, shortid)


def generate_upload_path(instance, filename):
	ts = instance.global_game.match_start
	return _generate_upload_path(ts, instance.shortid)


class GlobalGame(models.Model):
	"""
	Represents a globally unique game (e.g. from the server's POV).

	The fields on this object represent information that is public
	to all players and spectators. When the same game is uploaded
	by multiple players or spectators they will all share a
	reference to a single global game.

	When a replay or raw log file is uploaded the server first checks
	for the existence of a GlobalGame record. It looks for any games
	that occured on the same region where both players have matching
	battle_net_ids and where the match start timestamp is within +/- 1
	minute from the timestamp on the upload.
	The +/- range on the match start timestamp is to account for
	potential clock drift between the computer that generated this
	replay and the computer that uploaded the earlier record which
	first created the GlobalGame record. If no existing GlobalGame
	record is found, then one is created.
	"""

	id = models.BigAutoField(primary_key=True)

	# We believe game_id is not monotonically increasing as it appears
	# to roll over and reset periodically.
	game_handle = models.IntegerField(
		"Game handle",
		null=True, blank=True,
		help_text="Game ID on the Battle.net server"
	)
	server_address = models.GenericIPAddressField(null=True, blank=True)
	server_port = models.IntegerField(null=True, blank=True)
	server_version = models.IntegerField(null=True, blank=True)

	build = models.PositiveIntegerField(
		null=True, blank=True,
		help_text="Hearthstone build number the game was played on."
	)

	match_start = models.DateTimeField(null=True, db_index=True)
	match_end = models.DateTimeField(null=True)

	game_type = IntEnumField("Game type", enum=BnetGameType, null=True, blank=True)
	format = IntEnumField("Format type", enum=FormatType, default=FormatType.FT_UNKNOWN)

	# ladder_season is nullable since not all games are ladder games
	ladder_season = models.IntegerField(
		"Ladder season", null=True, blank=True,
		help_text="The season as calculated from the match start timestamp."
	)

	# Nullable, since not all replays are TBs.
	# Will currently have no way to calculate this so it will always be null for now.
	brawl_season = models.IntegerField(
		"Tavern Brawl season", default=0,
		help_text="The brawl season which increments every time the brawl changes."
	)

	# Nullable, We currently have no way to discover this.
	scenario_id = models.IntegerField(
		"Scenario ID", null=True, blank=True, db_index=True,
		help_text="ID from DBF/SCENARIO.xml or Scenario cache",
	)

	# The following basic stats are globally visible to all
	num_turns = models.IntegerField(null=True, blank=True)
	num_entities = models.IntegerField(null=True, blank=True)

	digest = models.CharField(
		max_length=40, unique=True, null=True, db_index=True,
		help_text="SHA1 of str(game_handle), str(server_address), str(lo1), str(lo2)"
	)

	class Meta:
		ordering = ("-match_start", )

	def __str__(self):
		return " vs ".join(str(p) for p in self.players.all())

	@property
	def duration(self):
		return self.match_end - self.match_start

	@property
	def is_ranked(self):
		return self.game_type in (
			BnetGameType.BGT_RANKED_WILD,
			BnetGameType.BGT_RANKED_STANDARD,
		)

	@property
	def is_casual(self):
		return self.game_type in (
			BnetGameType.BGT_CASUAL_WILD,
			BnetGameType.BGT_CASUAL_STANDARD,
		)

	@property
	def is_tavern_brawl(self):
		return self.game_type in (
			BnetGameType.BGT_TAVERNBRAWL_PVP,
			BnetGameType.BGT_TAVERNBRAWL_1P_VERSUS_AI,
			BnetGameType.BGT_TAVERNBRAWL_2P_COOP,
		)

	@property
	def num_own_turns(self):
		return ceil(self.num_turns / 2)


class GlobalGamePlayer(models.Model):
	id = models.BigAutoField(primary_key=True)
	game = models.ForeignKey(GlobalGame, on_delete=models.CASCADE, related_name="players")

	name = models.CharField("Player name", blank=True, max_length=64, db_index=True)
	real_name = models.CharField("Real name", blank=True, max_length=64, db_index=True)
	user = models.ForeignKey(
		settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
	)

	player_id = PlayerIDField(blank=True)
	account_hi = models.BigIntegerField(
		"Account Hi", blank=True, null=True,
		help_text="The region value from account hilo"
	)
	account_lo = models.BigIntegerField(
		"Account Lo", blank=True, null=True,
		help_text="The account ID value from account hilo"
	)
	is_ai = models.BooleanField(
		"Is AI", default=False,
		help_text="Whether the player is an AI.",
	)
	is_first = models.BooleanField(
		"Is first player",
		help_text="Whether the player is the first player",
	)

	hero = models.ForeignKey(Card, on_delete=models.PROTECT)
	hero_premium = models.BooleanField(
		"Hero Premium", default=False,
		help_text="Whether the player's initial hero is golden."
	)

	final_state = IntEnumField(
		"Final State", enum=PlayState, default=PlayState.INVALID,
	)

	deck_list = models.ForeignKey(
		Deck, on_delete=models.PROTECT,
		help_text="As much as is known of the player's starting deck list."
	)

	# Game type metadata

	rank = models.SmallIntegerField(
		"Rank", null=True, blank=True,
		help_text="1 through 25, or 0 for legend.",
	)
	legend_rank = models.PositiveIntegerField(null=True, blank=True)
	stars = models.PositiveSmallIntegerField(null=True, blank=True)
	wins = models.PositiveIntegerField(
		"Wins", null=True, blank=True,
		help_text="Number of wins in the current game mode (eg. ladder season, arena key...)",
	)
	losses = models.PositiveIntegerField(
		"Losses", null=True, blank=True,
		help_text="Number of losses in the current game mode (current season)",
	)
	deck_id = models.IntegerField("Deck ID", null=True, blank=True)
	cardback_id = models.IntegerField("Cardback ID", null=True, blank=True)

	class Meta:
		unique_together = ("game", "player_id")

	def __str__(self):
		return self.name or self.real_name

	@property
	def won(self):
		return self.final_state in (PlayState.WINNING, PlayState.WON)

	@property
	def opponent(self):
		try:
			if self.player_id == 1:
				return GlobalGamePlayer.objects.get(game=self.game_id, player_id=2)
			else:
				return GlobalGamePlayer.objects.get(game=self.game_id, player_id=1)
		except:
			# Some games where processing failed midway through do not have both players
			return None

	@property
	def hero_class_name(self):
		return self.hero.card_class.name


class Visibility(IntEnum):
	Public = 1
	Unlisted = 2


class GameReplayManager(models.Manager):
	def live(self):
		return self.filter(is_deleted=False)

	def find_by_short_id(self, shortid):
		"""Fetch a game replay by its ShortID

		If the ShortID is an alias it will get resolved and the replay will be returned.

		Returns None if a replay cannot be found.
		"""
		try:
			return GameReplay.objects.get(shortid=shortid, is_deleted=False)
		except GameReplay.DoesNotExist:
			try:
				alias = ReplayAlias.objects.get(shortid=shortid)
				if not alias.replay.is_deleted:
					return alias.replay
			except ReplayAlias.DoesNotExist:
				pass

		return None


class GameReplay(models.Model):
	"""
	Represents a replay as captured from the point of view of a single
	packet stream sent to a Hearthstone client.

	Replays can be uploaded by either of the players or by any number
	of spectators who watched the match. It is possible
	that the same game could be uploaded from multiple points of view.
	When this happens each GameReplay will point
	to the same GlobalGame record via the global_game foreign key.

	It is possible that different uploads of the same game will have
	different information in them.
	For example:
	- If Player A and Player B are Real ID Friends and Player C is
	Battle.net friends with just Player B, then when Player C spectates
	a match between Players A and B, his uploaded replay will show the
	BattleTag as the name of Player A. However if Player B uploads a
	replay of the same match, his replay will show the real name for
	Player A.

	- Likewise, if Player C either starts spectating the game after it has
	already begun or stops spectating before it ends, then his uploaded
	replay will have fewer turns of gameplay then Player B's replay.
	"""
	class Meta:
		ordering = ("global_game", )
		# Replays are unique to a perspective on the game (global_game):
		# - client_handle: the *same* client cant be used for multiple replays
		# - reconnecting: We do not currently unify games where the player reconnects
		# - friendly_player_id: Unique across a client_handle and a spectator_mode
		unique_together = (
			"global_game", "client_handle", "friendly_player_id",
			"spectator_mode", "reconnecting"
		)

	id = models.BigAutoField(primary_key=True)
	shortid = ShortUUIDField("Short ID")
	upload_token = models.ForeignKey(
		AuthToken, on_delete=models.SET_NULL, null=True, blank=True, related_name="replays"
	)
	user = models.ForeignKey(
		settings.AUTH_USER_MODEL,
		on_delete=models.SET_NULL, null=True, blank=True, related_name="replays"
	)
	global_game = models.ForeignKey(
		GlobalGame, on_delete=models.CASCADE, related_name="replays",
		help_text="References the single global game that this replay shows."
	)

	# The "friendly player" is the player whose cards are at the bottom of the
	# screen when watching a game. For spectators this is determined by which
	# player they started spectating first (if they spectate both).
	friendly_player_id = PlayerIDField(
		"Friendly PlayerID",
		null=True, help_text="PlayerID of the friendly player (1 or 2)",
	)

	# This is useful to know because replays that are spectating both players
	# will have more data then those from a single player.
	# For example, they will have access to the cards that are in each players hand.
	# This is detectable from the raw logs, although we currently intend to have
	# the client uploading the replay provide it.
	spectator_mode = models.BooleanField(default=False)
	spectator_password = models.CharField("Spectator Password", max_length=16, blank=True)
	client_handle = models.IntegerField(null=True, blank=True)
	aurora_password = models.CharField(max_length=16, blank=True)

	build = models.PositiveIntegerField("Hearthstone Build", null=True, blank=True)

	replay_xml = models.FileField(upload_to=generate_upload_path)
	hsreplay_version = models.CharField(
		"HSReplay library version",
		max_length=16, help_text="The HSReplay library version used to generate the replay",
	)
	hslog_version = models.CharField(
		"hslog library version",
		max_length=24, help_text="The python-hearthstone library version at processing",
		null=True,  # TODO: Remove this once the table is clean of NULLs
	)

	# The fields below capture the preferences of the user who uploaded it.
	is_deleted = models.BooleanField(
		"Soft deleted",
		default=False, help_text="Indicates user request to delete the upload"
	)

	won = models.NullBooleanField()
	disconnected = models.BooleanField(default=False)
	reconnecting = models.BooleanField(
		"Is reconnecting", default=False,
		help_text="Whether the player is reconnecting to an existing game",
	)
	resumable = models.NullBooleanField()

	visibility = IntEnumField(enum=Visibility, default=Visibility.Public)
	hide_player_names = models.BooleanField(default=False)

	views = models.PositiveIntegerField(default=0)
	objects = GameReplayManager()

	def __str__(self):
		return str(self.global_game)

	@property
	def pretty_name(self):
		return self.build_pretty_name()

	@property
	def pretty_name_spoilerfree(self):
		return self.build_pretty_name(spoilers=False)

	def build_pretty_name(self, spoilers=True):
		players = self.global_game.players.values_list("player_id", "final_state", "name")
		if len(players) != 2:
			return "Broken game (%i players)" % (len(players))
		if players[0][0] == self.friendly_player_id:
			friendly, opponent = players
		else:
			opponent, friendly = players
		if spoilers:
			if self.disconnected:
				state = "Disconnected"
			elif self.won:
				state = "Won"
			elif friendly[1] == opponent[1]:
				state = "Tied"
			else:
				state = "Lost"
			return "%s (%s) vs. %s" % (friendly[2], state, opponent[2])
		return "%s vs. %s" % (friendly[2], opponent[2])

	def get_absolute_url(self):
		return reverse("games_replay_view", kwargs={"id": self.shortid})

	def generate_description(self):
		tpl = "Watch a game of Hearthstone between %s (%s) and %s (%s) in your browser."
		players = self.global_game.players.all()
		player1, player2 = players[0], players[1]
		return tpl % (
			player1, player1.hero.card_class.name.capitalize(),
			player2, player2.hero.card_class.name.capitalize()
		)

	def player(self, number):
		for player in self.global_game.players.all():
			if player.player_id == number:
				return player

	@property
	def friendly_player(self):
		return self.global_game.players.get(player_id=self.friendly_player_id)

	@property
	def opposing_player(self):
		return self.global_game.players.exclude(player_id=self.friendly_player_id).get()

	def related_replays(self, num=3):
		"""
		Returns RelatedReplayRecommendation objects similar to this one.

		The criteria used to generate the recommendations may vary across game types.
		"""
		from hsreplaynet.games.recommendations import recommend_related_replays

		return recommend_related_replays(self, num)


class ReplayAlias(models.Model):
	"""
	Replay aliases are alternative shortids that reference the same GameReplay.

	Each time a power.log is uploaded it is assigned a distinct ShortID. However, if that
	power.log is a duplicate then instead of creating an additional GameReplay, we create
	a ReplayAlias for that ShortID which can be used to lookup the game replay.
	"""
	id = models.BigAutoField(primary_key=True)
	shortid = ShortUUIDField("Short ID")
	replay = models.ForeignKey(GameReplay, on_delete=models.CASCADE, related_name="aliases")


@receiver(models.signals.post_delete, sender=GameReplay)
def cleanup_hsreplay_file(sender, instance, **kwargs):
	from hsreplaynet.utils import delete_file_async
	file = instance.replay_xml
	if file.name:
		delete_file_async(file.name)
