import json
from hashlib import sha1
from io import StringIO
from dateutil.parser import parse as dateutil_parse
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from hearthstone import __version__ as hslog_version
from hearthstone.enums import CardType, GameTag
from hearthstone.hslog.parser import LogParser
from hearthstone.hslog.export import EntityTreeExporter
from hsreplay import __version__ as hsreplay_version
from hsreplay.document import HSReplayDocument
from hsreplaynet.cards.models import Card, Deck
from hsreplaynet.utils import guess_ladder_season, log
from hsreplaynet.utils.influx import influx_metric
from hsreplaynet.uploads.models import UploadEventStatus
from .metrics import InstrumentedExporter
from .models import (
	GameReplay, GlobalGame, GlobalGamePlayer,
	_generate_upload_path, ReplayAlias
)


class ProcessingError(Exception):
	pass


class ParsingError(ProcessingError):
	pass


class GameTooShort(ProcessingError):
	pass


class UnsupportedReplay(ProcessingError):
	pass


def eligible_for_unification(meta):
	return all([meta.get("game_handle"), meta.get("server_ip")])


def get_valid_match_start(match_start, upload_date):
	"""
	Returns a valid match_start value given the match_start and upload_date.
	If the upload_date is greater than the match_start, return the match_start.
	If it's greater than the match_start, return the upload_date, modified to
	use the match_start's timezone.
	"""
	if upload_date > match_start:
		return match_start

	log.info("match_start=%r>upload_date=%r - rejecting match_start", match_start, upload_date)
	return upload_date.astimezone(match_start.tzinfo)


def create_hsreplay_document(parser, entity_tree, meta, global_game):
	hsreplay_doc = HSReplayDocument.from_parser(parser, build=meta["build"])
	game_xml = hsreplay_doc.games[0]
	game_xml.game_type = global_game.game_type
	game_xml.id = global_game.game_handle
	if meta["reconnecting"]:
		game_xml.reconnecting = True

	for player in entity_tree.players:
		player_meta = meta.get("player%i" % (player.player_id), {})
		player_xml = game_xml.players[player.player_id - 1]
		player_xml.rank = player_meta.get("rank")
		player_xml.legendRank = player_meta.get("legend_rank")
		player_xml.cardback = player_meta.get("cardback")
		player_xml.deck = player_meta.get("deck")

	return hsreplay_doc


def save_hsreplay_document(hsreplay_doc, shortid, existing_replay):
	# Not using get_absolute_url() to avoid tying into Django
	# (not necessarily avail on lambda)
	url = "https://hsreplay.net/replay/%s" % (shortid)

	xml_str = hsreplay_doc.to_xml()
	# Add the replay's full URL as a comment
	xml_str += "\n<!-- %s -->\n" % (url)

	return ContentFile(xml_str)


def generate_globalgame_digest(meta, lo1, lo2):
	game_handle = meta["game_handle"]
	server_address = meta["server_ip"]
	values = (game_handle, server_address, lo1, lo2)
	ret = "-".join(str(k) for k in values)
	return sha1(ret.encode("utf-8")).hexdigest()


def find_or_create_global_game(entity_tree, meta):
	ladder_season = meta.get("ladder_season")
	if not ladder_season:
		ladder_season = guess_ladder_season(meta["end_time"])

	defaults = {
		"game_handle": meta.get("game_handle"),
		"server_address": meta.get("server_ip"),
		"server_port": meta.get("server_port"),
		"server_version": meta.get("server_version"),
		"game_type": meta.get("game_type", 0),
		"format": meta.get("format", 0),
		"build": meta["build"],
		"match_start": meta["start_time"],
		"match_end": meta["end_time"],
		"brawl_season": meta.get("brawl_season", 0),
		"ladder_season": ladder_season,
		"scenario_id": meta.get("scenario_id"),
		"num_entities": len(entity_tree.entities),
		"num_turns": entity_tree.tags.get(GameTag.TURN),
	}

	if eligible_for_unification(meta):
		# If the globalgame is eligible for unification, generate a digest
		# and get_or_create the object
		players = entity_tree.players
		lo1, lo2 = players[0].account_lo, players[1].account_lo
		digest = generate_globalgame_digest(meta, lo1, lo2)
		log.info("GlobalGame digest is %r" % (digest))
		global_game, created = GlobalGame.objects.get_or_create(digest=digest, defaults=defaults)
	else:
		global_game = GlobalGame.objects.create(digest=None, **defaults)
		created = True

	log.debug("Prepared GlobalGame(id=%r), created=%r", global_game.id, created)
	return global_game, created


def find_or_create_replay(parser, entity_tree, meta, upload_event, global_game, players):
	client_handle = meta.get("client_handle") or None
	existing_replay = upload_event.game
	shortid = existing_replay.shortid if existing_replay else upload_event.shortid
	replay_xml_path = _generate_upload_path(global_game.match_start, shortid)
	log.debug("Will save replay %r to %r", shortid, replay_xml_path)

	# The user that owns the replay
	user = upload_event.token.user if upload_event.token else None
	friendly_player = players[meta["friendly_player"]]

	hsreplay_doc = create_hsreplay_document(parser, entity_tree, meta, global_game)

	common = {
		"global_game": global_game,
		"client_handle": client_handle,
		"spectator_mode": meta.get("spectator_mode", False),
		"reconnecting": meta["reconnecting"],
		"friendly_player_id": friendly_player.player_id,
	}
	defaults = {
		"shortid": shortid,
		"aurora_password": meta.get("aurora_password", ""),
		"spectator_password": meta.get("spectator_password", ""),
		"resumable": meta.get("resumable"),
		"build": meta["build"],
		"upload_token": upload_event.token,
		"won": friendly_player.won,
		"replay_xml": replay_xml_path,
		"hsreplay_version": hsreplay_version,
		"hslog_version": hslog_version,
	}

	# Create and save hsreplay.xml file
	# Noop in the database, as it should already be set before the initial save()
	xml_file = save_hsreplay_document(hsreplay_doc, shortid, existing_replay)
	influx_metric("replay_xml_num_bytes", {"size": xml_file.size})

	if existing_replay:
		log.debug("Found existing replay %r", existing_replay.shortid)
		# Clean up existing replay file
		filename = existing_replay.replay_xml.name
		if filename and filename != replay_xml_path and default_storage.exists(filename):
			# ... but only if it's not the same path as the new one (it'll get overwridden)
			log.debug("Deleting %r", filename)
			default_storage.delete(filename)

		# Now update all the fields
		defaults.update(common)
		for k, v in defaults.items():
			setattr(existing_replay, k, v)

		# Save the replay file
		existing_replay.replay_xml.save("hsreplay.xml", xml_file, save=False)

		# Finally, save to the db and exit early with created=False
		existing_replay.save()
		return existing_replay, False

	# No existing replay, so we assign a default user/visibility to the replay
	# (eg. we never update those fields on existing replays)
	if user:
		defaults["user"] = user
		defaults["visibility"] = user.default_replay_visibility

	if client_handle:
		# Get or create a replay object based on our defaults
		replay, created = GameReplay.objects.get_or_create(defaults=defaults, **common)
		log.debug("Replay %r has created=%r, client_handle=%r", replay.id, created, client_handle)
	else:
		# The client_handle is the minimum we require to update an existing replay.
		# If we don't have it, we won't try deduplication, we instead get_or_create by shortid.
		defaults.update(common)
		replay, created = GameReplay.objects.get_or_create(defaults=defaults, shortid=shortid)
		log.debug("Replay %r has created=%r (no client_handle)", replay.id, created)

	# Save the replay file
	replay.replay_xml.save("hsreplay.xml", xml_file, save=False)

	if replay.shortid != upload_event.shortid:
		# We must ensure an alias for this upload_event.shortid is recorded
		# We use get or create in case this is not the first time processing this replay
		ReplayAlias.objects.get_or_create(replay=replay, shortid=upload_event.shortid)

	return replay, created


def handle_upload_event_exception(e):
	"""
	Returns a (status, reraise) tuple.
	The status will be set on the UploadEvent.
	If reraise is True, the exception will bubble up.
	"""
	if isinstance(e, ParsingError):
		return UploadEventStatus.PARSING_ERROR, True
	elif isinstance(e, (GameTooShort, EntityTreeExporter.EntityNotFound)):
		return UploadEventStatus.UNSUPPORTED, False
	elif isinstance(e, UnsupportedReplay):
		return UploadEventStatus.UNSUPPORTED, True
	elif isinstance(e, ValidationError):
		return UploadEventStatus.VALIDATION_ERROR, True
	else:
		return UploadEventStatus.SERVER_ERROR, True


def process_upload_event(upload_event):
	"""
	Wrapper around do_process_upload_event() to set the event's
	status and error/traceback as needed.
	"""
	upload_event.error = ""
	upload_event.traceback = ""
	if upload_event.status != UploadEventStatus.PROCESSING:
		upload_event.status = UploadEventStatus.PROCESSING
		upload_event.save()

	try:
		replay = do_process_upload_event(upload_event)
	except Exception as e:
		from traceback import format_exc
		upload_event.error = str(e)
		upload_event.traceback = format_exc()
		upload_event.status, reraise = handle_upload_event_exception(e)
		upload_event.save()
		if reraise:
			raise
		else:
			return
	else:
		upload_event.game = replay
		upload_event.status = UploadEventStatus.SUCCESS
		upload_event.save()

		if not upload_event.test_data:
			capture_class_distribution_stats(replay)

	return replay


def capture_class_distribution_stats(replay):
	fields = {
		"num_turns": replay.global_game.num_turns,
		"opposing_player_deck_digest": replay.opposing_player.deck_list.digest,
		"friendly_player_deck_digest": replay.friendly_player.deck_list.digest,
	}

	tags = {
		"game_type": replay.global_game.game_type,
		"scenario_id": replay.global_game.scenario_id,
		"region": replay.friendly_player.account_hi,
		"opposing_player_rank": replay.opposing_player.rank,
		"opposing_player_class": replay.opposing_player.hero.card_class.name,
		"opposing_player_final_state": replay.opposing_player.final_state,
		"friendly_player_rank": replay.friendly_player.rank,
		"friendly_player_class": replay.friendly_player.hero.card_class.name,
		"friendly_player_final_state": replay.friendly_player.final_state,
	}

	influx_metric("replay_outcome_stats", fields=fields, **tags)


def parse_upload_event(upload_event, meta):
	orig_match_start = dateutil_parse(meta["match_start"])
	match_start = get_valid_match_start(orig_match_start, upload_event.created)
	if match_start != orig_match_start:
		upload_event.tainted = True
		upload_event.save()

	log_bytes = upload_event.log_bytes()
	if not log_bytes:
		raise ValidationError("The uploaded log file is empty.")
	influx_metric("raw_power_log_upload_num_bytes", {"size": len(log_bytes)})
	powerlog = StringIO(log_bytes.decode("utf-8"))
	upload_event.file.close()

	try:
		parser = LogParser()
		parser._game_state_processor = "GameState"
		parser._current_date = match_start
		parser.read(powerlog)
	except Exception as e:
		log.exception("Got exception %r while parsing log", e)
		raise ParsingError(str(e))  # from e

	return parser


def validate_parser(parser, meta):
	# Validate upload
	if len(parser.games) != 1:
		raise ValidationError("Expected exactly 1 game, got %i" % (len(parser.games)))
	packet_tree = parser.games[0]
	exporter = InstrumentedExporter(packet_tree, meta).export()
	entity_tree = exporter.game

	if len(entity_tree.players) != 2:
		raise ValidationError("Expected 2 players, found %i" % (len(entity_tree.players)))

	for player in entity_tree.players:
		# Set the player's name
		player.name = parser.games[0].manager.get_player_by_id(player.id).name
		if player.name is None:
			# If it's None, this is an unsupported replay.
			log.error("Cannot find player %i name. Replay not supported.", player.player_id)
			raise GameTooShort("The game was too short to parse correctly")

		if not player.heroes:
			raise UnsupportedReplay("No hero found for player %r" % (player.name))
		player._hero = list(player.heroes)[0]

		try:
			db_hero = Card.objects.get(id=player._hero.card_id)
		except Card.DoesNotExist:
			raise UnsupportedReplay("Hero %r not found." % (player._hero))
		if db_hero.type != CardType.HERO:
			raise ValidationError("%r is not a valid hero." % (player._hero))

	if not meta.get("friendly_player"):
		from hearthstone.hslog.export import FriendlyPlayerExporter
		id = packet_tree.export(cls=FriendlyPlayerExporter)
		if not id:
			raise ValidationError("Friendly player ID not present at upload and could not guess it.")
		meta["friendly_player"] = id

	if "reconnecting" not in meta:
		meta["reconnecting"] = False

	# Add the start/end time to meta dict
	meta["start_time"] = packet_tree.start_time
	meta["end_time"] = packet_tree.end_time

	return entity_tree, exporter


def get_player_names(player):
	if not player.is_ai and " " in player.name:
		return "", player.name
	else:
		return player.name, ""


def update_global_players(global_game, entity_tree, meta):
	# Fill the player metadata and objects
	players = {}

	for player in entity_tree.players:
		player_meta = meta.get("player%i" % (player.player_id), {})
		decklist = player_meta.get("deck")
		if not decklist:
			decklist = [c.card_id for c in player.initial_deck if c.card_id]

		name, real_name = get_player_names(player)

		deck, created = Deck.objects.get_or_create_from_id_list(decklist)
		log.debug("Prepared deck %i (created=%r)", deck.id, created)

		common = {
			"game": global_game,
			"player_id": player.player_id,
		}
		defaults = {
			"account_hi": player.account_hi,
			"account_lo": player.account_lo,
			"is_first": player.tags.get(GameTag.FIRST_PLAYER, False),
			"is_ai": player.is_ai,
			"hero_id": player._hero.card_id,
			"hero_premium": player._hero.tags.get(GameTag.PREMIUM, False),
			"final_state": player.tags.get(GameTag.PLAYSTATE, 0),
			"deck_list": deck,
		}

		update = {
			"name": name,
			"real_name": real_name,
			"rank": player_meta.get("rank"),
			"legend_rank": player_meta.get("legend_rank"),
			"stars": player_meta.get("stars"),
			"wins": player_meta.get("wins"),
			"losses": player_meta.get("losses"),
			"deck_id": player_meta.get("deck_id") or None,
			"cardback_id": player_meta.get("cardback"),
		}

		defaults.update(update)
		game_player, created = GlobalGamePlayer.objects.get_or_create(defaults=defaults, **common)
		log.debug("Prepared player %r (%i) (created=%r)", game_player, game_player.id, created)

		if not created:
			# Go through the update dict and update values on the player
			# This gets us extra data we might not have had when the player was first created
			updated = False
			for k, v in update.items():
				if v and getattr(game_player, k) != v:
					setattr(game_player, k, v)
					updated = True

			# Skip updating the deck if we already have a bigger one
			# TODO: We should make deck_list nullable and only create it here
			if len(decklist) > game_player.deck_list.size():
				# XXX: Maybe we should also check friendly_player_id for good measure
				game_player.deck_list = deck
				updated = True

			if updated:
				log.debug("Saving updated player to the database.")
				game_player.save()

		players[player.player_id] = game_player

	return players


def do_process_upload_event(upload_event):
	meta = json.loads(upload_event.metadata)

	# Parse the UploadEvent's file
	parser = parse_upload_event(upload_event, meta)
	# Validate the resulting object and metadata
	entity_tree, exporter = validate_parser(parser, meta)

	# Create/Update the global game object and its players
	global_game, created = find_or_create_global_game(entity_tree, meta)
	players = update_global_players(global_game, entity_tree, meta)

	# Create/Update the replay object itself
	replay, created = find_or_create_replay(
		parser, entity_tree, meta, upload_event, global_game, players
	)

	if not upload_event.test_data:
		exporter.write_payload(replay.replay_xml.name)

	return replay
