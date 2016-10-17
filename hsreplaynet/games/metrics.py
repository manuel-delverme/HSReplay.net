from math import floor
from django.utils.timezone import now
from hearthstone.enums import GameTag, BlockType, BnetGameType
from hearthstone.hslog.export import EntityTreeExporter
from hsreplaynet.utils.influx import influx_write_payload


class InstrumentedExporter(EntityTreeExporter):
	def __init__(self, packet_tree, meta):
		super(InstrumentedExporter, self).__init__(packet_tree)
		self._payload = []
		self._meta = meta

	def handle_block(self, packet):
		super(InstrumentedExporter, self).handle_block(packet)
		if packet.type == BlockType.PLAY:
			entity = self.game.find_entity_by_id(packet.entity)
			self.record_entity_played(entity)

	def record_entity_played(self, entity):
		timestamp = now()
		player = entity.controller
		if not player:
			return
		player_meta = self._meta.get("player%i" % (player.player_id), {})

		if not player.starting_hero:
			return

		game_type = self._meta.get("game_type", 0)
		try:
			game_type = BnetGameType(game_type).name
		except Exception:
			game_type = "UNKNOWN_%s" % (game_type)

		payload = {
			"measurement": "played_card_stats",
			"tags": {
				"game_type": game_type,
				"card_id": entity.card_id,
			},
			"fields": {
				"rank": self.to_rank_bucket(player_meta.get("rank")),
				"mana": self.to_mana_crystals(player),
				"hero": self.to_hero_class(player),
				"region": player.account_hi,
			},
			"time": timestamp.isoformat()
		}

		self._payload.append(payload)

	def to_hero_class(self, player):
		if player.is_ai:
			return "AI"
		elif player.starting_hero.card_id.startswith("HERO_"):
			return player.starting_hero.card_id[0:7]
		else:
			return "OTHER"

	def to_rank_bucket(self, rank):
		if not rank:
			return None
		elif rank == 0:
			return "LEGEND"
		else:
			min = 1 + floor((rank - 1) / 5) * 5
			max = min + 4
			return "%s-%s" % (min, max)

	def to_mana_crystals(self, player):
		return player.tags.get(GameTag.RESOURCES, 0)

	def write_payload(self, replay_xml_path):
		# We include the replay_xml_path so that we can more accurately target
		# map-reduce jobs to only process replays where the cards of interest
		# were actually played.
		# Populate the payload with it before writing to influx
		for pl in self._payload:
			pl["fields"]["replay_xml"] = replay_xml_path
		influx_write_payload(self._payload)
