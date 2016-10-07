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
		controller_hero = player.starting_hero.card_id
		game_type = BnetGameType(self._meta.get("game_type", 0)).name

		payload = {
			"measurement": "cards_played_stats",
			"tags": {
				"game_type": game_type,
				"controller_hero": controller_hero,
				"region": player.account_hi,
				"controller_rank": player_meta.get("rank"),
			},
			"fields": {
				"card_id": entity.card_id,
				"turn": self.game.tags.get(GameTag.TURN, 0),
			},
			"time": timestamp.isoformat()
		}

		self._payload.append(payload)

	def write_payload(self, shortid):
		# We include the upload shortid so that we can more accurately target
		# map-reduce jobs to only process replays where the cards of interest
		# were actually played.
		# Populate the payload with it before writing to influx
		for pl in self._payload:
			pl["fields"]["shortid"] = shortid
		influx_write_payload(self._payload)
