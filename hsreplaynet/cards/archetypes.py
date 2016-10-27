from collections import defaultdict
from hearthstone.enums import Race, CardClass, FormatType
from .models import Archetype


def edit_distance(canonical_list, unclassified_deck):
	"""Determines the edit distance to transform the unclassified deck into the canonical"""
	UNREVEALED = "*"
	UNREVEALED_PENALTY = .5
	DELETE_PENALTY = 1.0
	INSERT_PENALTY = 1.0

	unrevealed_deck = [UNREVEALED for i in range(30 - len(unclassified_deck))]
	normalized_deck = unclassified_deck.card_id_list() + unrevealed_deck
	distance = 0.0

	canonical_copy = list(canonical_list.card_id_list())
	for card in normalized_deck:
		if card == UNREVEALED:
			distance += UNREVEALED_PENALTY
		elif card in canonical_copy:
			canonical_copy.pop(canonical_copy.index(card))
		else:
			distance += DELETE_PENALTY

	distance += (len(canonical_copy) * INSERT_PENALTY)

	return distance


def classify_deck(
	unclassified_deck,
	player_class=CardClass.INVALID,
	format=FormatType.FT_UNKNOWN
):
	""" Return an Archetype or None

	Classification proceeds in two steps:

	1) First a set of explicit rules is executed, if the deck matches against any of these
	rules, then the Archetype is automatically assigned.

	2) Second, if no Archetype was discovered than an Archetype was assigned by determining
	the minimum edit distance to an existing Archetype.

	However, if the deck is not within at least 5 cards from an Archetype then no Archetype
	will be assigned.
	"""

	candidates = Archetype.objects.archetypes_for_class(player_class, format)

	distances = []
	# 5 divergent cards will require 10 units of distance
	# 5 units for the deletes and 5 for the inserts
	# Likewise a partial deck of 10 or more cards that perfectly matches an Archetype
	# Will make the cutoff since 20 UNREVEALED cards will also have a distance of 10
	CUTOFF_DISTANCE = 10
	for archetype, canonical_deck in candidates.items():
		dist = edit_distance(canonical_deck, unclassified_deck)
		if dist <= CUTOFF_DISTANCE:
			distances.append((archetype, dist))

	if distances:
		return sorted(distances, key=lambda t: t[1])[0][0]
	else:
		return None

# ##### ALL CODE BELOW IS PART OF THE EXPERT SYSTEM MODEL FOR CLASSIFICATION ##### #

MALYGOS = "EX1_563"
CURATOR = "KAR_061"
THUNDERBLUFF_VALIANT = "AT_049"
SECRETKEEPER = "EX1_080"
HIGHMANE = "EX1_534"
MANA_WYRM = "NEW1_012"
ICE_BLOCK = "EX1_295"
DARKSHIRE = "OG_109"
SILVERWARE_GOLEM = "KAR_205"
MALCH_IMP = "KAR_089"
TENTACLES = "OG_114"
POWER_OVERWHELM = "EX1_316"
CTHUN = "OG_280"
NZOTH = "OG_133"
GADZET = "EX1_095"
BLUEGILL = "CS2_173"
VIOLET_TEACHER = "NEW1_026"
MENAGERIE_WARDEN = "KAR_065"
ONYX_BISHOP = "KAR_204"
ETHEREAL_PEDDLER = "KAR_070"
RENO = "LOE_011"
JUSTICAR = "AT_132"
ELISE = "LOE_079"
DESERT_CAMEL = "LOE_020"


def guess_class(deck):
	class_map = defaultdict(int)

	for include in deck.includes.all():
		card = include.card
		if card.card_class != 0 and card.card_class != 12:
			class_map[card.card_class] += 1

	sorted_cards = sorted(class_map.items(), key=lambda t: t[1], reverse=True)
	if len(sorted_cards) > 0:
		return sorted_cards[0][0]
	else:
		return ""


def count_race(deck, race):
	count = 0
	for include in deck.includes.all():
		card = include.card
		if card.race == race:
			count += 1
	return count


def guess_archetype(deck):
	class_guess = guess_class(deck)
	card_ids = deck.card_id_list()

	if class_guess == CardClass.DRUID:
		if CTHUN in card_ids:
			return "CTHUN_DRUID"
		elif CURATOR in card_ids:
			return "CURATOR_DRUID"
		elif VIOLET_TEACHER in card_ids:
			return "TOKEN_DRUID"
		elif MENAGERIE_WARDEN in card_ids:
			return "BEAST_DRUID"
		elif MALYGOS in card_ids:
			return "MALY_DRUID"
		else:
			return ""
	elif class_guess == CardClass.HUNTER:
		if SECRETKEEPER in card_ids:
			return "SECRET_HUNTER"
		elif DESERT_CAMEL in card_ids:
			return "CAMEL HUNTER"
		elif HIGHMANE in card_ids:
			return "MIDRANGE_HUNTER"
		else:
			return ""
	elif class_guess == CardClass.MAGE:
		if ICE_BLOCK in card_ids:
			return "FREEZE_MAGE"
		elif MANA_WYRM in card_ids:
			return "TEMPO_MAGE"
		else:
			return ""
	elif class_guess == CardClass.PALADIN:
		if BLUEGILL in card_ids:
			return "MURLOC_PALADIN"
		else:
			return ""

	elif class_guess == CardClass.PRIEST:
		dragon_count = count_race(deck, Race.DRAGON)
		if ONYX_BISHOP in card_ids:
			return "RESSURECT_PRIEST"
		elif dragon_count >= 4:
			return "DRAGON_PRIEST"
		else:
			return ""
	elif class_guess == CardClass.ROGUE:
		if GADZET in card_ids:
			return "MIRACLE_ROGUE"
		else:
			return ""
	elif class_guess == CardClass.SHAMAN:
		if THUNDERBLUFF_VALIANT in card_ids:
			return "MIDRANGE_SHAMAN"
		else:
			return ""

	elif class_guess == CardClass.WARLOCK:
		if RENO in card_ids:
			return "RENO_LOCK"
		elif DARKSHIRE in card_ids and SILVERWARE_GOLEM in card_ids:
			return "DISCARD_LOCK"
		elif POWER_OVERWHELM in card_ids or TENTACLES in card_ids:
			return "ZOO_LOCK"
		else:
			return ""

	elif class_guess == CardClass.WARRIOR:
		dragon_count = count_race(deck, Race.DRAGON)
		pirate_count = count_race(deck, Race.PIRATE)

		if CTHUN in card_ids:
			return "CTHUN_WARRIOR"
		elif NZOTH in card_ids:
			return "NZOTH_WARRIOR"
		elif dragon_count >= 4:
			return "DRAGON_WARRIOR"
		elif pirate_count >= 4:
			return "PIRATE_WARRIOR"
		elif JUSTICAR in card_ids or ELISE in card_ids:
			return "CONTROL_WARRIOR"
		else:
			return ""
	else:
		return ""
