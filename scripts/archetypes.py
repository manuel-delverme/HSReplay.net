from collections import defaultdict
from hearthstone.enums import Race, CardClass

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
