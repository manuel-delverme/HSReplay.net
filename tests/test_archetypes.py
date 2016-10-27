import pytest
import time
from hsreplaynet.cards.models import Deck
from hearthstone.enums import BnetGameType

freeze_mage_variation_full_deck = [
	"CS2_031",
	"CS2_031",
	"EX1_015",
	"EX1_015",
	"EX1_012",
	"CS2_024",
	"CS2_024",
	"EX1_096",
	"EX1_096",
	"NEW1_021",
	"NEW1_021",
	"CS2_023",
	"CS2_023",
	"LOE_002",
	"LOE_002",
	"EX1_289",
	"EX1_289",
	"EX1_007",
	"EX1_295",
	"EX1_295",
	"CS2_026",
	"CS2_026",
	"CS2_029",
	"CS2_029",
	"CS2_028",
	"CS2_028",
	"BRM_028",
	"CS2_032",
	"EX1_561",
	"EX1_279"
]

freeze_mage_variation_short_partial_deck = [
	"CS2_031",
	"CS2_031",
	"EX1_015",
	"EX1_015",
	"EX1_012",
]

# On average we see 14 cards from the opponents deck list.
tempo_mage_variation_partial_deck = [
	"NEW1_012",
	"NEW1_012",
	"EX1_277",
	"EX1_277",
	"KAR_009",
	"AT_004",
	"AT_004",
	"CS2_024",
	"CS2_024",
	"EX1_608",
	"EX1_608",
	"CS2_025",
	"EX1_012",
	"OG_303",
]


@pytest.mark.django_db
def test_archetype_classification(freeze_mage_archetype, tempo_mage_archetype):

	# First try a full deck
	start_time1 = time.time()
	deck1, created = Deck.objects.get_or_create_from_id_list(
		freeze_mage_variation_full_deck,
		hero_id="HERO_08",
		game_type=BnetGameType.BGT_RANKED_STANDARD,
		classify_into_archetype=True
	)
	end_time1 = time.time()
	duration_sec1 = end_time1 - start_time1
	assert duration_sec1 <= .1
	assert deck1.archetype == freeze_mage_archetype

	# Then assert that we can classify partial decks around our average length
	start_time2 = time.time()
	deck2, created = Deck.objects.get_or_create_from_id_list(
		tempo_mage_variation_partial_deck,
		hero_id="HERO_08",
		game_type=BnetGameType.BGT_RANKED_STANDARD,
		classify_into_archetype=True
	)
	end_time2 = time.time()
	duration_sec2 = end_time2 - start_time2
	assert duration_sec2 <= .1
	assert deck2.archetype == tempo_mage_archetype

	# Finally, check that when we see too few cards we don't classify
	start_time3 = time.time()
	deck3, created = Deck.objects.get_or_create_from_id_list(
		freeze_mage_variation_short_partial_deck,
		hero_id="HERO_08",
		game_type=BnetGameType.BGT_RANKED_STANDARD,
		classify_into_archetype=True
	)
	end_time3 = time.time()
	duration_sec3 = end_time3 - start_time3
	assert duration_sec3 <= .1
	assert deck3.archetype is None
