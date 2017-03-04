import pytest
import time
from hsreplaynet.cards.models import Deck
from hearthstone.enums import BnetGameType
from scripts.detect_archetype import DeckClassifier

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
	"BRM_002",
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


def test_archetype_classification():
	def get_match(deck, archetype):
		match_ = 0
		for card in deck:
			c = classifier.card_db[card]
			if c in archetype:
				match_ += 1
		return match_

	global classifier
	for test_deck in (
		freeze_mage_variation_full_deck, freeze_mage_variation_short_partial_deck, tempo_mage_variation_partial_deck):

		# First try a full deck
		start_time1 = time.time()
		predicted_deck, confidence = classifier.predict_update([test_deck], "MAGE")
		end_time1 = time.time()
		duration_sec1 = end_time1 - start_time1
		print("speed result:", duration_sec1 <= .1)
		test_deck = set(test_deck) # should be using classifier.vectorizer
		match = get_match(test_deck, predicted_deck)
		print("prediction results:", float(match) / len(test_deck))


classifier = DeckClassifier()
train_data_path = "/home/vagrant/hsreplay.net/scripts/train_decks.csv"  # TODO reload from state
loaded_data = classifier.load_train_data_from_file(train_data_path)
vectorizer = classifier.classifier_state['vectorizer']
archetypes = classifier.fit_transform({'MAGE': loaded_data['MAGE']})
canonical_decks = classifier.calculate_canonical_decks(vectorizer)
test_archetype_classification()
