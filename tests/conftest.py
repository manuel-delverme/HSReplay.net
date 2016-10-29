import pytest
from django.core.management import call_command
from base64 import b64encode
from hsreplaynet.cards.models import Deck, Archetype, CanonicalDeck
from hearthstone.enums import CardClass, FormatType


def pytest_addoption(parser):
	parser.addoption(
		"--all",
		action="store_true",
		help="run slower tests not enabled by default"
	)


@pytest.fixture(scope='session')
def django_db_setup(django_db_setup, django_db_blocker):
	with django_db_blocker.unblock():
		call_command("load_cards")


@pytest.mark.django_db
@pytest.yield_fixture(scope="module")
def freeze_mage_archetype():
	freeze_mage = [
		"EX1_561",
		"CS2_032",
		"EX1_295",
		"EX1_295",
		"EX1_012",
		"CS2_031",
		"CS2_031",
		"CS2_029",
		"CS2_029",
		"CS2_023",
		"CS2_023",
		"CS2_024",
		"CS2_024",
		"EX1_096",
		"EX1_096",
		"EX1_015",
		"EX1_015",
		"EX1_007",
		"EX1_007",
		"CS2_028",
		"CS2_028",
		"BRM_028",
		"NEW1_021",
		"NEW1_021",
		"CS2_026",
		"CS2_026",
		"LOE_002",
		"LOE_002",
		"EX1_289",
		"OG_082",
	]

	deck, deck_created = Deck.objects.get_or_create_from_id_list(freeze_mage)
	archetype, archetype_created = Archetype.objects.get_or_create(
		name="Freeze Mage",
		player_class=CardClass.MAGE
	)
	if archetype_created:
		CanonicalDeck.objects.create(
			archetype=archetype,
			deck=deck,
			format=FormatType.FT_STANDARD
		)
	yield archetype


@pytest.mark.django_db
@pytest.yield_fixture(scope="module")
def tempo_mage_archetype():
	tempo_mage = [
		"CS2_032",
		"KAR_076",
		"KAR_076",
		"EX1_284",
		"EX1_284",
		"CS2_029",
		"CS2_029",
		"OG_303",
		"OG_303",
		"KAR_009",
		"AT_004",
		"AT_004",
		"EX1_277",
		"EX1_277",
		"EX1_012",
		"EX1_298",
		"CS2_033",
		"CS2_033",
		"CS2_024",
		"CS2_024",
		"NEW1_012",
		"NEW1_012",
		"BRM_002",
		"BRM_002",
		"OG_207",
		"OG_207",
		"CS2_023",
		"CS2_023",
		"EX1_608",
		"EX1_608",
	]

	deck, deck_created = Deck.objects.get_or_create_from_id_list(tempo_mage)
	archetype, archetype_created = Archetype.objects.get_or_create(
		name="Tempo Mage",
		player_class=CardClass.MAGE
	)
	if archetype_created:
		CanonicalDeck.objects.create(
			archetype=archetype,
			deck=deck,
			format=FormatType.FT_STANDARD
		)
	yield archetype


@pytest.yield_fixture(scope="session")
def upload_context():
	yield None


@pytest.yield_fixture(scope="session")
def upload_event():
	yield {
		"headers": {
			"Authorization": "Token beh7141d-c437-4bfe-995e-1b3a975094b1",
		},
		"body": b64encode('{"player1_rank": 5}'.encode("utf8")).decode("ascii"),
		"source_ip": "127.0.0.1",
	}


@pytest.yield_fixture(scope="session")
def s3_create_object_event():
	yield {
		"Records": [{
			"s3": {
				"bucket": {
					"name": "hsreplaynet-raw-log-uploads",
				},
				"object": {
					"key": "raw/2016/07/20/10/37/hUHupxzE9GfBGoEE8ECQiN/power.log",
				}
			}
		}]
	}
