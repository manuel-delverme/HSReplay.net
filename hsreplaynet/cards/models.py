import hashlib
import pickle
import random
from datetime import datetime

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import models, connection
from hearthstone import enums

from hsreplaynet.utils.fields import IntEnumField


### WORK-IN-PROGRESS - START ###


class ClassifierManager(models.Manager):

	def train_classifier(self, deck_list_iter):
		"""Train a classifier based on the provided list of decks.

		This is the primary entry point for our management command.

		:param deck_list_iter: An iterable of tuples. The first element
		 is the enums.CardClass value to represent the player's class.
		 The second element is a list of card_id strings.
		"""
		# Get the current classifier, in case we are doing online training
		current = Classifier.objects.current()
		new_classifier = self._generate_new_classifier(current, deck_list_iter)

		if new_classifier.is_valid():
			new_classifier.validated = True
			new_classifier._generate_and_update_archetypes()

			if current:
				# May not exist if this is the initial run
				current.retired = datetime.now()
				current.save()
		else:
			new_classifier.regression = True

		new_classifier.save()

	def current(self):
		return Classifier.objects.filter(retired__isnull=True).order_by("-created").first()

	def _generate_new_classifier(self, current, deck_list_iter):
		"""
		An internal method for actually generating the newest classifier

		:param current: (Optional) the soon-to-be-replaced current classifier. Can be None.
		:param deck_list_iter: The iterable of tuples passed into train_classifier()
		"""
		result = Classifier()
		from scripts.detect_archetype import DeckClassifier
		classifier = DeckClassifier(settings.CLASSIFIER_CONFIG)

		# Make sure at the end of training the resulting state is picked and saved to the
		# Classifier.classifier_cache field, e.g.
		classifier_data = classifier.classifier_state
		classifier_pickle = ContentFile(pickle.dumps(classifier_data))
		result.classifier_cache.save("classifier.pickle", classifier_pickle)
		return result

	def classifier_for(self, as_of_date):
		"""
		Provides access to historical classifiers for tagging old replays

		Returns None if one can't be found.
		"""
		if not as_of_date:
			return self.current()
		else:
			return Classifier.objects.filter(
				created__lte=as_of_date,
				validated=True
			).order_by("-created").first()

	def elapsed_minutes_from_previous_training(self):
		current = self.current()
		if current:
			now = datetime.now()
			delta = now - current.created
			return delta.minutes


def _generate_classifier_cache_path(instance, filename):
	ts = instance.created
	timestamp = ts.strftime("%Y/%m/%d/%H/%M")
	return "classifiers/%s/classifier.pkl" % timestamp


class Classifier(models.Model):
	id = models.BigAutoField(primary_key=True)
	objects = ClassifierManager()
	created = models.DateTimeField(auto_now_add=True)
	retired = models.DateTimeField(null=True)
	parent = models.ForeignKey(
		"Classifier",
		null = True,
		related_name="descendants")
	validated = models.BooleanField(
		default=False,
		help_text="Only true if the parent's CanonicalDeck for each Archetype was "
				  "classified into the same Archetype by this classifier."
	)
	regression = models.BooleanField(
		default=False,
		help_text="Only set True if validation was run and failed",
	)

	# Internal storage of classifier state
	# This file will transparently be stored on S3
	# So it will be reachable from all Lambda functions
	classifier_cache = models.FileField(upload_to=_generate_classifier_cache_path)

	def is_valid(self):
		validation_set = Archetype.objects.filter(in_validation_set=True)
		if validation_set.count() == 0:
			return True

		for archetype in validation_set.all():
			card_id_list = archetype.canonical_deck().card_id_list()
			new_classification = self.classify_deck(card_id_list, archetype.player_class)

			# We anticipate training the classifier every few hours.
			# Each time we train it ~ 90% of the data will be data seen by the previous
			# classifier, and ~ 10% will be new data.

			# We intend the Archetypes we include in the validation set to be so core
			# that the CanonicalList for the previous classifier should not be a
			# discontinuous Archetype in the next classifier. Instead we expect to see
			# the incremental evolution of these core Archetypes captured by the
			# variations in the CanonicalList that each Classifier generates.
			if new_classification.id != archetype.id:
				return False

		return True

	def classify_deck(self, card_list, player_class = None):
		# TODO: This should return an instance of Archetype or None
		# Note, the classifier should select among the approved Archetypes
		candidates = Archetype.objects.filter(approved=True).all()

		# classifier_data = {"weights": []}
		# It will be whatever objects were pickled on Line: 56
		classifier_data = self._retrieve_classifier_weights_from_cache()

		# Determine here which candidate is correct, or None if one can't be found.
		return None

	def _retrieve_classifier_weights_from_cache(self):
		if not hasattr(self,"_current_classifier_data"):
			self.classifier_cache.open(mode="rb")
			classifier = pickle.loads(self.classifier_cache.read())
			self._current_classifier_data = classifier
		return self._current_classifier_data

	def _generate_and_update_archetypes(self):
		# Calling this is only legal if this Classifier is being installed as the current
		assert self.validated

		# First we create an Archetype record for any new Archetypes we've discovered.
		# Each of these Archetypes will need to be reviewed in the Admin panel and labeled.
		for archetype in self._get_newly_discovered_archetypes():
			self._create_new_archetype(archetype)

		# Second we update the CanonicalList for each of the existing Archetypes
		for archetype in self._get_existing_archetypes():
			self._update_archetype(archetype)

	def _get_newly_discovered_archetypes(self):
		# Returns new Archetype models that haven't been saved to the DB yet.
		return []

	def _get_existing_archetypes(self):
		# Returns a list of models from existing Archetpyes that need to be updated.
		return []

	def _create_new_archetype(self, archetype):
		archetype.save()
		self._create_or_update_canonical_list_for_archetype(archetype)
		for race_affiliation in archetype.race_affiliations.all():
			race_affiliation.save()

	def _update_archetype(self, archetype):
		archetype.save()
		self._create_or_update_canonical_list_for_archetype(archetype)
		# Note: racial affiliations don't usually change over time unless
		# One archetype morphs into another. E.g. Handlock -> Deamonlock

	def _create_or_update_canonical_list_for_archetype(self, archetype):
		current_canonical_list = archetype.canonical_deck()
		# Creating a new CanonicalDeck each time we retrain the classifier is how
		# We capture the change overtime in the meta (might be interesting to visualize)

		if current_canonical_list:
			# First, we retire the old canonical list if one exists
			current_canonical_list.current = False
			current_canonical_list.save()

		new_canonical = self._generate_canonical_list(archetype)
		new_canonical.current = True
		new_canonical.save()

	def _generate_canonical_list(self, archetype):
		result = CanonicalDeck(archetype = archetype)
		# TODO: We need to pull the card list out of the clusters and set it
		return result

### WORK-IN-PROGRESS - END ###

class CardManager(models.Manager):
	def random(self, cost=None, collectible=True, card_class=None):
		"""
		Return a random Card.

		Keyword arguments:
		cost: Restrict the set of candidate cards to cards of this mana cost.
		By default will be in the range 1 through 8 inclusive.
		collectible: Restrict the set of candidate cards to the set of collectible cards.
		card_class: Restrict the set of candidate cards to this class.
		"""
		cost = random.randint(1, 8) if cost is None else cost
		cards = super(CardManager, self).filter(collectible=collectible)
		cards = cards.exclude(type=enums.CardType.HERO).filter(cost=cost)

		if card_class is not None:
			cards = [c for c in cards if c.card_class in (0, card_class)]

		if cards:
			return random.choice(cards)

	def get_valid_deck_list_card_set(self):
		if not hasattr(self, "_usable_cards"):
			card_list = Card.objects.filter(collectible=True).exclude(type=enums.CardType.HERO)
			self._usable_cards = set(c[0] for c in card_list.values_list("id"))

		return self._usable_cards

	def get_by_partial_name(self, name):
		"""Makes a best guess attempt to return a card based on a full or partial name."""
		return Card.objects.filter(collectible=True).filter(name__icontains=name).first()


class Card(models.Model):
	id = models.CharField(primary_key=True, max_length=50)
	objects = CardManager()

	name = models.CharField(max_length=50)
	description = models.TextField(blank=True)
	flavortext = models.TextField(blank=True)
	how_to_earn = models.TextField(blank=True)
	how_to_earn_golden = models.TextField(blank=True)
	artist = models.CharField(max_length=255, blank=True)

	card_class = IntEnumField(enum=enums.CardClass, default=enums.CardClass.INVALID)
	card_set = IntEnumField(enum=enums.CardSet, default=enums.CardSet.INVALID)
	faction = IntEnumField(enum=enums.Faction, default=enums.Faction.INVALID)
	race = IntEnumField(enum=enums.Race, default=enums.Race.INVALID)
	rarity = IntEnumField(enum=enums.Rarity, default=enums.Rarity.INVALID)
	type = IntEnumField(enum=enums.CardType, default=enums.CardType.INVALID)

	collectible = models.BooleanField(default=False)
	battlecry = models.BooleanField(default=False)
	divine_shield = models.BooleanField(default=False)
	deathrattle = models.BooleanField(default=False)
	elite = models.BooleanField(default=False)
	evil_glow = models.BooleanField(default=False)
	inspire = models.BooleanField(default=False)
	forgetful = models.BooleanField(default=False)
	one_turn_effect = models.BooleanField(default=False)
	poisonous = models.BooleanField(default=False)
	ritual = models.BooleanField(default=False)
	secret = models.BooleanField(default=False)
	taunt = models.BooleanField(default=False)
	topdeck = models.BooleanField(default=False)

	atk = models.IntegerField(default=0)
	health = models.IntegerField(default=0)
	durability = models.IntegerField(default=0)
	cost = models.IntegerField(default=0)
	windfury = models.IntegerField(default=0)

	spare_part = models.BooleanField(default=False)
	overload = models.IntegerField(default=0)
	spell_damage = models.IntegerField(default=0)

	craftable = models.BooleanField(default=False)

	class Meta:
		db_table = "card"

	@classmethod
	def from_cardxml(cls, card, save=False):
		obj = cls(id=card.id)
		for k in dir(card):
			if k.startswith("_"):
				continue
			# Transfer all existing CardXML attributes to our model
			if hasattr(obj, k):
				setattr(obj, k, getattr(card, k))

		if save:
			obj.save()

		return obj

	def __str__(self):
		return self.name


class DeckManager(models.Manager):
	def get_or_create_from_id_list(
		self,
		id_list,
		hero_id=None,
		game_type=None,
		classify_into_archetype=False
	):
		deck, created = self._get_or_create_deck_from_db(id_list)

		archetypes_enabled = settings.ARCHETYPE_CLASSIFICATION_ENABLED
		if archetypes_enabled and classify_into_archetype and created:
			player_class = self._convert_hero_id_to_player_class(hero_id)
			format = self._convert_game_type_to_format(game_type)
			self.classify_deck_with_archetype(deck, player_class, format)

		return deck, created

	def _get_or_create_deck_from_db(self, id_list):
		if len(id_list):
			# This native implementation in the DB is to reduce the volume
			# of DB chatter between Lambdas and the DB
			cursor = connection.cursor()
			cursor.callproc("get_or_create_deck", (id_list,))
			result_row = cursor.fetchone()
			deck_id = int(result_row[0])
			created_ts = result_row[1]
			digest = result_row[2]
			created = result_row[3]
			cursor.close()
			d = Deck(id=deck_id, created=created_ts, digest=digest)
			return d, created
		else:
			digest = generate_digest_from_deck_list(id_list)
			return Deck.objects.get_or_create(digest=digest)

	def _convert_hero_id_to_player_class(self, hero_id):
		if hero_id:
			return Card.objects.get(id=hero_id).card_class
		return enums.CardClass.INVALID

	def _convert_game_type_to_format(self, game_type):
		# TODO: Move this to be a helper on the enum itself
		STANDARD_GAME_TYPES = [
			enums.BnetGameType.BGT_CASUAL_STANDARD,
			enums.BnetGameType.BGT_RANKED_STANDARD,
		]
		WILD_GAME_TYPES = [
			enums.BnetGameType.BGT_CASUAL_WILD,
			enums.BnetGameType.BGT_RANKED_WILD,
			enums.BnetGameType.BGT_ARENA
		]

		if game_type:
			if game_type in STANDARD_GAME_TYPES:
				return enums.FormatType.FT_STANDARD
			elif game_type in WILD_GAME_TYPES:
				return enums.FormatType.FT_WILD

		return enums.FormatType.FT_UNKNOWN

	def classify_deck_with_archetype(self, deck, player_class, format):
		from hsreplaynet.cards.archetypes import classify_deck
		archetype = classify_deck(deck, player_class, format)
		if archetype:
			deck.archetype = archetype
			deck.save()


def generate_digest_from_deck_list(id_list):
	sorted_cards = sorted(id_list)
	m = hashlib.md5()
	m.update(",".join(sorted_cards).encode("utf-8"))
	return m.hexdigest()


class Deck(models.Model):
	"""
	Represents an abstract collection of cards.

	The default sorting for cards when iterating over a deck is by
	mana cost and then alphabetical within cards of equal cost.
	"""
	id = models.BigAutoField(primary_key=True)
	objects = DeckManager()
	cards = models.ManyToManyField(Card, through="Include")
	digest = models.CharField(max_length=32, unique=True, db_index=True)
	created = models.DateTimeField(auto_now_add=True, null=True, blank=True)
	archetype = models.ForeignKey("Archetype", null=True, on_delete=models.SET_NULL)

	def __str__(self):
		return repr(self)

	def __repr__(self):
		values = self.includes.values("card__name", "count", "card__cost")
		alpha_sorted = sorted(values, key=lambda t: t["card__name"])
		mana_sorted = sorted(alpha_sorted, key=lambda t: t["card__cost"])
		value_map = ["%s x %i" % (c["card__name"], c["count"]) for c in mana_sorted]
		return "[%s]" % (", ".join(value_map))

	def __iter__(self):
		# sorted() is stable, so sort alphabetically first and then by mana cost
		alpha_sorted = sorted(self.cards.all(), key=lambda c: c.name)
		mana_sorted = sorted(alpha_sorted, key=lambda c: c.cost)
		return mana_sorted.__iter__()

	def save(self, *args, **kwargs):
		EMPTY_DECK_DIGEST = "d41d8cd98f00b204e9800998ecf8427e"
		if self.digest != EMPTY_DECK_DIGEST and self.includes.count() == 0:
			# A client has set a digest by hand, so don't recalculate it.
			return super(Deck, self).save(*args, **kwargs)
		else:
			self.digest = generate_digest_from_deck_list(self.card_id_list())
			return super(Deck, self).save(*args, **kwargs)

	@property
	def all_includes(self):
		"""
		Use instead of .includes if you know you will use all of them
		this will prefetch the related cards. (eg. in a deck list)
		"""
		fields = ("id", "count", "deck_id", "card__name")
		return self.includes.all().select_related("card").only(*fields)

	def card_id_list(self):
		result = []

		includes = self.includes.values_list("card__id", "count")
		for id, count in includes:
			for i in range(count):
				result.append(id)

		return result

	def size(self):
		"""
		The number of cards in the deck.
		"""
		return sum(i.count for i in self.includes.all())


class Include(models.Model):
	id = models.BigAutoField(primary_key=True)
	deck = models.ForeignKey(Deck, on_delete=models.CASCADE, related_name="includes")
	card = models.ForeignKey(Card, on_delete=models.PROTECT, related_name="included_in")
	count = models.IntegerField(default=1)

	def __str__(self):
		return "%s x %s" % (self.card.name, self.count)

	class Meta:
		unique_together = ("deck", "card")

class ArchetypeManager(models.Manager):
	def archetypes_for_class(self, player_class, format):
		result = {}

		for archetype in Archetype.objects.filter(player_class=player_class):
			canonical_deck = archetype.canonical_deck(format)
			if canonical_deck:
				result[archetype] = canonical_deck

		return result


class Archetype(models.Model):
	"""
	Archetypes cluster decks with minor card variations that all share the same strategy
	into a common group.

	E.g. 'Freeze Mage', 'Miracle Rogue', 'Pirate Warrior', 'Zoolock', 'Control Priest'
	"""
	id = models.BigAutoField(primary_key=True)
	objects = ArchetypeManager()
	name = models.CharField(max_length=250, blank=True)
	player_class = IntEnumField(enum=enums.CardClass, default=enums.CardClass.INVALID)

	def canonical_deck(self, format=enums.FormatType.FT_STANDARD, as_of=None):
		if as_of is None:
			canonical = self.canonical_decks.filter(
				format=format,
			).order_by('-created').prefetch_related("deck__includes").first()
		else:
			canonical = self.canonical_decks.filter(
				format=format,
				created__lte=as_of
			).order_by('-created').prefetch_related("deck__includes").first()

		if canonical:
			return canonical.deck
		else:
			return None

	def __str__(self):
		return self.name


class CanonicalDeck(models.Model):
	"""The CanonicalDeck for an Archetype is the list of cards that is most commonly
	associated with that Archetype.

	The canonical deck for an Archetype may evolve incrementally over time and is likely to
	evolve more rapidly when new card sets are first released.
	"""
	id = models.BigAutoField(primary_key=True)
	archetype = models.ForeignKey(
		Archetype,
		related_name="canonical_decks",
		on_delete=models.CASCADE
	)
	deck = models.ForeignKey(
		Deck,
		related_name="canonical_for_archetypes",
		on_delete=models.PROTECT
	)
	created = models.DateTimeField(auto_now_add=True)
	format = IntEnumField(enum=enums.FormatType, default=enums.FormatType.FT_STANDARD)
