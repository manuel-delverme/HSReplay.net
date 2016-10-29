from django.contrib import admin
from .models import Card, Deck, Include, Archetype, CanonicalDeck


@admin.register(Card)
class CardAdmin(admin.ModelAdmin):
	list_display = (
		"__str__", "id", "description", "card_set", "rarity", "type",
		"card_class", "artist"
	)
	list_filter = (
		"collectible", "card_set",
		"card_class", "type", "rarity", "cost",
		"battlecry", "deathrattle", "inspire",
		"divine_shield", "taunt", "windfury",
		"overload", "spell_damage"
	)
	search_fields = (
		"name", "description", "id",
	)


class IncludeInline(admin.TabularInline):
	model = Include
	raw_id_fields = ("card", )
	extra = 15


@admin.register(Deck)
class DeckAdmin(admin.ModelAdmin):
	list_display = ("__str__", "archetype", "created")
	inlines = (IncludeInline, )

	def get_ordering(self, request):
		return ["-id"]


class CanonicalDeckInline(admin.TabularInline):
	model = CanonicalDeck
	raw_id_fields = ("deck",)
	extra = 0


@admin.register(Archetype)
class ArchetypeAdmin(admin.ModelAdmin):
	list_display = ("__str__", "player_class_name", "canonical_deck")
	list_filter = ("player_class",)
	inlines = (CanonicalDeckInline,)

	def player_class_name(self, obj):
		return "%s" % obj.player_class.name
	player_class_name.short_description = "Class"
	player_class_name.admin_order_field = "player_class"

	def canonical_deck(self, obj):
		deck = obj.canonical_deck()
		if deck:
			return str(deck)
		else:
			return "Not Set"
	canonical_deck.short_description = "Canonical Deck"

	def get_ordering(self, request):
		return ["player_class", "name"]
