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
	date_hierarchy = "created"
	inlines = (IncludeInline, )


@admin.register(Archetype)
class ArchetypeAdmin(admin.ModelAdmin):
	list_display = ("__str__", "player_class_name", "canonical_deck")

	def player_class_name(self, obj):
		return "%s" % obj.player_class.name
	player_class_name.short_description = "Class"

	def canonical_deck(self, obj):
		deck = obj.canonical_deck()
		if deck:
			return repr(deck)
		else:
			return "Not Set"
	canonical_deck.short_description = "Canonical Deck"


@admin.register(CanonicalDeck)
class CanonicalDeckAdmin(admin.ModelAdmin):
	pass
