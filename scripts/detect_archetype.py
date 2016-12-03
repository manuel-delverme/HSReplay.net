import collections
import sys
import csv
import numpy as np
from hearthstone import cardxml
from sklearn.cluster import DBSCAN
from sklearn.decomposition import LatentDirichletAllocation


class DeckClassifier(object):
	def __init__(self, config: dict = None, min_samples: int = 1) -> None:
		"""

		Args:
			config: dictionary like object
			min_samples: scaling factor to control how many times a deck has to be seen before being considered
		"""

		self.CLASSIFIER_CACHE = "klass_classifiers.pkl"
		self.card_db, _ = cardxml.load()

		self.dimension_to_card_name = {}

		if config:
			min_samples = config['min_samples_modifier']

		self.classifier_state = {
			'min_samples_mod': min_samples,
		}

	# sk-learn API
	def fit(self, data: dict) -> None:
		"""

		Args:
			data: dict of (n_decks, n_dims) np.ndarray, keys are class names, values are the decks
		"""
		samples_mod = self.classifier_state['min_samples_mod']

		classifier = {}
		labels = {}
		for klass in data.keys():
			classifier[klass], labels[klass] = self.train_classifier(data[klass], samples_mod, klass)

		self.classifier_state['classifier'] = classifier
		self.classifier_state['labels'] = labels
		self.classifier_state['vectorizer'] = self.dimension_to_card_name

	def predict(self, deck: np.ndarray, klass: str) -> (list, np.ndarray):
		"""

		Args:
			deck: np.ndarray (1, n_features), a deck
			klass: class of the deck

		Returns:
			a list of cards (archetype), confidence scores for the various class archetypes

		"""
		x = self.deck_to_vector([deck], klass)
		classifier = self.classifier_state['classifier'][klass]
		confidence = classifier.predict(x)
		index = confidence.argmax()
		canonical_deck = self.canonical_decks[klass][index]
		return canonical_deck, confidence

	@staticmethod
	def load_decks_from_file(file_name: str) -> (dict, list):
		"""

		Args:
			file_name: path to file to load
			require_complete: if True; drops decks with size != 30

		Returns: dict[klass] = decklist

		"""
		decks = collections.defaultdict(list)
		with open(file_name, 'r') as f:
			hsreplay_format = False
			csvreader = csv.reader(f)

			for entry in csvreader:
				if entry == ['deck_id', 'player_class', 'card_list', 'card_ids']:
					hsreplay_format = True
					continue

				if hsreplay_format:
					klass, deck = entry[1], entry[3]
					if klass[-1].islower():
						klass = klass[:-1]
						deck = deck.split(", ")
				else:
					if isinstance(entry, list):
						klass, entry[0] = entry[0].split(":")
						deck = entry
					else:
						klass, deck = entry.strip().split(":")
						deck = deck.split(", ")

				if deck == "None":
					continue
				if "None" in deck:
					continue
				decks[klass].append(deck)
		return dict(decks)

	def deck_to_vector(self, decks: list, klass: str) -> np.ndarray:
		klass_data = []
		ignored_cards = 0
		for deck in decks:
			datapoint = np.zeros(len(self.dimension_to_card_name[klass]), dtype=np.float)
			for card in deck:
				try:
					card_dimension = self.dimension_to_card_name[klass].index(card)
					card_value = 1.0 / self.card_db[card].max_count_in_deck
				except ValueError:
					ignored_cards += 1
					continue
				if isinstance(deck, list):
					datapoint[card_dimension] += card_value
				else:
					datapoint[card_dimension] = deck[card]
			klass_data.append(datapoint)
		data = np.array(klass_data)
		if ignored_cards > 0:
			print("[{}] {} cards were ignored when vectorizing".format(klass, ignored_cards))
		if len(data.shape) == 1:
			return data.reshape(1, -1)
		else:
			return data

	def load_train_data_from_file(self, file_name, require_complete=True):
		decks = self.load_decks_from_file(file_name)

		data = {}
		dropped = 0
		dropped_cards = set()
		decks_in_file = 0
		for klass in decks.keys():
			cards_seen_for_klass = set()
			for deck in decks[klass]:
				decks_in_file += 1
				filtered_deck = []
				for card_id in deck:
					card = self.card_db[card_id]
					if card.collectible:  # and card.card_class not in (Neutral, klass)
						cards_seen_for_klass.add(card_id)
						filtered_deck.append(card_id)
					else:
						dropped_cards.add(card)
				if require_complete and len(filtered_deck) != 30:
					dropped += 1
					continue
				cards_seen_for_klass.update(filtered_deck)
			self.dimension_to_card_name[klass] = list(cards_seen_for_klass)
			data[klass] = self.deck_to_vector(decks[klass], klass)
		print("[{}] dropped {} cards as not collectible:".format(klass, len(dropped_cards)))
		return data

	def train_classifier(self, data, min_samples_mod, klass):
		if len(data) == 0:
			print("no data; skipping", klass)
		print("training:", klass)
		min_samples = self.fit_clustering_parameters(data)

		min_samples = int(min_samples_mod * min_samples)
		print("min_samples:", min_samples)

		labels, classifier = self.label_data(data, min_samples, klass)
		return classifier, labels

	def load_classifier_state(self, state):
		self.classifier_state = state

	def label_data(self, data, min_samples, klass):
		def find_nr_archetypes(x, min_samples_):
			dbscan = DBSCAN(eps=1, min_samples=min_samples_, metric='manhattan')
			dbscan.fit(x)
			dbscan.labels_.reshape(-1, 1)
			n_archetypes = dbscan.labels_.max() + 1
			return n_archetypes

		num_core_decks = find_nr_archetypes(data, min_samples)
		model = LatentDirichletAllocation(n_topics=num_core_decks, max_iter=500, evaluate_every=20,
		                                  learning_method="batch")

		classification_results = model.fit_transform(data)
		pArchetype_Card = model.components_

		topcards = 15
		self.classifier_state['canonical_decks'][klass] = []  # klass SHOULD NOT BE HERE TODO: refactor!
		for archetype_index, archetype_card_dist in enumerate(pArchetype_Card):
			self.classifier_state['canonical_decks'][klass].append([])
			archetype_card_ids = np.argsort(archetype_card_dist)[:-(topcards + 1): -1]
			print("topic {}:".format(archetype_index))
			for card_dim in archetype_card_ids:
				card_id = self.dimension_to_card_name[klass][card_dim]
				card_title = self.card_db[card_id]
				print("{}\t".format(card_title), end="")
				self.classifier_state['canonical_decks'][klass][archetype_index].append(card_title)
			print("")

		model.predict = model.transform  # TODO: wtf!
		# return model.labels_, model
		return classification_results, model

	def get_canonical_decks(self, data, transform, labels, lookup):
		transformed_data = False
		if data.shape[1] > self.PCA_DIMENSIONS:
			data = transform.transform(data)
			transformed_data = True
		canonical_decks = {}
		mask = np.ones_like(labels, dtype=bool)
		for label in set(labels):
			mask = labels == label
			centroid = np.average(data[mask], axis=0)
			if transformed_data:
				avg_deck = transform.inverse_transform(centroid)
			else:
				avg_deck = centroid
			card_indexes = reversed(avg_deck.argsort()[-30:])
			canonical_deck = []
			for index in card_indexes:
				if len(canonical_decks) <= 30:
					try:
						card_name = self.card_db[lookup[index]]
					except KeyError:
						pass
					canonical_deck.append(card_name.name + " " + str(int(avg_deck[index] * 100) / 100))
			canonical_decks[label] = canonical_deck
		return canonical_decks

	@staticmethod
	def fit_clustering_parameters(data):
		min_samples = int(len(data) * 7.5 / 100.0)  # atleast {}% of the decks
		return min_samples


def main():
	dataset_path = sys.argv[1]
	results_path = sys.argv[2]
	map_path = sys.argv[3]
	train_data_path = "train_decks.csv"  # TODO reload from state

	classifier = DeckClassifier()
	loaded_data = classifier.load_train_data_from_file(train_data_path)
	classifier.fit(loaded_data)

	del loaded_data

	decks = classifier.load_decks_from_file(dataset_path)
	print("done")
	return
	with open(results_path, 'w') as results:
		results_writer = csv.writer(results)
		for klass, klass_decks in decks.items():
			klass = int(''.join(filter(str.isdigit, klass)))
			hero_to_class = ['UNKNOWN', 'WARRIOR', 'SHAMAN', 'ROGUE', 'PALADIN',
			                 'HUNTER', 'DRUID', 'WARLOCK', 'MAGE', 'PRIEST']
			klass = hero_to_class[klass]
			results = []
			for deck in klass_decks:
				predicted_deck, prob = classifier.predict(deck, klass)
				archetype_number = prob.argmax()
				results_writer.writerow([archetype_number] + deck)

	with open(map_path, 'w') as archetype_map:
		map_writer = csv.writer(archetype_map)
		for klass, archetypes in classifier.canonical_decks.items():
			for i, archetype in enumerate(archetypes):
				map_writer.writerow([klass, i] + [card.name for card in archetype])
	print("done")


if __name__ == '__main__':
	main()
