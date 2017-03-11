import collections
import pickle
import json
import sys
import csv
import numpy as np
from collections import defaultdict

from typing import Tuple, Dict, List
from hearthstone import cardxml
from sklearn.cluster import DBSCAN
from sklearn.decomposition import LatentDirichletAllocation


class vectorizer_1hot(object):
	def __init__(self, card_db):
		self.dimension_to_card_name = {}
		self.max_count_in_deck = {}
		for card in card_db:
			self.max_count_in_deck[card_db[card].id] = card_db[card].max_count_in_deck

	def train_klass(self, klass: str, cards_seen_for_klass) -> None:
		self.dimension_to_card_name[klass] = list(cards_seen_for_klass)

	def invert(self, klass: str, card_dim: int) -> np.ndarray:
		return self.dimension_to_card_name[klass][card_dim]

	def transform(self, klass: str, decks: list) -> np.ndarray:
		klass_data = []
		ignored_cards = 0
		for deck in decks:
			datapoint = np.zeros(len(self.dimension_to_card_name[klass]), dtype=np.float)
			for card in deck:
				try:
					card_dimension = self.dimension_to_card_name[klass].index(card)
					card_value = 1.0 / self.max_count_in_deck[card]
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


class DeckClassifier(object):
	def __init__(self, config: dict = None, min_samples: float = 1.5 / 100) -> None:
		"""

		Args:
			config: dictionary like object
			min_samples: scaling factor to control how many times a deck has to be seen before being considered
		"""
		self.card_db, _ = cardxml.load()

		if config:
			min_samples = config['min_samples_ratio']

		self.classifier_state = {
			'min_samples_ratio': min_samples,
			'vectorizer'       : {},
			'hero_to_class'    : ['UNKNOWN', 'UNKNOWN', 'DRUID', 'HUNTER', 'MAGE', 'PALADIN', 'PRIEST', 'ROGUE',
			                      'SHAMAN', 'WARLOCK', 'WARRIOR']
		}

	# sk-learn API
	def fit_transform(self, data: dict, popular_decks: np.ndarray, spd) -> Dict[str, List[List[int]]]:
		"""

		Args:
			data: dict of (n_decks, n_dims) np.ndarray, keys are class names, values are the decks
		"""
		samples_mod = self.classifier_state['min_samples_ratio']

		classifier = {}
		archetypes = {}
		min_samples = {}

		for klass in data.keys():
			if len(data[klass]) == 0:
				print("no data; skipping", klass)
			print("training:", klass)

			clas_, arch_, min_samp_ = self.train_classifier(data[klass], popular_decks[klass], spd[klass], samples_mod)
			classifier[klass], archetypes[klass] = clas_, arch_
			min_samples[klass] = min_samp_

		self.classifier_state['classifier'] = classifier
		# self.classifier_state['labels'] = archetypes
		self.classifier_state['min_samples'] = min_samples
		return archetypes

	def _update_classifier(self, data: np.ndarray, klass: str) -> None:
		assert len(data) > 0

		min_samples_ratio = self.classifier_state['min_samples_ratio']
		min_samples = self.update_fit_clustering_params(data, min_samples_ratio)

		classifier, archetypes = self.detect_archetypes(data, min_samples)

		num_core_decks = self.find_nr_archetypes(data, min_samples)
		print("detected: {} archetypes".format(num_core_decks))
		model = LatentDirichletAllocation(n_topics=num_core_decks, max_iter=500, evaluate_every=20,
		                                  learning_method="batch")

		if num_core_decks == 0:
			print("NO ARCHETYPES FOUND!")
			classification_results = []
		else:
			classification_results = model.fit_transform(data)

		model.predict = model.transform  # TODO: wtf!

		return classifier, archetypes, min_samples

	# sk-learn API
	def predict_update(self, data: List[str], klass: str) -> Tuple[List[str], List[float]]:
		"""

		Args:
		    klass:
			data:
		"""
		# SETUP
		# canonical_decks = self.classifier_state['canonical_decks']
		# samples_mod = self.classifier_state['min_samples_mod']
		# classifier = self.classifier_state['classifier']
		# labels = self.classifier_state['labels']

		x = self.deck_to_1hot_vector(data, klass, self.classifier_state['vectorizer'][klass])

		# UPDATE
		self._update_classifier(x, klass)

		# CALCULATE
		canonical_deck, confidence = self.predict(x, klass)

		index = confidence.argmax()
		canonical_deck = self.classifier_state['canonical_decks'][klass][index]
		return canonical_deck, confidence

	def predict(self, deck: np.ndarray, klass: str) -> (list, np.ndarray):
		"""

		Args:
			deck: np.ndarray (1, n_features), a deck
			klass: class of the deck

		Returns:
			a list of cards (archetype), confidence scores for the various class archetypes

		"""
		x = self.deck_to_1hot_vector([deck], klass, self.classifier_state['vectorizer'][klass])
		classifier = self.classifier_state['classifier'][klass]
		confidence = classifier.predict(x)
		index = confidence.argmax()
		canonical_deck = self.classifier_state['canonical_decks'][klass][index]
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

	def load_train_data_from_file(self, file_name: str, require_complete: bool = True) -> np.ndarray:
		decks = self.load_decks_from_file(file_name)
		vectorizer = vectorizer_1hot(self.card_db)

		assert len(decks) > 0

		data = {}
		dropped = 0
		dropped_cards = set()
		decks_in_file = 0
		dimension_to_card_name = {}

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
			if len(cards_seen_for_klass) > 0:
				dimension_to_card_name[klass] = list(cards_seen_for_klass)
				data[klass] = vectorizer.transform(decks[klass], klass, dimension_to_card_name[klass])
			else:
				print("[{}] ignored, had no valid cards".format(klass))

		print("[{}] dropped {} cards as not collectible:".format(klass, len(dropped_cards)))
		self.classifier_state['vectorizer'] = vectorizer
		return data

	def train_classifier(self, data: np.ndarray, popular_decks: np.ndarray, spd, min_samples_ratio: int) -> Tuple[
		LatentDirichletAllocation, list, int]:
		assert len(data) > 0
		classifier, archetypes = self.detect_archetypes(data, popular_decks, spd)

		return classifier, archetypes, 0

	@staticmethod
	def find_nr_archetypes(x):
		dbscan = DBSCAN(eps=4, min_samples=1, metric='manhattan', algorithm='ball_tree')
		dbscan.fit(x)
		n_archetypes = dbscan.labels_.max() + 1
		return n_archetypes

	def detect_archetypes(self, data: np.ndarray, popular_decks: np.ndarray, spd) -> Tuple[
		LatentDirichletAllocation, list]:
		num_core_decks = self.find_nr_archetypes(popular_decks)
		print("detected: {} archetypes".format(num_core_decks))
		model = LatentDirichletAllocation(n_topics=num_core_decks, max_iter=500, evaluate_every=20,
		                                  learning_method="batch")

		if num_core_decks == 0:
			print("NO ARCHETYPES FOUND!")
			classification_results = []
		else:
			classification_results = model.fit_transform(spd)

		model.predict = model.transform  # TODO: wtf!
		return model, classification_results

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

	def fit_clustering_parameters(self, data: np.ndarray, min_samples_ratio: float) -> int:
		self.classifier_state['dataset_size'] = len(data)
		min_samples = int(len(data) * min_samples_ratio)
		return min_samples

	def update_fit_clustering_params(self, data: np.ndarray, min_samples_ratio: float) -> int:
		self.classifier_state['dataset_size'] += len(data)
		min_samples = int(self.classifier_state['dataset_size'] * min_samples_ratio)
		return min_samples

	def calculate_canonical_decks(self) -> Dict[str, list]:
		canonical_decks = {}
		archetype_data = {}
		for klass in self.classifier_state['classifier'].keys():
			archetype_classifier = self.classifier_state['classifier'][klass]

			try:
				pArchetype_Card = archetype_classifier.components_
			except AttributeError:
				pArchetype_Card = archetype_classifier["components_"]

			threshold = 0.50
			canonical_decks[klass] = []
			archetype_data[klass] = {}

			for archetype_index, archetype_card_dist in enumerate(pArchetype_Card):
				archetype_data[klass][archetype_index] = {}
				deck_name = ""
				archetype_report = ""

				canonical_decks[klass].append([])
				normaliz_prob = archetype_card_dist / archetype_card_dist.max()
				most_significant_indixes = np.where(normaliz_prob > threshold)[0]
				most_significant_weights = archetype_card_dist[most_significant_indixes]
				# archetype_card_ids = np.argsort(most_significant_cards)  # archetype_card_dist)
				most_significant = dict(zip(most_significant_indixes, most_significant_weights))
				for card_dim in most_significant:
					card_id = self.classifier_state['vectorizer'].invert(klass, card_dim)
					card = self.card_db[card_id]
				# most_significant[card_dim] /= card.max_count_in_deck  # TODO: this should have been normalized already!!!!!!!

				top_card_dims = sorted(most_significant, key=most_significant.get, reverse=True)
				deck_value = 0
				mana_curve = [0 for _ in range(31)]
				race_dist = defaultdict(int)
				for card_dim in top_card_dims:
					card_id = self.classifier_state['vectorizer'].invert(klass, card_dim)
					card = self.card_db[card_id]
					if card.health > 0:
						mana_curve[card.cost] += most_significant[card_dim]
						if card.race:
							race_dist[card.race] += most_significant[card_dim]
					else:
						# penalize spells
						mana_curve[card.cost] += most_significant[card_dim] / 3
					deck_value += most_significant[card_dim]
					archetype_report += "{} {}\t".format(card, int(100 * most_significant[card_dim]) / 100.)
					# self.classifier_state['canonical_decks'][klass][archetype_index].append(card_title)
					canonical_decks[klass][archetype_index].append(card)
				archetype_report += "\n"

				earliness_ratio = sum(mana_curve[0:3]) / (0.01 + sum(mana_curve[3:]))
				if earliness_ratio > 2:
					deck_name += "Aggro ({}) ".format(earliness_ratio)
				elif earliness_ratio > 0.5:
					deck_name += "Tempo ({}) ".format(earliness_ratio)
				elif earliness_ratio == 0:
					pass
				else:
					deck_name += "Control ({}) ".format(earliness_ratio)

				mvp_id = self.classifier_state['vectorizer'].invert(klass, top_card_dims[0])
				mvp = self.card_db[mvp_id].name
				if race_dist:
					race_dist_ladder = sorted(race_dist, key=race_dist.get, reverse=True)
					best_race = race_dist_ladder[0]
					best_race_score = race_dist[race_dist_ladder[0]]
					other_races_score = deck_value - best_race_score
					if other_races_score != 0:
						if best_race_score / other_races_score > 0.3:
							# one race dominates
							mvp = str(best_race)
						mvp += " ({})".format(best_race_score / other_races_score)
				else:
					mvp += " ({})".format(most_significant[top_card_dims[0]])

				deck_name += "\"{0}\" ".format(mvp)
				deck_name += klass.lower()
				archetype_report += "deck name: {0}\n".format(deck_name)
				archetype_report += "deck value:{0}\n".format(int(deck_value))
				archetype_report += "\n"
				archetype_data[klass][archetype_index]['value'] = deck_value
				archetype_data[klass][archetype_index]['report'] = archetype_report

			vals = sorted([arch['value'] for arch in archetype_data[klass].values()])

			while archetype_data[klass]:
				max_val = vals.pop()
				for idx, arch in archetype_data[klass].items():
					if arch['value'] == max_val:
						print("topic {}:".format(idx))
						print(arch['report'])
						del archetype_data[klass][idx]
						break

		self.classifier_state['canonical_decks'] = canonical_decks
		return canonical_decks

	# takes a file, returns decks
	def load_decks_from_json_file(self, file_name):
		with open(file_name) as f:
			kara_json = json.load(f)

		decks_in_file, popular_decks, semi_popular_decks = self.load_decks_from_json(kara_json)
		print("loaded", len(decks_in_file), "decks")
		return decks_in_file, popular_decks, semi_popular_decks

	def load_decks_from_json(self, kara_json, drop_noise=True):
		points = {}
		card_to_id = kara_json['map']
		decks_in_file = 0
		popular_points = {}
		semi_popular_points = {}
		vectorizer = vectorizer_1hot(self.card_db)

		for klass_id, deck_list in kara_json['decks'].items():
			klass = self.classifier_state['hero_to_class'][int(klass_id)]
			klass_decks = []
			cards_seen_for_klass = set()
			times_decks_were_seen = []

			for deck_json in deck_list:
				time_seen = deck_json['observations']
				cards = deck_json['cards']
				deck = []
				for card_external_id, num_cards in cards.items():
					card_id = card_to_id[card_external_id]
					cards_seen_for_klass.add(card_id)
					# card = self.card_db[card_id]
					for _ in range(num_cards):
						deck.append(card_id)
				decks_in_file += time_seen
				times_decks_were_seen.append(time_seen)
				klass_decks.append(deck)

			vectorizer.train_klass(klass, cards_seen_for_klass)
			unique_klass_points = vectorizer.transform(klass, klass_decks)

			samples_mod = self.classifier_state['min_samples_ratio']
			dataset_size = sum(times_decks_were_seen)
			min_samples = int(dataset_size * samples_mod)
			dropped_points = 0
			dropped_decks = 0
			klass_points = []
			popular_klass_points = []
			semipopular_klass_points = []
			for klass_point, time_seen in zip(unique_klass_points, times_decks_were_seen):
				for _ in range(time_seen):
					klass_points.append(klass_point)

				if time_seen >= min_samples:
					popular_klass_points.append(klass_point)

				if time_seen >= min_samples / 10:
					semipopular_klass_points.append(klass_point)
				else:
					dropped_decks += 1
					dropped_points += time_seen

			print("[{}]".format(klass), dropped_decks, "unique decks were noise, total datapoints", dropped_points)
			points[klass] = np.array(klass_points)
			popular_points[klass] = np.array(popular_klass_points)
			semi_popular_points[klass] = np.array(semipopular_klass_points)
		self.classifier_state['vectorizer'] = vectorizer
		return points, popular_points, semi_popular_points

	def persist(self, file_name):
		state = {}
		for key, val in self.classifier_state.items():
			state[key] = {}
			if key == "classifier":
				for klass, classifier in val.items():
					state[key][klass] = classifier.get_params()
					state[key][klass]["components_"] = classifier.components_
			else:
				state[key] = val
		with open(file_name, 'wb') as f:
			pickle.dump(state, f)

	def load_state_from_file(self, file_name):
		with open(file_name, 'rb') as f:
			state = pickle.load(f)

		for key, val in state.items():
			if key == "classifier":
				self.classifier_state[key] = {}
				for klass, params in val.items():
					self.classifier_state[key][klass] = LatentDirichletAllocation.set_params(params)
			else:
				self.classifier_state[key] = val


def main():
	classifier = DeckClassifier()
	classifier.load_state_from_file("/tmp/kara_classifier_state")
	classifier.calculate_canonical_decks()
	return

	dataset_path = sys.argv[1]
	results_path = sys.argv[2]
	map_path = sys.argv[3]
	# train_data_path = "datasets/Deck_List_Training_Data.csv"
	kara_data = "datasets/kara_data.json"

	classifier = DeckClassifier()
	# loaded_data = classifier.load_train_data_from_file(train_data_path)
	loaded_data, popular_decks, semi_popular_decks = classifier.load_decks_from_json_file(kara_data)

	# classifier.fit_transform({'MAGE': loaded_data['MAGE']})
	classifier.fit_transform(loaded_data, popular_decks, semi_popular_decks)
	classifier.calculate_canonical_decks()
	classifier.persist("/tmp/kara_classifier_state")
	classifier.load_state_from_file("/tmp/kara_classifier_state")

	del loaded_data

	decks = classifier.load_decks_from_file(dataset_path)
	with open(results_path, 'w') as results:
		results_writer = csv.writer(results)
		for klass, klass_decks in decks.items():
			klass = int(''.join(filter(str.isdigit, klass)))
			hero_to_class = ['UNKNOWN', 'WARRIOR', 'SHAMAN', 'ROGUE', 'PALADIN',
			                 'HUNTER', 'DRUID', 'WARLOCK', 'MAGE', 'PRIEST']
			klass = hero_to_class[klass]
			results = []
			for deck in klass_decks:
				predicted_deck, prob = classifier.predict_update(deck, klass)
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
