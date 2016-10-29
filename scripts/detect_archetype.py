import collections
import json
import pickle
from django.utils import timezone

import itertools

import datetime
from scipy.spatial import distance
import random
import sys
import csv

import nltk
import numpy as np
import multiprocessing
import sklearn

from hearthstone import cardxml
from nltk import FreqDist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


class DeckClassifierAPI(object):  # Resource):
	@classmethod
	def make_api(cls, classifier):
		cls.classifier = classifier
		return cls

	@staticmethod
	def get_placeholder_deck():
		x = np.array([0] * 669)
		for i in range(30):
			x[random.randint(0, len(x))] += 1
		return x.reshape(1, -1)

	def post(self):
		deck = json.loads(request.form['deck'])
		klass = request.form['klass']

		lookup = self.classifier.dimension_to_card_name
		x = self.classifier.deck_to_vector([deck], lookup[klass])

		index = int(self.classifier.dbscan_predict(x, klass))
		name = self.classifier.cluster_names[klass][index]
		# races = self.classifier.cluster_races[klass][index]
		# categories = self.classifier.cluster_categories[klass][index]
		canonical_deck = self.classifier.canonical_decks[klass][index]

		return (name, canonical_deck), 201


def print_data(deck_names, clusters):
	sets = collections.defaultdict(list)
	for (i, name) in enumerate(deck_names):
		sets[clusters[i]].append(name)
	groups = []
	for cluster_number in sets:
		groups.append(sets[cluster_number])

	for group in sorted(groups, key=len):
		print(len(group), group, "\n")
	print("found {} clusters".format(len(set(clusters))))


class DeckClassifier(object):
	CLASSIFIER_CACHE = "klass_classifiers.pkl"
	MAX_DIMENSIONS = 30

	def __init__(self, config=None):
		self.created = timezone.now()
		# self.app = Flask(__name__)
		# app_api = Api(self.app)
		# classifier_api = DeckClassifierAPI.make_api(self)
		# app_api.add_resource(classifier_api, "/")  # "/api/v0.1/detect_archetype")

		self.classifier_state = None
		self.test_labels = []
		self.card_db, _ = cardxml.load()
		self.cluster_names = {}
		self.pca = {}
		self.klass_classifiers = {}
		self.dimension_to_card_name = {}
		self.canonical_decks = {}

		REDIS_ADDR = "localhost"
		REDIS_PORT = 6379
		REDIS_DB = 0
		if config:
			eps = config['eps_modifier']
			min_samples = config['eps_modifier']
			DATA_FILE = config['data_file']
		else:
			eps = float(sys.argv[1])
			min_samples = float(sys.argv[2])
			DATA_FILE = "data/decks.csv"  # "all_decks.pkl"
		self.redis_db = None  # redis.StrictRedis(host=REDIS_ADDR, port=REDIS_PORT, db=REDIS_DB)
		self.maybe_train_classifier(DATA_FILE, eps, min_samples)

	def run(self):
		pass
		# self.app.run(host="0.0.0.0", port=31337)

	@staticmethod
	def load_decks_from_file(file_name):
		decks = collections.defaultdict(list)
		deck_names = {}
		with open(file_name, 'r') as f:
			dropped = 0
			decks_in_file = 0
			hsreplay_format = False
			csvreader = csv.reader(f)
			for entry in csvreader:
				if entry == ['deck_id', 'player_class', 'card_list', 'card_ids']:
					hsreplay_format = True
					continue

				if hsreplay_format:
					klass, deck = entry[1], entry[3]
				else:
					klass, deck = entry.strip().split(":")
				if deck == "None":
					continue
				deck = deck.split(", ")
				decks_in_file += 1
				if len(deck) != 30:
					dropped += 1
					continue
				decks[klass].append(deck)
				deck_names[klass] = []
			print("dropped", dropped, "out of", decks_in_file)
		return dict(decks), deck_names

	def test_accuracy(self, test_data, test_labels, klass):
		hits = 0
		unknowns = 0
		for (deck, target_label) in zip(test_data, test_labels):
			label = self.dbscan_predict(deck, klass)
			label = self.cluster_names[klass][label[0]]
			if target_label in label:
				hits += 1
			else:
				if label == "UNKNOWN":
					unknowns += 1
				else:
					# print("predicted [", label, "] was [", target_label, end=' ]\t')
					for cluster_index, prob in zip(self.klass_classifiers[klass].classes_,
												   self.dbscan_explain(deck, klass)[0]):
						prob = int(prob * 100)
						if prob != 0:
							pass
							# print(self.cluster_names[klass][cluster_index], prob, "%", end="; ")
							# print("")
		# print("predicted UNKNOWN", unknowns, "times")
		if test_data:
			return float(hits) / len(test_data)
		else:
			return -1

	def deck_to_vector(self, decks, dimension_to_card_name):
		klass_data = []
		for deck in decks:
			datapoint = np.zeros(len(dimension_to_card_name))
			for card in deck:
				try:
					card_dimension = dimension_to_card_name.index(card)
				except ValueError:
					continue
				if isinstance(deck, list):
					if datapoint[card_dimension] > 0:
						datapoint[card_dimension] += 1  # += 0.2
					else:
						datapoint[card_dimension] = 1
				else:
					datapoint[card_dimension] = deck[card]
			klass_data.append(datapoint)
		data = np.array(klass_data)
		return data

	def load_data_from_file(self, file_name):
		decks, deck_names = self.load_decks_from_file(file_name)

		data = {}
		# TODO: use vectorizer
		for klass in decks.keys():
			self.dimension_to_card_name[klass] = list({card for deck in decks[klass] for card in deck})
			data[klass] = self.deck_to_vector(decks[klass], self.dimension_to_card_name[klass])

		return data, deck_names

	def train_classifier(self, data, eps_mod, min_samples_mod, klass):
		if data.shape[1] > self.MAX_DIMENSIONS:
			transform = PCA(n_components=self.MAX_DIMENSIONS)
			data = transform.fit_transform(data)
		else:
			transform = None

		# get some statistics
		min_datapoints = 100
		SIZE = int(len(data) / 25)# consider only a subset of the dataset
		if SIZE < min_datapoints:
			SIZE = min(min_datapoints, len(data))

		pairs = list(itertools.product(range(SIZE), repeat=2))
		avg = 0
		max_dist = 0
		min_dist = 99999

		# dsts = np.zeros((data.shape[0], data.shape[0]))
		dsts = np.zeros((SIZE, SIZE))
		for i, j in pairs:
			dist = distance.cityblock(data[i], data[j])
			dsts[i][j] = dist
			max_dist = max(dist, max_dist)
			if dist > 0.001:
				min_dist = min(dist, min_dist)
				avg += dist / len(pairs)
		# plt.matshow(dsts)
		# for i, j in pairs:
		#	 plt.annotate(int(dsts[i][j]), xy=(i, j))
		print("klass", klass, "distances: avg:", avg, "max:", max_dist, "min", min_dist)
		# plt.show()
		eps = min_dist + (avg - min_dist) / 10
		eps *= eps_mod
		print("eps:", eps)
		min_samples = int(len(data) / 50)
		min_samples = int(min_samples_mod * min_samples)
		print("min_samples:", min_samples)
		labels, db = self.label_data(data, min_samples, eps)

		noise_mask = labels == -1
		dims = data.shape[1]

		X = data[~noise_mask].reshape(-1, dims)
		Y = labels[~noise_mask]
		classes = set(Y)
		if len(classes) > 2:
			classifier = sklearn.svm.SVC(probability=True)
			classifier.fit(X, Y)
		elif len(classes) == 1:
			print(klass, "has only one cluster")
			classifier = None
		else:
			print(klass, "classifier failed with ", len(data), "datapoints")
			classifier = None
		return classifier, transform, labels, db

	@staticmethod
	def get_decks_names_in_cluster(labels, cluster_index, deck_names):
		decks = []
		for i in range(len(labels)):
			if labels[i] == cluster_index:
				decks.append(deck_names[i])
		return decks

	@staticmethod
	def get_decks_in_cluster(labels, cluster_index):
		decks = np.where(labels == cluster_index)
		return decks[0]

	def dumb_train(self, data, step, min_samples):
		for klass in data:
			print(klass)
			X = data[klass]

			neighbors_model = NearestNeighbors(radius=step, p=0)
			neighbors_model.fit(X)
			# This has worst case O(n^2) memory complexity
			neighborhoods = neighbors_model.radius_neighbors(X, step, return_distance=False)

			n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])

			# Initially, all samples are noise.
			labels = -np.ones(X.shape[0], dtype=np.intp)

			min_samples = 5
			step = 20
			# A list of all core samples found.
			core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
			labels = self.dumb_cluster(core_samples, n_neighbors, step, X)
		return np.where(core_samples)[0], labels

	def update_classifier(self, decks, old_classifier):
		pass

	def load_classifier_state(self, state):
		self.classifier_state = state

	def maybe_train_classifier(self, data_file, eps_mod, samples_mod):
			loaded_data, loaded_deck_names = self.load_data_from_file(data_file)
			data, deck_names, test_data, test_labels = self.split_dataset(loaded_data, loaded_deck_names)
			del loaded_data
			del loaded_deck_names
			labels = {}
			# dumb_results = self.dumb_train(data, step, samples_mod)

			self.classifier_state = {}
			for klass in data.keys():
				classifier, transform, labels, _ = self.train_classifier(data[klass], eps_mod, samples_mod, klass)
				# cluster_names, _, _ = self.name_clusters(deck_names[klass], klass, labels)

				self.classifier_state[klass] = {
					'classifier': classifier,
					'transform': transform,
					'labels': labels,
					#'cluster_names': cluster_names,
				}

			# print("train results:")
			# self.eval_train_results(data, labels, deck_names)
			# print("test results:")
			# self.eval_test_results(test_data, test_labels)

			# with open(self.CLASSIFIER_CACHE, 'wb') as d:
			#	state_tuple = (self.klass_classifiers, self.dimension_to_card_name, self.pca, self.cluster_names, self.canonical_decks)
			#	pickle.dump(state_tuple, d)

	# consider the newest decks more important
	def dbscan_predict(self, x_new, klass):
		x_new = x_new.reshape(1, -1)
		x_new = self.pca[klass].transform(x_new)
		prediction = self.klass_classifiers[klass].predict(x_new)
		return prediction

	def dbscan_explain(self, x_new, klass):
		x_new = x_new.reshape(1, -1)
		x_new = self.pca[klass].transform(x_new)
		probs = self.klass_classifiers[klass].predict_proba(x_new)
		return probs

	def name_clusters(self, deck_names, klass, labels):
		cluster_decknames = collections.defaultdict(list)
		cluster_names = {}
		cluster_races = {}
		cluster_categories = {}
		pRaces = None
		pCategories = None

		for (i, name) in enumerate(deck_names):
			deck_label = labels[i]
			cluster_decknames[deck_label].append(name)

		for cluster_index, decknames in cluster_decknames.items():
			if cluster_index == -1:
				cluster_name = "UNKNOWN"
			else:
				klass_ = klass.lower()
				decknames = [n.lower().replace(klass_, "") for n in decknames if n.lower()]
				# stopwords = set(nltk.corpus.stopwords.words('english'))

				# Freq
				tokenizer = nltk.RegexpTokenizer(r'\w+')
				words = [word for name in decknames for word in tokenizer.tokenize(name)]  # if word not in stopwords]
				fdist = FreqDist(words)

				keywords = fdist.most_common(10)
				cluster_name = ""
				naming_cutoff = 0.5 * keywords[0][1]

				categories = ['aggro', 'combo', 'control', 'fatigue', 'midrange', 'ramp', 'tempo', 'token']
				pCategories = {}
				for cat in categories:
					pCategories[cat] = fdist[cat] / len(deck_names)

				pRaces = {}
				races = ['murloc', 'dragon', 'pirate', 'mech', 'beast']
				for race in races:
					pRaces[race] = fdist[race] / len(deck_names)

				for dn in keywords:
					if dn[1] > naming_cutoff:
						cluster_name += " " + dn[0]

			cluster_names[cluster_index] = cluster_name.lstrip()
			cluster_races[cluster_index] = pRaces
			cluster_categories[cluster_index] = pCategories
		return cluster_names, cluster_races, cluster_categories

	@staticmethod
	def split_dataset(loaded_data, loaded_deck_names):
		known_archetypes = {
			'Warrior': {"patron", "control", "pirate", "dragon", "warrior"},
			'Paladin': {"aggro murloc", "aggro", "control", "dragon"},
			'Shaman': {"aggro", "midrange"},
			'Druid': {"beast", "control", "aggro", "ramp"},
			'Priest': {"control", "dragon"},
			'Mage': {"freeze", "reno", "tempo"},
			'Hunter': {"midrange"},
			'Rogue': {"miracle", "old", "mill"},
			'Warlock': {"reno", "zoo"},
		}
		test_dataset = {}
		test_labels = {}
		train_data = {}
		deck_names = {}

		for klass in loaded_data.keys():
			test_dataset[klass] = []
			test_labels[klass] = []
			klass_data = loaded_data[klass]

			test_data_size = int(len(klass_data) * 0.02)

			if not loaded_deck_names:
				test_dataset[klass] = loaded_data[klass][:test_data_size]
				train_data[klass] = loaded_data[klass][test_data_size:]
			else:
				normalized_names = [name.lower().replace(klass.lower(), "").strip() for name in
									loaded_deck_names[klass]]

				possibilities = []
				for index, name in enumerate(normalized_names):
					if name in known_archetypes[klass]:
						possibilities.append(index)
				random.shuffle(possibilities)
				test_data_size = min(len(possibilities), test_data_size)
				# reversed so to delete from the bottom of the list
				test_indexes = list(reversed(sorted(possibilities[:test_data_size])))

				mask = np.ones_like(klass_data, dtype=bool)

				for index in test_indexes:
					name = normalized_names[index]
					test_dataset[klass].append(klass_data[index])
					test_labels[klass].append(name)

				for index in test_indexes:
					mask[index] = False
					del loaded_deck_names[klass][index]
					del normalized_names[index]

				deck_names[klass] = loaded_deck_names[klass]
				train_data[klass] = klass_data[mask].reshape(-1, klass_data.shape[1])
		return train_data, deck_names, test_dataset, test_labels

	def label_data(self, data, min_samples, eps):
		model = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
		# model = hdbscan.HDBSCAN(metric="manhattan", min_cluster_size=cluster_size, min_samples=min_samples)
		model.fit(data)
		model.labels_.reshape(-1, 1)
		return model.labels_, model

	def get_canonical_decks(self, data, transform, labels, lookup):
		transformed_data = False
		if data.shape[1] > self.MAX_DIMENSIONS:
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
					# n_cards = avg_deck[index]
					# n_cards = min(2, int(np.round(n_cards)))
					# for i in range(n_cards):
					canonical_deck.append(card_name.name + " " + str(int(avg_deck[index] * 100) / 100))
			canonical_decks[label] = canonical_deck
		return canonical_decks

	def plot_clusters(self, cluster_indexes, db, X, transform):
		core_samples_mask = np.zeros_like(cluster_indexes, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

		print('Estimated number of clusters: %d' % n_clusters_)

		# Black removed and is used for noise instead.
		unique_labels = set(labels)
		colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

		if transform:
			X1 = transform.transform(X)
		else:
			X1 = X

		tsne_model = TSNE(n_components=2)
		embed = tsne_model.fit_transform(X1)
		for k, col, in zip(unique_labels, colors):
			if k == -1:
				# Black used for noise.
				col = 'k'

			class_member_mask = (labels == k)

			xy = embed[class_member_mask & core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
					 markeredgecolor='k', markersize=12, alpha=0.9, zorder=1)
			for x, y in xy:
				plt.annotate(str(k), xy=(x, y), fontsize=8)
			xy = X[class_member_mask & ~core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], 'x', markerfacecolor=col,
					 markeredgecolor='k', markersize=16, alpha=0.4, zorder=2)

		plt.title('Estimated number of clusters: %d' % n_clusters_)
		plt.show()

	def eval_train_results(self, data, labels, deck_names):
		num_cluster_names = 0
		for klass, names in self.cluster_names.items():
			num_cluster_names += len(names)
		mean_unknown_ratio = 0
		if num_cluster_names != 0:
			for klass, cluster_names in self.cluster_names.items():
				print(klass, "clusters", len(cluster_names), end='{')
				for cluster_index, cluster_name in cluster_names.items():
					decks = self.get_decks_names_in_cluster(labels[klass], cluster_index, deck_names[klass])
					if cluster_name == "UNKNOWN":
						# print(int((float(len(decks)) / len(data[klass])) * 100), end=" ")
						print("}")
						unknown_ratio = (float(len(decks)) / len(data[klass])) * 100
						mean_unknown_ratio += unknown_ratio / len(self.cluster_names)
						print("\t{}[{}, {:.0f}%]".format(cluster_name, len(decks), unknown_ratio))
					else:
						print(cluster_name, len(decks), end=", ")
				self.canonical_decks[klass] = self.get_canonical_decks(data[klass], self.pca[klass],
																	   labels[klass],
																	   self.dimension_to_card_name[klass])
		else:
			for klass, cluster_indexes in labels.items():
				cluster_set = list(reversed(sorted(set(cluster_indexes))))
				print(klass, "num clusters:", len(cluster_set), "{")
				canonical_deck = self.get_canonical_decks(data[klass], self.pca[klass], labels[klass],
														  self.dimension_to_card_name[klass])
				self.canonical_decks[klass] = canonical_deck
				for cluster_index in cluster_set:
					decks = self.get_decks_in_cluster(cluster_indexes, cluster_index)
					print(cluster_index, ":", len(decks), "decks in this cluster")
					printed_cards = 0
					while canonical_deck[cluster_index] and printed_cards < 12:
						card = canonical_deck[cluster_index].pop(0)
						print("\t", card, end=" ")
						printed_cards += 1
						if canonical_deck[cluster_index] and card == canonical_deck[cluster_index][0]:
							canonical_deck[cluster_index].pop(0)
							print("x2", end="")
						if (printed_cards % 2) == 0:
							print("\t")
						else:
							print(" ", end="")
					if (printed_cards % 2) == 1:
						print("")
					if cluster_index == -1:
						# print(int((float(len(decks)) / len(data[klass])) * 100), end=" ")
						print("}")
						unknown_ratio = (float(len(decks)) / len(data[klass])) * 100
						mean_unknown_ratio += unknown_ratio / len(data)
						print("\t{}[{}, {:.0f}%]".format("unknown", len(decks), unknown_ratio))
						print("-" * 30)

				# self.plot_clusters(cluster_indexes, db[klass], data[klass], self.pca[klass])
		print("mean unknown ratio {:.2f}%".format(mean_unknown_ratio))

	def eval_test_results(self, test_data, test_labels):
		mean_accuracy = 0
		for klass in self.klass_classifiers:
			accuracy = self.test_accuracy(test_data[klass], test_labels[klass], klass)
			mean_accuracy += accuracy / len(self.klass_classifiers)
			# print(int(accuracy * 100))
			print(klass, "accuracy {:.2f}%".format(accuracy * 100))

		print("mean accuracy {:.2f}%".format(mean_accuracy * 100))

	def dumb_cluster(self, core_samples, cluster_sizes, step, decks, small_step=6):
		clusters = []
		for i, is_core in enumerate(core_samples):
			if is_core == 1:
				clusters.append(i)

		roots = {c: c for c in clusters}
		sizes = {c: cluster_sizes[c] for c in clusters}
		super_sizes = {c: 1 for c in clusters}
		distances = np.zeros((len(clusters), len(clusters)))
		lookup = {}
		for i, c1 in enumerate(clusters):
			lookup[c1] = i
			for j, c2 in enumerate(clusters):
				distances[i][j] = distance.cityblock(decks[c1], decks[c2])

		def plot(X):
			transform = PCA(n_components=80)
			tsne_model = TSNE(n_components=2)
			pcad = transform.fit_transform(X)
			embed = tsne_model.fit_transform(pcad)
			unique_labels = set(roots.values())
			colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
			for label, col, in zip(unique_labels, colors):
				members = members_of(label)
				for member in members:
					xy = embed[member]
					plt.plot(xy[0], xy[1], 'o', markeredgecolor='k', markersize=12, alpha=0.9, markerfacecolor=col)
			plt.show()

		def dist(i: int, j: int) -> int:
			i = lookup[i]
			j = lookup[j]
			return distances[i][j]

		def root(a: int) -> int:
			return roots[a]

		def members_of(node: int) -> list:
			return [name for name, root in roots.items() if root == node]

		def merge(acceptor: int, donor: int):
			acceptor_r = root(acceptor)
			donor_r = root(donor)
			members = members_of(donor_r)
			super_sizes[acceptor_r] += super_sizes[donor_r]
			del super_sizes[donor_r]
			for node in members:
				roots[node] = acceptor_r

		def size(a: int) -> int:
			return sizes[a]

		def cluster_size(a: int) -> int:
			return super_sizes[a]

		def variance(i: int) -> int:
			return variance_peek(i, i)

		def variance_peek(i, j):
			X = []
			labels = {root(i), root(j)}
			for index, father in roots.items():
				if father in labels:
					for _ in range(size(index)):
						X.append(decks[index])
			X = np.array(X)
			var = np.sum(np.var(X, axis=0))
			return var

		print("initial")
		for c1, c2 in itertools.combinations(clusters, 2):
			if root(c1) == root(c2):
				continue
			d = dist(c1, c2)
			if d == 0:
				merge(c1, c2)
			elif d <= small_step:
				merge(c1, c2)

		for _ in range(1000):
			change = 0
			for c1, c2 in itertools.combinations(super_sizes.keys(), 2):
				if root(c1) == root(c2):
					continue
				d = dist(c1, c2)
				if d > step:
					continue
				old_var = variance(c1)
				new_var = variance_peek(c1, c2)
				if new_var * 0.9 > old_var:
					# print(dist(c1, c2), old_var, new_var, "not merging")
					continue
				old_var = variance(c1)
				new_var = variance_peek(c1, c2)
				print(dist(c1, c2), old_var, new_var, "merging")
				merge(c1, c2)
				change += 1
			if change == 0:
				break

		print("found", len(super_sizes), "clusters")
		counter = collections.Counter()
		for v in super_sizes.values():
			counter[v] += 1

		print("len", "count")
		for k, v in counter.items():
			print(k, v)
		return roots


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from flask import Flask
	from flask_restful import Resource, Api
	from flask import request
	DeckClassifier().run()
