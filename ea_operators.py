from random import randrange, uniform, sample, random
from copy import copy
import math
import numpy as np

from geometry import Geometry as geom
from population import Offspring
from utils import Utilities as util

class EaOperators:

	#################################################
	# CROSSOVER
	#################################################

	@staticmethod
	def uniform_crossover_geometry(ind1, ind2, indgen, indprob, parent_ids, random_genes=False):
		"""
		Executes a uniform crossover between two individuals in the collection and
		generates two offspring with different 'first parent'. Buildings are selected
		according to the *indpb* probability, while *indgen* defines how much genetic 
		material will be used from the first parent.

		:param ind1: The first individual participating in the crossover.
		:param ind2: The second individual participating in the crossover.
		:param indgen: The minimum amount of genetic material to be taken from
		ind1.
		:param indpb: Independent probability for each building polygon to be kept.
		:returns: A tuple of two individuals.
		"""

		# get geometry information from individuals
		heights1, polygons1, colors1, centroids1, size1 = ind1.heights, ind1.polygons, ind1.colors, ind1.centroids, ind1.size
		heights2, polygons2, colors2, centroids2, size2 = ind2.heights, ind2.polygons, ind2.colors, ind2.centroids, ind2.size

		#keep a minimum amount of genetic material from parent 1, according to indgen
		#some hacky stuff to avoid weird or failed crossover when individuals have only a few buildings
		if(len(polygons2) < 4 or len(polygons1) < 4):

			# get ids for individual 1
			if(len(polygons1) > 4):
				poly_ids_1 = util.get_poly_ids(polygons1, random_genes, indgen)
			else:
				poly_ids_1 = [randrange(0, len(polygons1))]

			# get ids for individual 2
			if(len(polygons2) > 4):
				poly_ids_2 = util.get_poly_ids(polygons2, random_genes, indgen)
			else:
				poly_ids_2 = [randrange(0, len(polygons2))]

			# let's make offspring 1
			# get polygons for individual 1
			polygons1_, heights1_, centroids1_, colors1_ = polygons1[poly_ids_1], heights1[poly_ids_1], centroids1[poly_ids_1], colors1[poly_ids_1]
			assert polygons1_.shape[0] == heights1_.shape[0] == centroids1_.shape[0] == colors1_.shape[0]
			assert len(polygons1_.shape) == len(heights1_.shape)
			assert len(centroids1_.shape) == len(colors1_.shape)

			# create the first offspring
			intersection_matrix = np.zeros((polygons1_.shape[0], len(polygons2)))
			for k, p in enumerate(polygons1_):
				intersection_matrix[k, :] = geom.find_intersections(p, polygons2)

			bools = np.sum(intersection_matrix, axis=0).astype(bool)
			mask = ~bools
			p2 = polygons2[mask]
			hts2_ = heights2[mask]
			colors2_ = colors2[mask]
			assert p2.shape[0] == hts2_.shape[0] == colors2_.shape[0]
			assert len(p2.shape) == len(hts2_.shape)

			polygons_cross_1 = np.hstack((polygons1_, p2))
			colors_cross_1 = np.vstack((colors1_, colors2_))
			heights_cross_1 = np.hstack((heights1_, hts2_))
			assert polygons_cross_1.shape[0] == colors_cross_1.shape[0] == heights_cross_1.shape[0]
			offspring_1 = Offspring(polygons_cross_1, colors_cross_1, heights_cross_1, size1, parent_ids)
			
			# let's make offspring 2
			# get polygons for individual 2
			polygons2_, heights2_, centroids2_, colors2_ = polygons2[poly_ids_2], heights2[poly_ids_2], centroids2[poly_ids_2], colors2[poly_ids_2]
			assert polygons2_.shape[0] == heights2_.shape[0] == centroids2_.shape[0] == colors2_.shape[0]
			assert len(polygons2_.shape) == len(heights2_.shape)
			assert len(centroids2_.shape) == len(colors2_.shape)

			# create the second offspring
			intersection_matrix = np.zeros((polygons2_.shape[0], len(polygons1)))
			for k, p in enumerate(polygons2_):
				intersection_matrix[k, :] = geom.find_intersections(p, polygons1)

			bools = np.sum(intersection_matrix, axis=0).astype(bool)
			mask = ~bools
			p1 = polygons1[mask]
			hts1_ = heights1[mask]
			colors1_ = colors1[mask]
			assert p1.shape[0] == hts1_.shape[0] == colors1_.shape[0]
			assert len(p1.shape) == len(hts1_.shape)

			polygons_cross_2 = np.hstack((polygons2_, p1))
			colors_cross_2 = np.vstack((colors2_, colors1_))
			heights_cross_2 = np.hstack((heights2_, hts1_))
			assert polygons_cross_2.shape[0] == colors_cross_2.shape[0] == heights_cross_2.shape[0]
			offspring_2 = Offspring(polygons_cross_2, colors_cross_2, heights_cross_2, size2, parent_ids)
		else:
			poly_ids_1 = util.get_poly_ids(polygons1, random_genes, indgen)
			poly_ids_2 = util.get_poly_ids(polygons2, random_genes, indgen)

			# let's make offspring 1
			# get polygons for individual 1
			polygons1_, heights1_, centroids1_, colors1_ = polygons1[poly_ids_1], heights1[poly_ids_1], centroids1[poly_ids_1], colors1[poly_ids_1]
			assert polygons1_.shape[0] == heights1_.shape[0] == centroids1_.shape[0] == colors1_.shape[0]
			assert len(polygons1_.shape) == len(heights1_.shape)
			assert len(centroids1_.shape) == len(colors1_.shape)

			# create the first offspring
			intersection_matrix = np.zeros((polygons1_.shape[0], len(polygons2)))
			for k, p in enumerate(polygons1_):
				intersection_matrix[k, :] = geom.find_intersections(p, polygons2)

			bools = np.sum(intersection_matrix, axis=0).astype(bool)
			mask = ~bools
			p2 = polygons2[mask]
			hts2_ = heights2[mask]
			colors2_ = colors2[mask]
			assert p2.shape[0] == hts2_.shape[0] == colors2_.shape[0]
			assert len(p2.shape) == len(hts2_.shape)

			polygons_cross_1 = np.hstack((polygons1_, p2))
			colors_cross_1 = np.vstack((colors1_, colors2_))
			heights_cross_1 = np.hstack((heights1_, hts2_))
			assert polygons_cross_1.shape[0] == colors_cross_1.shape[0] == heights_cross_1.shape[0]
			offspring_1 = Offspring(polygons_cross_1, colors_cross_1, heights_cross_1, size1, parent_ids)
			
			# let's make offspring 2

			# get polygons for individual 2
			polygons2_, heights2_, centroids2_, colors2_ = polygons2[poly_ids_2], heights2[poly_ids_2], centroids2[poly_ids_2], colors2[poly_ids_2]
			assert polygons2_.shape[0] == heights2_.shape[0] == centroids2_.shape[0] == colors2_.shape[0]
			assert len(polygons2_.shape) == len(heights2_.shape)
			assert len(centroids2_.shape) == len(colors2_.shape)

			# create the second offspring
			intersection_matrix = np.zeros((polygons2_.shape[0], len(polygons1)))
			for k, p in enumerate(polygons2_):
				intersection_matrix[k, :] = geom.find_intersections(p, polygons1)

			bools = np.sum(intersection_matrix, axis=0).astype(bool)
			mask = ~bools
			p1 = polygons1[mask]
			hts1_ = heights1[mask]
			colors1_ = colors1[mask]
			assert p1.shape[0] == hts1_.shape[0] == colors1_.shape[0]
			assert len(p1.shape) == len(hts1_.shape)

			polygons_cross_2 = np.hstack((polygons2_, p1))
			colors_cross_2 = np.vstack((colors2_, colors1_))
			heights_cross_2 = np.hstack((heights2_, hts1_))
			assert polygons_cross_2.shape[0] == colors_cross_2.shape[0] == heights_cross_2.shape[0]
			offspring_2 = Offspring(polygons_cross_2, colors_cross_2, heights_cross_2, size2, parent_ids)
			
		return offspring_1, offspring_2

	@staticmethod
	def feasible_infeasible_crossover(ind1, ind2, indprob, parent_ids):
		"""
		Executes an FI crossover and calculates fitness with respect to infeasibility

		:param ind1: The first individual participating in the crossover.
		:param ind2: The second individual participating in the crossover.
		:param indgen: The minimum amount of genetic material to be taken from
		ind1.
		:param indpb: Independent probability for each building polygon to be kept.
		:returns: A tuple of two individuals.
		"""

		# get geometry information from individuals
		heights1, polygons1, colors1, centroids1, size1 = ind1.heights, ind1.polygons, ind1.colors, ind1.centroids, ind1.size
		heights2, polygons2, colors2, centroids2, size2 = ind2.heights, ind2.polygons, ind2.colors, ind2.centroids, ind2.size
		#print("Individual 1 has {} buildings".format(len(polygons1)))
		#print("Individual 2 has {} buildings".format(len(polygons2)))

		#select from ind1
		probs = np.array([uniform(0,1) for j in range(0, len(polygons1))])
		selection = [probs < indprob]
		poly_ids = np.arange(0, len(polygons1))[tuple(selection)]
		p1 = polygons1[poly_ids]
		hts1_ = heights1.reshape(-1, 1)[poly_ids]
		centroids1_ = centroids1[poly_ids]
		colors1_ = colors1[poly_ids]
		assert p1.shape[0] == hts1_.shape[0] == centroids1_.shape[0] == colors1_.shape[0]
		assert len(p1.shape) == len(hts1_.shape)
		assert len(centroids1_.shape) == len(colors1_.shape)

		#select from ind2
		probs = np.array([uniform(0,1) for j in range(0, len(polygons2))])
		selection = [probs < indprob]
		poly_ids = np.arange(0, len(polygons2))[tuple(selection)]
		p2 = np.array(polygons2)[poly_ids]
		hts2_ = heights2.reshape(-1, 1)[poly_ids]
		centroids2_ = centroids2[poly_ids]
		colors2_ = colors2[poly_ids]
		assert p2.shape[0] == hts2_.shape[0] == centroids2_.shape[0] == colors2_.shape[0]

		# join material from both individuals
		p_cross = np.hstack((p1, p2))
		hts_cross = np.vstack((hts1_, hts2_))
		colors_cross = np.vstack((colors1_, colors2_))
		#keep a minimum amount of genetic material from parent 1, according to indgen

		# calculate feasibility fitness through self-intersection
		intersection_matrix = np.zeros((len(p_cross), len(p_cross)))
		for k, p in enumerate(p_cross):
		    intersection_matrix[k, :] = geom.find_intersections(p, p_cross)
		#remove self-intersection for each polygon
		intersection_events = np.sum(intersection_matrix, axis=0)-1
		fi_fitness = np.where(intersection_events>0)[0].shape[0]/intersection_events.shape[0]

		offspring = Offspring(p_cross, colors_cross, hts_cross, size1, parent_ids)
		offspring.fi_fitness = fi_fitness

		return offspring



	#################################################
	# MUTATION
	#################################################

	@staticmethod
	def polynomial_bounded(ind, cmap, eta: float, low: float, up: float, mut_pb: float):
		"""Return a polynomial bounded mutation, as defined in the original NSGA-II paper by Deb et al.
		Mutations are applied directly on `individual`, which is then returned.
		Inspired from code from the DEAP library (https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py).

		Parameters
		----------
		:param individual
			The individual to mutate.
		:param eta: float
			Crowding degree of the mutation.
			A high ETA will produce mutants close to its parent,
			a small ETA will produce offspring with more differences.
		:param low: float
			Lower bound of the search domain.
		:param up: float
			Upper bound of the search domain.
		:param mut_pb: float
			The probability for each item of `individual` to be mutated.
		"""
		mut_heights = copy(ind.heights)
		for i in range(len(ind.heights)):
			if random() < mut_pb:
				x = ind.heights[i].astype(float)
				if(x<low):
					x=low
				if(x>up):
					x=up
				delta_1 = (x - low) / (up - low)
				delta_2 = (up - x) / (up - low)
				rand = random()
				mut_pow = 1. / (eta + 1.)

				if rand < 0.5:
					xy = 1. - delta_1
					val = 2. * rand + (1. - 2. * rand) * xy**(eta + 1.)
					delta_q = val**mut_pow - 1.
				else:
					xy = 1. - delta_2
					val = 2. * (1. - rand) + 2. * (rand - 0.5) * xy**(eta + 1.)
					delta_q = 1. - val**mut_pow

				x += delta_q * (up - low)
				x = min(max(x, low), up)
				if(math.isnan(x)):
					x = randrange(low, up)
				mut_heights[i] = x

		mut_colors = np.array([util.height_to_color(cmap, height) for height in mut_heights])
		offspring = Offspring(ind.polygons, mut_colors, mut_heights, ind.size, ind.parent_ids)

		return offspring

	@staticmethod
	def crossover_mutation(ind1, ind2, cmap, eta: float, low: float, up: float, mut_pb: float, cross_pb: 'default'):
		"""
		Parameters
		----------
		:param ind1
			The first individual participating in the mutation.
		:param ind2
			The second individual participating in the mutation.
		:param eta: float
			Crowding degree of the mutation.
			A high ETA will produce mutants close to its parent,
			a small ETA will produce offspring with more differences.
		:param low: float
			Lower bound of the search domain.
		:param up: float
			Upper bound of the search domain.
		:param mut_pb: float
			The probability for each item of `individual` to be mutated.
		"""
		
		mut_heights_1 = copy(ind1.heights)
		mut_heights_2 = copy(ind2.heights)

		# crossover first
		if (cross_pb == 'default'):
			for i in range(len(ind1.heights)):
				if (random() < 1/len(ind1.heights)):
					mut_heights_1[i] = ind2.heights[randrange(0, len(ind2.heights)-1)]

			for i in range(len(ind2.heights)):
				if (random() < 1/len(ind2.heights)):
					mut_heights_2[i] = ind1.heights[random.randrange(0, len(ind1.heights)-1)]
		else:
			for i in range(len(ind1.heights)):
				if (random() < cross_pb):
					mut_heights_1[i] = ind2.heights[random.randrange(0, len(ind2.heights)-1)]

			for i in range(len(ind2.heights)):
				if (random() < cross_pb):
					mut_heights_2[i] = ind1.heights[randrange(0, len(ind1.heights)-1)]

		# mutate after
		for i in range(len(ind1.heights)):
			if (random() < mut_pb):
				x = ind1.heights[i].astype(float)
				delta_1 = (x - low) / (up - low)
				delta_2 = (up - x) / (up - low)
				rand = random()
				mut_pow = 1. / (eta + 1.)

				if rand < 0.5:
					xy = 1. - delta_1
					val = 2. * rand + (1. - 2. * rand) * xy**(eta + 1.)
					delta_q = val**mut_pow - 1.
				else:
					xy = 1. - delta_2
					val = 2. * (1. - rand) + 2. * (rand - 0.5) * xy**(eta + 1.)
					delta_q = 1. - val**mut_pow

				x += delta_q * (up - low)
				x = min(max(x, low), up)
				mut_heights_1[i] = x

		# mutate after
		for i in range(len(ind2.heights)):
			if (random() < mut_pb):
				x = ind2.heights[i].astype(float)
				delta_1 = (x - low) / (up - low)
				delta_2 = (up - x) / (up - low)
				rand = random()
				mut_pow = 1. / (eta + 1.)

				if rand < 0.5:
					xy = 1. - delta_1
					val = 2. * rand + (1. - 2. * rand) * xy**(eta + 1.)
					delta_q = val**mut_pow - 1.
				else:
					xy = 1. - delta_2
					val = 2. * (1. - rand) + 2. * (rand - 0.5) * xy**(eta + 1.)
					delta_q = 1. - val**mut_pow

				x += delta_q * (up - low)
				x = min(max(x, low), up)
				mut_heights_2[i] = x


		mut_colors_1 = np.array([util.height_to_color(cmap, height) for height in mut_heights_1]).astype(int)
		mut_colors_2 = np.array([util.height_to_color(cmap, height) for height in mut_heights_2]).astype(int)

		offspring_1 = Offspring(ind1.polygons, mut_colors_1, mut_heights_1, ind1.size, ind1.parent_ids)
		offspring_2 = Offspring(ind2.polygons, mut_colors_2, mut_heights_2, ind2.size, ind2.parent_ids)

		return offspring_1, offspring_2
