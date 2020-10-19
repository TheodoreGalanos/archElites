import random
from random import randrange, uniform, sample
from copy import copy

import numpy as np

from geometry import Geometry as geom
from population import Offspring
from utils import Utilities as util

class EaOperators:

	#################################################
	# CROSSOVER
	#################################################

	@staticmethod
	def uniform_crossover_similarity(ind1, pairs, indgen, indprob):
		"""
		Executes an initialization of the population within a collection by
		creating an individual from two provided individuals. The buildings
		are selected from ind1 according to the	*indpb* probability, while
		*indgen* defines how much genetic material will be used from ind1.

		:param ind1: The first individual participating in the crossover.
		:param pairs: A list of paired individuals participating in the crossover.
		:param indgen: The minimum amount of genetic material to be taken from
		the seed individual
		:param indpb: Independent probability for each attribute to be exchanged.

		:returns: Polygons, colors, and heights of the new individual
		"""

	# get geometry information from first parent
		heights1, polygons1, colors1, centroids1, size1 = ind1.heights, ind1.polygons, ind1.colors, ind1.centroids, ind1.size
		#print("Individual 1 has {} buildings".format(len(polygons1)))

		all_polygons = []
		all_colors = []
		all_heights = []
		all_sizes = []
		for i, ind in enumerate(pairs):
			heights2, polygons2, colors2, centroids2, size2 = ind.heights, ind.polygons, ind.colors, ind.centroids, ind.size
			#print("Individual 2 has {} buildings".format(len(polygons2)))

			#keep a minimum amount of genetic material from parent 1, according to indgen

			selection = []
			poly_ids = random.sample(list(np.arange(0, len(polygons1))), int(len(polygons1) * indgen))
			"""
			tries = 0
			while np.array(selection).sum() <= int(len(polygons1) * indgen) and tries < 250:
				probs = np.array([uniform(0,1) for j in range(0, len(polygons1))])
				selection = [probs > indprob]
				poly_ids = np.arange(0, len(polygons1))[tuple(selection)]
				tries +=1
			print('{} iteration(s) until successful crossover mask was found!'.format(tries))
			"""
			print('total selected buildings from individual 1: {} out of {}'.format(len(poly_ids), len(polygons1)))

			# keep only selected buildings from the first individual and throw away all intersecting
			# building from the second individual
			p1 = np.array(polygons1)[poly_ids]
			hts1_ = heights1[poly_ids]
			centroids1_ = centroids1[poly_ids]
			colors1_ = colors1[poly_ids]
			assert p1.shape[0] == hts1_.shape[0] == centroids1_.shape[0] == colors1_.shape[0]

			#find intersections for each polygon selected in p1
			intersection_matrix = np.zeros((p1.shape[0], len(polygons2)))
			for k, p in enumerate(p1):
				intersection_matrix[k, :] = geom.find_intersections(p, polygons2)
			bools = np.sum(intersection_matrix, axis=0).astype(bool)
			mask = ~bools

			p2 = np.array(polygons2)[mask]
			hts2_ = np.array(heights2)[mask]
			colors2_ = colors2[mask]
			assert p2.data.shape[0] == hts2_.data.shape[0] == colors2_.data.shape[0]

			#join polygons from both parents and assign colors
			polygons_cross = np.hstack((p1, p2.data))
			colors_cross = np.vstack((colors1_, colors2_.data))
			heights_cross = np.hstack((hts1_, hts2_.data))
			assert polygons_cross.shape[0] == colors_cross.shape[0] == heights_cross.shape[0]
			#print("{} buildings present in offspring {}".format(polygons_cross.shape[0], i))

			all_polygons.append(polygons_cross)
			all_colors.append(colors_cross)
			all_heights.append(heights_cross)
			all_sizes.append(ind1.size)

		offsprings = [Offspring(polygons, colors, heights, size) for (polygons, colors, heights, sizes) in zip(all_polygons, all_colors, all_heights, all_sizes)]

		return offsprings

	@staticmethod
	def uniform_crossover_geometry(ind1, ind2, indgen, indprob):
    """
    Executes an initialization of the population within a collection by
    creating an individual from two provided individuals. The buildings
    are selected from ind1 according to the	*indpb* probability, while
    *indgen* defines how much genetic material will be used from ind1.

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
    print("Individual 1 has {} buildings".format(len(polygons1)))
    print("Individual 2 has {} buildings".format(len(polygons2)))

    #keep a minimum amount of genetic material from parent 1, according to indgen

    selection = []
    if(len(polygons2) < 4 or len(polygons1) < 4):

        if(len(polygons1) > 4):
            poly_ids = random.sample(list(np.arange(0, len(polygons1))), int(len(polygons1) * indgen))
        else:
            poly_ids = [random.randrange(0, len(polygons1))]
        print('total selected buildings from individual 1: {} out of {}'.format(len(poly_ids), len(polygons1)))

        p1 = np.array(polygons1)[poly_ids]
        hts1_ = heights1.reshape(-1, 1)[poly_ids]
        centroids1_ = centroids1[poly_ids]
        colors1_ = colors1[poly_ids]
        assert p1.shape[0] == hts1_.shape[0] == centroids1_.shape[0] == colors1_.shape[0]

        intersection_matrix = np.zeros((p1.shape[0], len(polygons2)))
        for k, p in enumerate(p1):
            intersection_matrix[k, :] = intersection(p, polygons2)
        bools = np.sum(intersection_matrix, axis=0).astype(bool)
        mask = ~bools

        p2 = np.array(polygons2)[mask]
        hts2_ = np.array(heights2).reshape(-1,1)[mask]
        colors2_ = colors2[mask]
        assert p2.data.shape[0] == hts2_.data.shape[0] == colors2_.data.shape[0]

        #join polygons from both parents and assign colors
        polygons_cross = np.hstack((p1, p2.data))
        colors_cross = np.vstack((colors1_, colors2_.data))
        heights_cross = np.vstack((hts1_, hts2_.data))
        assert polygons_cross.shape[0] == colors_cross.shape[0] == heights_cross.shape[0]
        print("{} buildings present in offspring".format(polygons_cross.shape[0]))

        offspring = Offspring(polygons_cross, colors_cross, heights_cross, size1)

    else:
        poly_ids = random.sample(list(np.arange(0, len(polygons1))), int(len(polygons1) * indgen))
        """
        tries = 0
        while np.array(selection).sum() <= int(len(polygons1) * indgen) and tries < 250:
            probs = np.array([uniform(0,1) for j in range(0, len(polygons1))])
            selection = [probs > indprob]
            poly_ids = np.arange(0, len(polygons1))[tuple(selection)]
            tries +=1
        print('{} iteration(s) until successful crossover mask was found!'.format(tries))
        """
        print('total selected buildings from individual 1: {} out of {}'.format(len(poly_ids), len(polygons1)))

        # keep only selected buildings from the first individual and throw away all intersecting
        # building from the second individual
        p1 = np.array(polygons1)[poly_ids]
        hts1_ = heights1[poly_ids]
        centroids1_ = centroids1[poly_ids]
        colors1_ = colors1[poly_ids]
        assert p1.shape[0] == hts1_.shape[0] == centroids1_.shape[0] == colors1_.shape[0]

        #find intersections for each polygon selected in p1
        intersection_matrix = np.zeros((p1.shape[0], len(polygons2)))
        for k, p in enumerate(p1):
            intersection_matrix[k, :] = intersection(p, polygons2)
        bools = np.sum(intersection_matrix, axis=0).astype(bool)
        mask = ~bools

        p2 = np.array(polygons2)[mask]
        hts2_ = np.array(heights2)[mask]
        colors2_ = colors2[mask]
        assert p2.data.shape[0] == hts2_.data.shape[0] == colors2_.data.shape[0]

        #join polygons from both parents and assign colors
        polygons_cross = np.hstack((p1, p2.data))
        colors_cross = np.vstack((colors1_, colors2_.data))
        heights_cross = np.hstack((hts1_, hts2_.data))
        assert polygons_cross.shape[0] == colors_cross.shape[0] == heights_cross.shape[0]
        print("{} buildings present in offspring".format(polygons_cross.shape[0]))

        offspring = Offspring(polygons_cross, colors_cross, heights_cross, size1)

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
			if random.random() < mut_pb:
				x = ind.heights[i]
				delta_1 = (x - low) / (up - low)
				delta_2 = (up - x) / (up - low)
				rand = random.random()
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
				mut_heights[i] = x

		mut_colors = np.array([util.height_to_color(cmap, height) for height in mut_heights]).astype(int)
		offspring = Offspring(ind.polygons, mut_colors, mut_heights, ind.size, ind.parent_ids)

		return offspring
