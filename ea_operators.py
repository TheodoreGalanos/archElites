from copy import copy
from random import random, randrange, uniform, sample
import math
from math import ceil, floor
import numpy as np

from population import OffspringGrid, IndividualGrid, NeighbourGrid
from utilities import height_to_color, get_poly_ids
from geometry import find_intersections, find_containments


def polynomial_bounded(ind, cmap, eta: float, low: float, up: float, mut_pb: float):
    """Return a polynomial bounded mutation, as defined in the original NSGA-II paper by Deb et al.
    Mutations are applied directly on `individual`, which is then returned.
    Inspired from code from the DEAP library (https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py).

    Args:
        ind ([type]): The individual to mutate.
        cmap (list): A list of RGB values representing the height range of the individual.
        eta (float): Crowding degree of the mutation. A high ETA will produce mutants close to its parent,
                     a small ETA will produce offspring with more differences.
        low (float): Lower bound of the search domain.
        up (float): Upper bound of the search domain.
        mut_pb (float): The probability for each item of `individual` to be mutated.

    Returns:
        offspring: A mutation of the input individual.
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

    mut_colors = np.array([height_to_color(cmap, height) for height in mut_heights]).astype(int)
    offspring = OffspringGrid(ind.polygons, mut_colors, mut_heights, ind.grid_ids, ind.status, ind.building_ids,
                              ind.added, ind.dropped)
    return offspring

def uniform_crossover_geometry(location, indgen:float, grid_id1=None, grid_id2=None, random=True):

    # Get two random individuals from the location
    if(random):
        # Create the 2 individual grids
        grid_ids = sample(list(location.grid_occupancy), 2)
        ind1 = IndividualGrid(location, grid_ids[0])
        ind2 = IndividualGrid(location, grid_ids[1])
        neighbourhood = NeighbourGrid(location, grid_ids[0])
    else:
        ind1 = IndividualGrid(location, grid_id1)
        ind2 = IndividualGrid(location, grid_id2)
        neighbourhood = NeighbourGrid(location, grid_id1)

    # Create an offspring by doing crossover of the selected polygons from the 1st grid to the second
    if(np.sum(ind1.status) == 0):
        offspring = None
        #return "Seed individual had empty genome"
    else:
        if(np.sum(ind1.status) < 10):
            if(np.sum(ind1.status)*indgen <= 1):
                poly_ids_1 = list(np.where(ind1.status == 1)[0])
                poly_ids_1.extend(np.where(ind1.status == 0)[0])
            else:
                rnd = randrange(1, max(1, ceil(np.sum(ind1.status)* indgen)))
                poly_ids_1 = sample(list(np.where(ind1.status == 1)[0]), rnd)
                poly_ids_1.extend(np.where(ind1.status == 0)[0])
        else:
            poly_ids_1 = get_poly_ids(ind1.polygons, ind1.status, indgen)
            poly_ids_1.extend(np.where(ind1.status == 0)[0])

        polygons1_, heights1_, centroids1_, colors1_, status1_, ids1_ = np.array(ind1.polygons)[poly_ids_1], \
                                                                        np.array(ind1.heights)[poly_ids_1], \
                                                                        np.array(ind1.centroids)[poly_ids_1], \
                                                                        np.array(ind1.colors)[poly_ids_1], \
                                                                        np.array(ind1.status)[poly_ids_1], \
                                                                        ind1.building_ids[0][poly_ids_1]

        dropped = list(set(ind1.building_ids[0]).difference(ids1_))

        assert polygons1_.shape[0] == heights1_.shape[0] == centroids1_.shape[0] == colors1_.shape[0] == ids1_.shape[0]
        assert len(polygons1_.shape) == len(heights1_.shape) == len(ids1_.shape)

        # Grab only active polygons from 2nd grid location
        active_elements = np.where(ind2.status == 1)[0]
        active_polygons = np.array(ind2.polygons)[active_elements]

        # Find intersections of the active polygons with the buildings of the selected individual
        intersection_matrix = np.zeros((polygons1_.shape[0], active_polygons.shape[0]))
        for k, p in enumerate(active_polygons):
            intersection_matrix[:, k] = find_intersections(p, polygons1_)
        bools1a = np.sum(intersection_matrix, axis=0).astype(bool)

        # Find containments of the active polygons with the buildings of the selected individual
        contain_matrix = np.zeros((polygons1_.shape[0], active_polygons.shape[0]))
        for k, p in enumerate(active_polygons):
            contain_matrix[:, k] = find_containments(p, polygons1_)
        bools1b = np.sum(contain_matrix, axis=0).astype(bool)

        # Find intersections of those polygons with the buildings in the neighbourhood of the selected individual
        intersection_matrix = np.zeros((len(neighbourhood.polygons), active_polygons.shape[0]))
        for k, p in enumerate(active_polygons):
            intersection_matrix[:, k] = find_intersections(p, neighbourhood.polygons)
        bools2a = np.sum(intersection_matrix, axis=0).astype(bool)

        # Find containments of the active polygons with the buildings of the selected individual
        contain_matrix = np.zeros((len(neighbourhood.polygons), active_polygons.shape[0]))
        for k, p in enumerate(active_polygons):
            contain_matrix[:, k] = find_containments(p, neighbourhood.polygons)
        bools2b = np.sum(contain_matrix, axis=0).astype(bool)

        bools = bools1a | bools1b | bools2a | bools2b

        mask = ~bools
        p2 = np.array(ind2.polygons)[active_elements][mask]
        hts2_ = ind2.heights[active_elements][mask]
        colors2_ = ind2.colors[active_elements][mask]
        status2_ = ind2.status[active_elements][mask]
        ids2_ = ind2.building_ids[0][active_elements][mask]

        added = ids2_

        assert p2.shape[0] == hts2_.shape[0] == colors2_.shape[0] == status2_.shape[0] == ids2_.shape[0]
        assert len(p2.shape) == len(hts2_.shape)

        polygons_cross_1 = np.hstack((polygons1_, p2))
        colors_cross_1 = np.vstack((colors1_, colors2_))
        heights_cross_1 = np.hstack((heights1_, hts2_))
        status_cross_1 = np.hstack((status1_, status2_))
        ids_cross_1 = np.hstack((ids1_, ids2_))

        assert polygons_cross_1.shape[0] == colors_cross_1.shape[0] == heights_cross_1.shape[0] == status_cross_1.shape[0] \
                                                                                        == ids_cross_1.shape[0]

        grid_ids = [ind1.grid_id, ind2.grid_id]

        offspring = OffspringGrid(polygons_cross_1, colors_cross_1, heights_cross_1, grid_ids, status_cross_1,
                                    ids_cross_1, added, dropped)
        return offspring