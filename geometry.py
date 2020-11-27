import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.affinity import translate

#############################################################
#############################################################

def create_path(polygon):
    """
    A function that splits a series of polygon points into an (x, y) array of coordinates.

    Args:
        polygon (array): a sequence of x, y points describing a building outline

    Returns:
        array: a sequence of points (x, y) defining a building outline.
    """
    x_coords = polygon[0::2]
    y_coords = polygon[1::2]

    return np.hstack((x_coords, y_coords))

#############################################################
#############################################################

def create_shapely_polygons(points, splits, grid_ids):
    """
    A function that generates shapely polygons out of points and splits (indices on where to split the points) of each individual.

    Args:
        points (list): a list of all the points for each individual.
        splits (list): a list of indices that show where to split the point list in order to create individual polygons.

    Returns:
        array: an array of all shapely polygons (buildings) of the individual.
    """
    polygons = np.array(np.vsplit(points.reshape(-1, 1), np.cumsum(splits)))[:-1]

    shapes = []
    for poly in polygons:
        path = create_path(poly)
        shapes.append(Polygon(path))

    return np.array(shapes)

def get_features(footprints, heights, boundary=(2500, 2500), b_color="white"):
    """
    Calculates urban density and network features for a set of footprints and heights.

    Args:
        footprints (list): A list of floor areas for each building of an individual.
        heights ([type]): A list of heights for each building of an individual.
        boundary (tuple, optional): The size of each individual in pixels. Defaults to (2500, 2500).
        b_color (str, optional): The background color (ground color) for each individual. Defaults to "white".

    Returns:
        float: floor space index (FSI), calculated using -> gross_floor_area / area_of_aggregation
        float: ground space index (GSI), calculated using -> footprint / area_of_aggregation
        float: oper space ratio (OSR), calculated using -> (1-GSI)/FSI
        float: building height (L), calculated using -> FSI/GSI
        float: tare (T), calculated using -> (area_of_aggregation - footprint) / area_of_aggregation
    """
    #calculate aggregation A
    area_of_aggregation = boundary[0] * boundary[1]
    #calculate GFA
    gross_floor_area = np.multiply(footprints, np.ceil(heights/4)).sum()
    total_footprint = np.array(footprints).sum()

    fsi = gross_floor_area / area_of_aggregation
    gsi = total_footprint / area_of_aggregation
    osr = (1-gsi) / fsi
    mean_height = fsi / gsi
    tare = (area_of_aggregation - total_footprint) / area_of_aggregation

    return fsi, gsi, osr, mean_height, tare

def centroids(polygons):
    """
        A function that calculates the centroids of a collection of shapely polygons.

    Args:
        polygons (list): A list of shapely polygons.

    Returns:
        list: a list of centroids for all polygons.
    """
    centroids = []
    for polygon in polygons:
        xy = polygon.centroid.xy
        coords = np.dstack((xy[0], xy[1])).flatten()
        centroids.append(coords)

    return centroids

def find_intersections(seed_polygon, target_polygons):
    """
        A function that finds intersections between a seed polygon and a list of candidate polygons.

    Args:
        seed_polygon (shapely polygon): A shapely polygon.
        target_polygons (list): A list of shapely polygons.

    Returns:
        array: The intersection matrix between the seed polygon and all individual target polygons.
    """
    intersect_booleans = []
    for _, poly in enumerate(target_polygons):
        intersect_booleans.append(seed_polygon.intersects(poly))

    return intersect_booleans

def find_containments(seed_polygon, target_polygons):
    """
        A function that finds intersections between a seed polygon and a list of candidate polygons.

    Args:
        seed_polygon (shapely polygon): A shapely polygon.
        target_polygons (list): A list of shapely polygons.

    Returns:
        array: The intersection matrix between the seed polygon and all individual target polygons.
    """
    contain_booleans = []
    for _, poly in enumerate(target_polygons):
        contain_booleans.append(seed_polygon.contains(poly))

    return contain_booleans

def move_polygons(polygons, from_grid, to_grid=None):
    """
    A function that moves polygons to the origin bounding box (0, 250, 250, 250)
    """

    if(to_grid):
        x_displacement = (from_grid // 10) - (to_grid // 10)
        y_displacement = (from_grid % 10) - (to_grid % 10)
    else:
        x_displacement = from_grid // 10
        y_displacement = from_grid % 10

    moved_polygons = []
    for poly in polygons:
        moved_polygons.append(translate(poly, xoff=-x_displacement*250, yoff=-y_displacement*250))

    return moved_polygons

def move_neighbour_polygons(polygons, from_grid, to_grid=None):
    """
    A function that moves a neighbourhood to the origin bounding box (0, 250, 250, 250)
    """

    if(to_grid):
        x_displacement = (from_grid // 10) - (to_grid // 10)
        y_displacement = (from_grid % 10) - (to_grid % 10)
    else:
        x_displacement = from_grid // 10 - 1
        y_displacement = from_grid % 10 - 1

    moved_polygons = []
    for poly in polygons:
        moved_polygons.append(translate(poly, xoff=-x_displacement*250, yoff=-y_displacement*250))

    return moved_polygons