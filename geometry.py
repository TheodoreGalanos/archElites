import numpy as np
from shapely.geometry import Polygon, mapping
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import aggdraw

class Geometry:

	@staticmethod
	def create_path(polygon):
		"""
		A function that splits a series of polygon points into an (x, y) array of coordinates.
		:param polygon: A polygon (i.e. array of x, y sequences) creates by a design software or extracted from a geometry file.
		"""
		x_coords = polygon[0::2]
		y_coords = polygon[1::2]

		return np.hstack((x_coords, y_coords))

	@staticmethod
	def create_shapely_polygons(points, splits):
		"""
		A function that generates shapely polygons out of points and splits (indices on where to split the points) of each individual.
		:param points: a list of points for each individual.
		:param splits: a list of indices that show where to split the point list in order to create individual polygons.
		"""
		polygons = np.array(np.vsplit(points.reshape(-1, 1), np.cumsum(splits)))[:-1]

		shapes = []
		for poly in polygons:
			path = Geometry.create_path(poly)
			shapes.append(Polygon(path))

		return np.array(shapes)

	@staticmethod
	def find_intersections(seed_polygon, target_polygons):
		"""
		A function that finds intersections between a seed polygon and a list of candidate polygons.
		:param seed_polygon: A shapely polygon.
		:param target_polygons: A collection of shapely polygons.
		"""
		intersect_booleans = []
		for _, poly in enumerate(target_polygons):
			intersect_booleans.append(seed_polygon.intersects(poly))

		return intersect_booleans

	@staticmethod
	def centroids(polygons):
		"""
		A function that calculates the centroids of a collection of shapely polygons.
		:param polygons: A collection of shapely polygons.
		"""
		centroids = []
		for polygon in polygons:
			xy = polygon.centroid.xy
			coords = np.dstack((xy[0], xy[1])).flatten()
			centroids.append(coords)

		return centroids


	@staticmethod
	def get_features(footprints, heights, boundary=(512, 512), b_color="white"):
		"""
		Calculates urban features for a set of footprints and heights. Features include:
		floor space index (FSI): gross floor area / area of aggregation
		ground space index (GSI): footprint / area of aggregation
		oper space ratio (OSR): (1-GSI)/FSI
		building height (L): FSI/GSI
		tare (T): (area of aggregation - footprint) / area of aggregation
		:param footprints: list of areas for each building of an individual
		:param heights: list of heights for each building of an individual
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

	@staticmethod
	def draw_polygons(polygons, colors, im_size=(512,512), b_color="white", fpath=None):
	    image = Image.new("RGB", im_size, color="white")
	    draw = aggdraw.Draw(image)

	    for poly, color in zip(polygons, colors):
	        # get x, y sequence of coordinates for each polygon
	        xy = poly.exterior.xy
	        coords = np.dstack((xy[0], xy[1])).flatten()
	        # create a brush according to each polygon color
	        brush = aggdraw.Brush((color[0], color[1], color[2]), opacity=255)
	        draw.polygon(coords, brush)

	    image = Image.frombytes("RGB", im_size, draw.tobytes())
	    if(fpath):
	        image.save(fpath)

	    return draw, image
