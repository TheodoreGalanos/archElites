import glob
import pickle
import numpy as np

#local imports
from geometry import Geometry as geom
from utils import Utilities as util

class Collection:
	def __init__(self, folder, color_file, gh_vectors=False):
		"""
		A class to create a collection of individuals based on the genetic information extracted
		from the generative grasshopper model.

		:param folder: The location where the genetic information of the individuals are stored.
		:param color_file: The location where the color file for the collection was stored.
		"""

		#the location where individuals were stored
		self.folder = folder
		#the location where the color file was stored
		self.color_file = color_file

		#adding files that include individuals' genetic information to the collection
		self.points = glob.glob(folder  + "\points\*.npy", recursive=False)
		self.heights = glob.glob(folder + "\heights\*.npy", recursive=False)
		self.splits = glob.glob(folder +  "\splits\*.npy", recursive=False)

		#couple of sanity checks to make sure a proper collection location was passed as an argument
		if (len(self.points)) == len(self.heights) == len(self.splits) == 0:
				raise ValueError(f"Collection folder is empty.")

		if(len(self.points) != len(self.heights) != len(self.splits)):
			raise ValueError(f"A collection requires the same number of points, heights, and splits values.")

		#properties of the collection
		self.collection = {'points': self.points, 'heights': self.heights, 'splits': self.splits}
		self.collection_length = len(self.points)
		#extract genome information from grasshopper parameters embedded in the filenames
		if(gh_vectors):
			self.parameter_vectors = [point.split('\\')[-1].split('_')[:-1][1::2] for point in self.points]
		#the color map for later translation of genotypes to visual representation (heightmaps) of the phenotype
		self.cmap = util.create_cmap(color_file)


class Individual:
	def __init__(self, collection, id_, cmap, size):
		"""
		A class to create an individual, along with its properties, out of the provided collection.

		:param id_: The indice of the individual in the collection
		:param cmap: The color gradient map that represents heights into colors, as it was extracted from
		the grasshopper model.
		:param size: The extend of the bounding box of an individual, in meters. Used to generate appropriate
		image outputs.
		"""

		self.collection = collection
		self.id_ = id_
		self.size = size
		self.parent_ids = None
		self.grid_position = None

		# get individual's genome properties from collection
		self.points  = np.load(collection.points[id_])
		self.heights = np.load(collection.heights[id_])
		self.splits  = np.load(collection.splits[id_])

		# generate phenotype
		self.colors  = np.array([util.height_to_color(cmap, height) for height in self.heights])
		self.polygons = geom.create_shapely_polygons(self.points, self.splits)

		# calcualte features and descriptors
		self.footprints = [polygon.area for polygon in self.polygons]
		self.feature_names = ['FSI', 'GSI', 'OSR', 'Mean_height', 'Tare']
		self.features = dict(zip(self.feature_names, geom.get_features(self.footprints, self.heights)))
		self.centroids = np.array(geom.centroids(self.polygons))
		self.std = util.calc_std(self.heights)
		self.dangerous = None
		self.sitting = None

	def draw_image(self):

		_, image = geom.draw_polygons(self.polygons, self.colors, self.size)

		return image

	def save_to_disk(self, fname):

		data = {'polygons': self.polygons, 'heights': self.heights,
				'colors': self.colors, 'footprints:': self.footprints,
				'features': self.features, 'parent_id': self.parent_ids,
				'grid_position': self.grid_position}

		with open(fname, 'wb') as file:
			pickle.dump(data, file)


class Offspring:
	def __init__(self, polygons, colors, heights, size, parent_ids):
		"""
		A class to create an offspring, along with its properties, out of the crossover or mutation
		of individuals.

		:param polygons: The polygons of the evolved individual.
		:param colors: The color of the evolved individual.
		:param heights: The heights of the evolved individual.
		:param size: The extend of the bounding box of an individual, in meters. Used to generate appropriate
		image outputs.
		"""

		# assign genome
		self.colors  = colors
		self.polygons = polygons
		self.heights= heights
		self.size = size
		self.parent_ids = parent_ids

		#assign position on the map
		self.grid_position = None

		#calculate phenotype
		self.footprints = [polygon.area for polygon in self.polygons]
		self.feature_names = ['FSI', 'GSI', 'OSR', 'Mean_height', 'Tare']
		self.features = dict(zip(self.feature_names, geom.get_features(self.footprints, self.heights)))
		self.std = util.calc_std(self.heights)
		self.centroids = geom.centroids(self.polygons)
		self.dangerous = None
		self.sitting = None

	def draw_image(self):

		_, image = geom.draw_polygons(self.polygons, self.colors, self.size)

		return image

	def save_to_disk(self, fname):

		data = {'polygons': self.polygons, 'heights': self.heights,
				'colors': self.colors, 'footprints:': self.footprints,
				'features': self.features, 'parent_id': self.parent_ids,
				'grid_position': self.grid_position}

		with open(fname, 'wb') as file:
			pickle.dump(data, file)
