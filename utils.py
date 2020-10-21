from math import floor, ceil
from random import uniform
import numpy as np
from scipy.spatial import cKDTree

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import aggdraw

from sklearn.metrics.pairwise import cosine_similarity
from random import sample

class Utilities:


	@staticmethod
	def calc_std(attribute):

		std = np.std(attribute)

		return std

	@staticmethod
	def genetic_similarity(ind_files, n_genes):
		"""
		Calculates similarity between individuals created in Grasshopper, using information
		(input parameters) saved in each individual's name convention.
		:param ind_files: A Collection of individuals generated through grasshopper.
		:param n_genes: The total number of parameters in the Grasshopper parametric model.
		"""
		genomes = []
		for file in ind_files:
			genomes.append(file.split('\\')[-1].split('_')[:-1][1::2])

		genomes = np.array(genomes).reshape(-1, n_genes).astype(int)
		similarity = cosine_similarity(genomes)

		return similarity

	@staticmethod
	def diverse_pairs(similarity_matrix, n_pairs):
		"""
		A function to find the indices of a specified number of individuals, for each parent, which
		are as different as possible from the parent, based on their genetic similarity.
		:param similarity_matrix: The similarity matrix computed for all individuals.
		:param n_pairs: The number of most dissimilar individuals to find for each parent.
		"""
		diverse_pairs = []
		for ind in similarity_matrix:
			diverse_pairs.append(np.argpartition(ind, n_pairs)[:n_pairs])

		return diverse_pairs

	@staticmethod
	def create_cmap(color_file):
		"""
		Creates an RGB color map out of a list of color values
		:param color_file: A csv file of the color gradient used to generate the heightmaps.
		"""
		colors = np.loadtxt(color_file, dtype=str)
		cmap = []
		for color in colors:
			cmap.append(color.split(',')[0])

		cmap = np.array(cmap, dtype=int)

		return cmap

	@staticmethod
	def height_to_color(cmap, height):
		"""
		Translates a building height value to a color, based on the given color map.
		:param cmap: A color map.
		:param height: A building height
		"""
		if(height > len(cmap)-1):
			color_value = 0
		else:
			modulo = height % 1
			if(modulo) == 0:
				color_value = cmap[height]
			else:
				minimum = floor(height)
				maximum = ceil(height)

				min_color = cmap[minimum+1]
				max_color = cmap[maximum+1]

				color_value = min_color + ((min_color-max_color) * modulo)

		return [color_value, color_value, color_value]

	@staticmethod
	def draw_polygons(polygons, colors, im_size=(512, 512), b_color="white", fpath=None):
		"""
		A function that draws a PIL image of a collection of polygons and colors.
		:param polygons: A list of shapely polygons.
		:param colors: A list of R, G, B values for each polygon.
		:param im_size: The size of the input geometry.
		:param b_color: The color of the image background.
		:param fpath: The file path to use if saving the image to disk.
		"""
		image = Image.new("RGB", im_size, color=b_color)
		draw = aggdraw.Draw(image)

		for poly, color in zip(polygons, colors):
			# get x, y sequence of coordinates for each polygon
			xy = poly.exterior.xy
			coords = np.dstack((xy[0], xy[1])).flatten()
			# create a brush according to each polygon color
			brush = aggdraw.Brush((color[0], color[1], color[2]), opacity=255)
			# draw the colored polygon on the draw object
			draw.polygon(coords, brush)

		#create a PIL image out of the aggdraw object
		image = Image.frombytes("RGB", im_size, draw.tobytes())

		if(fpath):
			image.save(fpath)

		return draw, image

	@staticmethod
	def get_poly_ids(polygons, random_genes, indgen):

		if(random_genes):
			keep_ = int(uniform(0,1))
			poly_ids = sample(list(np.arange(0, len(polygons))), keep_)
		else:
			poly_ids = sample(list(np.arange(0, len(polygons))), int(len(polygons) * indgen))

		return poly_ids