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

	@staticmethod
	def rotate_input(pil_img, degrees, interval=512):
		"""Method to rotate image by `degrees` in a COUNTER-CLOCKWISE direction.
		As some rotations cause the corners of the original image to be cropped,
		the `interval` argument allows the image to expand in size.
		"""
		def next_interval(current):
			c = int(current)
			if c % interval == 0:
				return c
			else:
				return interval * ((c // interval) + 1)

		def paste_top_left_coords(rot_width, rot_height, exp_width, exp_height):
			calc = lambda r, e: int((e - r) / 2)
			return calc(rot_width, exp_width), calc(rot_height, exp_height)

		if pil_img.mode != 'RGB':
			pil_img = pil_img.convert('RGB')

		degrees = degrees % 360
		if degrees % 90 != 0:
			rot_img = pil_img.rotate(
				angle=degrees,
				resample=Image.BICUBIC,
				expand=1,
				fillcolor=(255, 255, 255)
			)
			min_width, min_height = rot_img.size
			exp_width  = next_interval(min_width)
			exp_height = next_interval(min_height)
			pil_img = Image.new('RGB', (exp_width, exp_height), (255, 255, 255))
			paste_coords = paste_top_left_coords(min_width, min_height,
												exp_width, exp_height)
			pil_img.paste(rot_img, paste_coords)
		else:
			pil_img = pil_img.rotate(
				angle=degrees,
				resample=Image.BICUBIC,
				fillcolor=(255, 255, 255)
			)
		return pil_img

	def rotate_to_origin(pil_img, original_height, original_width, degrees):
		rot_img = pil_img.rotate(
			angle=degrees,
			resample=Image.BICUBIC,
			expand=1,
			fillcolor=(255, 255, 255)
		)
		rot_width, rot_height = rot_img.size
		return rot_img.crop((
			(rot_width  - original_width)  / 2,
			(rot_height - original_height) / 2,
			(rot_width  - original_width)  / 2 + original_width,
			(rot_height - original_height) / 2 + original_height
		))