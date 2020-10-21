import time
from time import sleep
import random
import pickle
import operator
import logging
import configparser
import random
from tqdm import tqdm
from shutil import copyfile
from datetime import datetime
from pathlib import Path
import numpy as np
np.set_printoptions(precision=2)

from abc import ABC, abstractmethod

#local imports
from ea_operators import EaOperators as eaop
from population import Collection, Individual, Offspring
from utils import Utilities as util
from inference import run_inf
from evaluation import evaluate_fn
from plot_utils import plot_heatmap

class MapElites(ABC):

	def __init__(self,
			     size,
				 color_file,
				 n_genes,
				 n_pairs,
				 n_curious,
				 iterations,
				 descriptors,
				 bootstrap_individuals,
				 mutation_op,
				 mutation_args,
				 crossover_op,
				 crossover_args,
				 n_bins,
				 bins,
				 plot_args,
				 input_dir,
				 log_dir,
				 config_path,
				 minimization=True
				 ):
		"""
		:param iterations: Number of evolutionary iterations
		:param n_genes: The number of different parameters in the generative model that created the geometries
		:param n_pairs: The number of distant individuals that we want to find.
		:param bootstrap_individuals: Number of individuals randomly generated to bootstrap the algorithm
        :param mutation_args: Mutation function arguments
        :param crossover_op: Crossover function
        :param crossover_args: Crossover function arguments
        :param bins: Bins for feature dimensions
        :param minimization: True if solving a minimization problem. False if solving a maximization problem.
		"""

		#collection specific
		self.input_dir = input_dir
		self.size = size
		self.n_genes = n_genes
		self.n_pairs = n_pairs
		self.n_curious = n_curious
		self.color_file = color_file
		self.collection = Collection(input_dir, self.color_file)
		self.cmap = util.create_cmap(color_file)
		self.elapsed_time = 0

		self.minimization = minimization
		#set the choice operator either to do minimization or a maximization
		if self.minimization:
			self.place_operator = operator.lt
		else:
			self.place_operator = operator.ge

		self.plot_args = plot_args

		self.iterations = iterations
		self.descriptors = descriptors
		self.bootstrap_individuals = bootstrap_individuals
		self.n_bins = n_bins
		self.bins = bins
		if not (len(self.bins) == len(descriptors)):
			raise Exception(
				f"Length of bins needs to match the number of descriptor dimensions ")

		self.mutation_op = mutation_op
		self.mutation_args = mutation_args
		self.crossover_op = crossover_op
		self.crossover_args = crossover_args
		#get the number of bins for each feature dimension
		ft_bins = [len(bin)-1 for bin in self.bins.values()]
		ft_bins = [ft_bins[int(self.n_bins[0])], ft_bins[int(self.n_bins[1])]]

		#Map of Elites: Initialize data structures to store solutions and fitness values
		#self.solutions = np.full(ft_bins, -np.inf, dtype=(float))
		if(self.minimization):
			self.performances = np.full(ft_bins, np.inf, dtype=(float))
		else:
			self.performances = np.full(ft_bins, -np.inf, dtype=(float))
		self.curiosity_score = np.full(ft_bins, -np.inf, dtype=(float))
		self.fpath_ids = np.full(ft_bins, 0, dtype=(int))
		self.previous_locations = []
		self.current_locations = []

		if log_dir:
			self.log_dir_path = Path(log_dir)
		else:
			now = datetime.now().strftime("%Y%m%d%H%M%S")
			self.log_dir_name = f"log_{now}"
			self.log_dir_path = Path(f'logs/{self.log_dir_name}')
		#create log dir
		self.log_dir_path.mkdir(parents=True, exist_ok=True)
		#create initial population dir
		genomes = self.log_dir_path / 'genomes' / 'initial_population'
		genomes.mkdir(parents=True, exist_ok=True)
		#save config file
		copyfile(config_path, self.log_dir_path / 'config.ini')

		#setup logging
		self.logger = logging.getLogger('map elites')
		self.logger.setLevel(logging.DEBUG)
		# create file handler which logs even debug messages
		fh = logging.FileHandler(self.log_dir_path / 'log.log', mode='w')
		fh.setLevel(logging.INFO)
		self.logger.addHandler(fh)
		self.logger.info("Configuration completed.")

	@classmethod
	def from_config(cls, config_path, log_dir=None, func=None, overwrite=False):
		"""
		Read config file and create a MAP-Elites instance.
		:param config_path: Path to config.ini file
		:param log_dir: Absolute path to logging directory
		:param func: Name of optimization function to use
		:param overwrite: Overwrite the log directory if already exists
		"""

		#read configuration file
		config = configparser.ConfigParser()
		config.read(config_path)

		#collection config
		input_dir = config['collection'].get('input_dir')
		color_file = config['collection'].get('color_file')
		size_x = config['collection'].getint('size_x')
		size_y = config['collection'].getint('size_y')
		size = (size_x, size_y)
		n_genes = config['collection'].getint('n_genes')
		n_pairs = config['collection'].getint('n_pairs')

		#main mapelites config
		iterations = config['mapelites'].getint('iterations')
		n_curious = config['mapelites'].getint('n_curious')
		bootstrap_individuals = config['mapelites'].getint('bootstrap_individuals')
		minimization = config['mapelites'].getboolean('minimization')

		#plotting config
		plot_args = dict()
		plot_args['highlight_best'] = config['plotting'].getboolean('highlight_best')
		plot_args['interactive'] = config['mapelites'].getboolean('interactive')

		#bins
		d = dict(config.items('quality_diversity'))

		bins_names = filter(lambda s: s.startswith("bin"), d.keys())
		bins = {_k: d[_k] for _k in bins_names}

		#descriptors
		descriptors = config['quality_diversity'].get('dimensions').split(',')
		n_bins = config['quality_diversity'].get('n_bins').split(',')
		#step = config['quality_diversity'].getfloat('step')
		step_fsi = config['quality_diversity'].getfloat('step_fsi')
		step_gsi = config['quality_diversity'].getfloat('step_gsi')
		step_osr = config['quality_diversity'].getfloat('step_osr')
		step_mh = config['quality_diversity'].getfloat('step_mh')
		step_tare = config['quality_diversity'].getfloat('step_tare')
		step_dangerous = config['quality_diversity'].getfloat('step_dangerous')
		step_sitting = config['quality_diversity'].getfloat('step_sitting')
		steps = [step_fsi, step_gsi, step_osr, step_mh, step_tare, step_dangerous, step_sitting]
		# substitute strings "inf" at start and end of bins with -np.inf and np.inf
		it=0
		for k,v in bins.items():
			b = v.split(',')
			edges = list(map(float, b[1:-1]))
			b = np.arange(edges[0], edges[1]+steps[it], steps[it], dtype=float)
			b = np.insert(b, (0,len(b)), [-np.inf, np.inf])
			bins[k] = b
			it +=1

		mutation_args = None
		mutation_op = config['mutation']['type']
		if mutation_op == "polynomial":
			mutation_args = {
				"eta": config['mutation'].getfloat('eta'),
				"low": config['mutation'].getfloat('low'),
				"up": config['mutation'].getfloat('up'),
				"mut_pb": config['mutation'].getfloat('mut_pb')
			}

		crossover_op = config['crossover']['type']
		if crossover_op == "uniform":
			crossover_args = {
				"indpb": config['crossover'].getfloat('indpb'),
				"indgen": config['crossover'].getfloat('indgen')
			}

		return cls(
			n_genes=n_genes,
			n_pairs=n_pairs,
			n_curious=n_curious,
			size=size,
			input_dir=input_dir,
			color_file=color_file,
			iterations=iterations,
			descriptors=descriptors,
			bootstrap_individuals=bootstrap_individuals,
			mutation_op=mutation_op,
			mutation_args=mutation_args,
			crossover_op=crossover_op,
			crossover_args=crossover_args,
			minimization=minimization,
			plot_args=plot_args,
			log_dir=log_dir,
			config_path=config_path,
			n_bins=n_bins,
			bins=bins
		)


	def generate_initial_population(self, inference=False):

		self.logger.info("Generating initial population from the collection")
		population_ids = list(np.arange(0, (len(self.collection.points))))
		seed_ids = random.sample(population_ids, self.bootstrap_individuals)
		collection = Collection(self.input_dir, self.color_file)
		for id_ in seed_ids:
			ind = Individual(collection, id_, collection.cmap, self.size)
			if(inference):
				self.place_in_mapelites_with_inference(ind)
			else:
				self.place_in_mapelites(ind)

	def crossover(self, individuals=2, iteration=None, mut_gen=None, curiosity=False):
		"""
		Generate an offspring out of two individuals in the collection with high curiosity score
		"""

		if(curiosity):
		#curiosity approach
			x_indxs, y_indxs = self.curiosity_score_selection(n_curious=self.n_curious, individuals=individuals)
			ind1_loc = self.current_locations[self.fpath_ids[x_indxs[0], y_indxs[0]]]
			ind2_loc = self.current_locations[self.fpath_ids[x_indxs[1], y_indxs[1]]]
		else:
		#random sampling apporach
			idxs = self.random_selection(individuals=individuals)
			ind1_loc = self.current_locations[self.fpath_ids[idxs[0]]]
			ind2_loc = self.current_locations[self.fpath_ids[idxs[1]]]

		with open(ind1_loc, 'rb') as ind1_pickle:
			content1 = pickle.load(ind1_pickle)

			polygons1 = content1['polygons']
			colors1 = content1['colors']
			heights1 = content1['heights']
			# get current grid position as parent id of the offspring
			parent_ids1 = content1['grid_position']

			ind1 = Offspring(polygons1, colors1, heights1, self.size, parent_ids1)

		with open(ind2_loc, 'rb') as ind2_pickle:
			content2 = pickle.load(ind2_pickle)

			polygons2 = content2['polygons']
			colors2 = content2['colors']
			heights2 = content2['heights']
			parent_ids2 = content2['grid_position']

			ind2 = Offspring(polygons2, colors2, heights2, self.size, parent_ids2)

		#offspring = eaop.uniform_crossover_geometry(ind1, ind2, self.crossover_args["indgen"]+mut_gen, self.crossover_args["indpb"], parent_ids=[ind1.parent_ids, ind2.parent_ids], random_genes=True)
		offspring_1, offspring_2 = eaop.uniform_crossover_geometry(ind1, ind2, self.crossover_args["indgen"]+mut_gen, self.crossover_args["indpb"], parent_ids=[ind1.parent_ids, ind2.parent_ids], random_genes=True)
		#return offspring
		return offspring_1, offspring_2

	def random_selection(self, individuals=1):
		"""
		Select an elite x from the current map of elites.
		The selection is done by selecting a random bin for each feature
		dimension, until a bin with a value is found.
		:param individuals: The number of individuals to randomly select
		:return: A list of N random elites
		"""

		def _get_random_index():
			"""
			Get a random cell in the N-dimensional feature space
			:return: N-dimensional tuple of integers
			"""
			indexes = tuple()
			for k, v in self.bins.items():
				if('sitting' in k or 'dangerous' in k):
					rnd_ind = np.random.randint(0, len(v)-1, 1)[0]
					indexes = indexes + (rnd_ind, )

			return indexes

		def _is_not_initialized(index):
			"""
			Checks if the selected index points to a NaN or Inf solution (not yet initialized)
			The solution is considered as NaN/Inf if any of the dimensions of the individual is NaN/Inf
			:return: Boolean
			"""
			return any([self.performances[index] == np.nan or np.abs(self.performances[index]) == np.inf])

		# individuals
		idxs = list()
		for _ in range(0, individuals):
			idx = _get_random_index()
			# we do not want to repeat entries
			while idx in idxs or _is_not_initialized(idx):
				idx = _get_random_index()
			idxs.append(idx)
		return idxs

	def curiosity_score_selection(self,  n_curious, individuals=2):
		"""
		Select an elite x from the current map of elites.
		The selection is based on  based on how many
		offspring it has added to the collection.
		:param individuals: The number of individuals to randomly select
		:return: A list of N random elites
		"""
		if(n_curious<=individuals):
			raise Exception (f"Sampled population needs to be larger than selection")
		elif(n_curious>= self.bootstrap_individuals):
			raise Exception (f"Sampled population needs to be smaller than the number of bootstrapped individuals during initialization")
		else:
			curiosity_flat = self.curiosity_score.flatten()
			curious_indxs = curiosity_flat.argsort()[-n_curious:]

			# convert the idx_1d back into indices arrays for each dimension
			x_idx, y_idx = np.unravel_index(curious_indxs, self.curiosity_score.shape)
			# select individuals from the sample
			rnd_ind = random.sample((0, len(x_idx)-1), individuals)
			x_selected, y_selected = x_idx[rnd_ind], y_idx[rnd_ind]

		return x_selected, y_selected


	def place_in_mapelites(self, x, pbar=None, init=True, iteration=None):
		"""
		Puts a solution inside the N-dimensional map of elites space.
		The following criteria is used:

		- Compute the feature descriptor of the solution to find the correct
				cell in the N-dimensional space
		- Compute the performance of the solution
		- Check if the cell is empty or if the previous performance is worse
			- Place new solution in the cell
		:param x: genotype of an individual
		:param pbar: TQDM progress bar instance
		"""

		#get coordinates in the feature space
		x_bin, y_bin = self.map_x_to_b(x)
		# save the parent id of the individual
		x.grid_position = [x_bin, y_bin]
		# performance of the optimization function
		sitting, dangerous, sitting_area, dangerous_area = self.performance_measure(x, fpath=r'./inferences_server/inferences/individual.png', exp_folder=r'F:\PhD_Research\CaseStudies\MAP-Elites\pv_urban\inference_server\minimal-ml-serverCPU\inferences')
		# save individual performance
		x.sitting = sitting_area
		x.dangerous = dangerous_area

		# place operator performs either minimization or maximization
		if self.place_operator(sitting, self.performances[x_bin, y_bin]):
			self.logger.debug(f"PLACE: Placing individual {x} at {x_bin, y_bin} with perf: {sitting}")

			# if this is during initalization simply update the grid
			if(init):
				# update performance score for the added individual
				self.performances[x_bin, y_bin] = sitting
				# reset curiosity score at the individual's grid position
				self.curiosity_score[x_bin, y_bin] = 0.0
				#save individual to disk for later use
				fpath = self.log_dir_path / 'genomes' / 'initial_population' / 'individual_{}_{}.pkl'.format(x_bin, y_bin)
				x.parent_ids = x.grid_position
				x.save_to_disk(fpath)
				# save individual's location on the disk
				self.current_locations.append(str(fpath.absolute()))
				# save individual's location id
				self.fpath_ids[x_bin, y_bin] = len(self.current_locations)-1
			else:
				self.performances[x_bin, y_bin] = sitting
				fpath = self.log_dir_path / 'genomes' / 'generation_{}'.format(iteration) / 'individual_{}_{}.pkl'.format(x_bin, y_bin)
				x.save_to_disk(fpath)
				self.current_locations.append(str(fpath.absolute()))
				self.fpath_ids[x_bin, y_bin] = len(self.current_locations)-1
				# increase curiosity score for the parents
				for parents in x.parent_ids:
					self.curiosity_score[parents[0], parents[1]] += 1.0
		else:
			if(init):
				self.logger.debug(f"PLACE: Individual {x} rejected at {x_bin, y_bin} with perf: {sitting} in favor of {self.performances[x_bin, y_bin]}")
			else:
				self.logger.debug(f"PLACE: Offspring {x} rejected at {x_bin, y_bin} with perf: {sitting} in favor of {self.performances[x_bin, y_bin]}")
				for parents in x.parent_ids:
					self.curiosity_score[parents[0], parents[1]] -= 0.5

		if pbar is not None:
			sleep(0.5)
			pbar.update(1)


	def place_in_mapelites_with_inference(self, x, pbar=None, init=True, iteration=None):
		"""
		Puts a solution inside the N-dimensional map of elites space.
		The following criteria is used:

		- Compute the feature descriptor of the solution to find the correct
				cell in the N-dimensional space
		- Compute the performance of the solution
		- Check if the cell is empty or if the previous performance is worse
			- Place new solution in the cell
		:param x: genotype of an individual
		:param pbar: TQDM progress bar instance
		"""

		#get coordinates in the feature space from individual's performance
		sitting, dangerous, sitting_area, dangerous_area = self.performance_measure(x, fpath=r'./inferences_server/inferences/individual.png', exp_folder=r'F:\PhD_Research\CaseStudies\MAP-Elites\pv_urban\inference_server\minimal-ml-serverCPU\inferences')
		#get coordinates in the feature space
		x_bin, y_bin = self.map_x_to_b_inference(sitting_area, dangerous_area)
		#x_bin, y_bin = self.map_x_to_b_inference(sitting*100, dangerous*100)
		# save the parent id of the individual
		x.grid_position = [x_bin, y_bin]
		# save individual performance
		x.sitting = sitting_area
		x.dangerous = dangerous_area

		# place operator performs either minimization or maximization
		if self.place_operator(x.features['FSI'], self.performances[x_bin, y_bin]):
			self.logger.debug(f"PLACE: Placing individual {x} at {x_bin, y_bin} with perf: {x.features['FSI']}")

			# if this is during initalization simply update the grid
			if(init):
				# update performance score for the added individual
				self.performances[x_bin, y_bin] = x.features['FSI']
				# reset curiosity score at the individual's grid position
				self.curiosity_score[x_bin, y_bin] = 0.0
				#save individual to disk for later use
				fpath = self.log_dir_path / 'genomes' / 'initial_population' / 'individual_{}_{}.pkl'.format(x_bin, y_bin)
				x.parent_ids = x.grid_position
				x.save_to_disk(fpath)
				# save individual's location on the disk
				self.current_locations.append(str(fpath.absolute()))
				# save individual's location id
				self.fpath_ids[x_bin, y_bin] = len(self.current_locations)-1
			else:
				self.performances[x_bin, y_bin] = x.features['FSI']
				fpath = self.log_dir_path / 'genomes' / 'generation_{}'.format(iteration) / 'individual_{}_{}.pkl'.format(x_bin, y_bin)
				x.save_to_disk(fpath)
				self.current_locations.append(str(fpath.absolute()))
				self.fpath_ids[x_bin, y_bin] = len(self.current_locations)-1
				# increase curiosity score for the parents
				for parents in x.parent_ids:
					self.curiosity_score[parents[0], parents[1]] += 1.0
		else:
			if(init):
				self.logger.debug(f"PLACE: Individual {x} rejected at {x_bin, y_bin} with perf: {x.features['FSI']} in favor of {self.performances[x_bin, y_bin]}")
			else:
				self.logger.debug(f"PLACE: Offspring {x} rejected at {x_bin, y_bin} with perf: {x.features['FSI']} in favor of {self.performances[x_bin, y_bin]}")
				for parents in x.parent_ids:
					self.curiosity_score[parents[0], parents[1]] -= 0.5

		if pbar is not None:
			sleep(0.5)
			pbar.update(1)

	def map_x_to_b_inference(self, x_metric, y_metric):
		"""
		Function to map a solution x to feature space dimensions
		:param x: genotype of a solution
		:return: phenotype of the solution (tuple of indices of the N-dimensional space)
		"""
		"""
		x_metric, y_metric
		if(y_metric > 50):
			y_metric = 50
		if(x_metric > 50):
			x_metric = 50
		"""
		# decide who are the behavioral dimensions
		#TODO: this should not be hard coded
		x_bin = np.digitize(x_metric, self.bins['bin_sitting'], right=True)
		y_bin = np.digitize(y_metric, self.bins['bin_dangerous'], right=True)
		#x_bin = np.digitize(osr, self.bins['bin_osr'], right=True)
		#y_bin = np.digitize(mh, self.bins['bin_mh'], right=True)
		#x_bin = np.digitize(fsi, self.bins['bin_fsi'], right=True)
		#y_bin = np.digitize(gsi, self.bins['bin_gsi'], right=True)

		return x_bin, y_bin

	def map_x_to_b(self, x):
		"""
		Function to map a solution x to feature space dimensions
		:param x: genotype of a solution
		:return: phenotype of the solution (tuple of indices of the N-dimensional space)
		"""
		features = []
		for k, v in x.features.items():
			features.append(v)

		fsi, gsi, osr, mh, tare = features[0], features[1], features[2], features[3], features[4]
		if(fsi > 8):
			fsi = 8
		if(gsi > 0.7):
			gsi = 0.7
		if(osr > 1.0):
			osr = 1.0
		if(mh > 25.0):
			mh = 25.0
		if(tare > 1.0):
			tare = 1.0

		# decide who are the behavioral dimensions
		#TODO: this should not be hard coded
		#x_bin = np.digitize(osr, self.bins['bin_osr'], right=True)
		#y_bin = np.digitize(mh, self.bins['bin_mh'], right=True)
		x_bin = np.digitize(fsi, self.bins['bin_fsi'], right=True)
		y_bin = np.digitize(gsi, self.bins['bin_gsi'], right=True)

		return x_bin, y_bin

	def performance_measure(self, x, fpath, exp_folder):
		"""
		Function to evaluate solution x and give a performance measure
		:param x: genotype of a solution
		:return: performance measure of that solution
		"""
		#NOTE: FIX THE TWO DIFFERENT RUN_INF SCRIPTS, one for init and one for a pair.

		image = x.draw_image()
		image.save(r'F:\PhD_Research\CaseStudies\MAP-Elites\pv_urban\inference_server\minimal-ml-serverCPU\inferences\individual.png')
		run_inf()
		sitting, dangerous, sitting_area, dangerous_area = evaluate_fn(exp_folder)

		return sitting, dangerous, sitting_area, dangerous_area

	def save_logs(self):
		"""
		Save logs, config file, individuals, and data structures to log folder.

		:param iteration: The current iteration of the algorithm.
		"""

		self.logger.info(f"Running time {time.strftime('%H:%M:%S', time.gmtime(self.elapsed_time))}")

		np.save(self.log_dir_path / 'performances', self.performances)
		#np.save(self.log_dir_path / "solutions", self.solutions)
		np.save(self.log_dir_path / 'curiosity', self.curiosity_score)

	def plot_map_of_elites(self, iteration):
		"""
		Plot a heatmap of elites
		"""
		# Stringify the bins to be used as strings in the plot axes
		x_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[5])]]
		y_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[6])]]

#		x_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[0])]]
#		y_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[1])]]

		plot_heatmap(self.performances,
					 x_ax,
					 y_ax,
					 v_min=0.0,
					 v_max=1.0,
					 savefig_path=self.log_dir_path,
					 iteration=iteration,
					 title=f"Urban Comfort QD Map of Elites",
					 **self.plot_args)

	def get_elapsed_time(self):
		return self.elapsed_time

	def run(self):
		"""
		Main iteration loop of MAP-Elites
		"""
		start_time = time.time()
		# start by creating an initial set of random solutions
		self.generate_initial_population(inference=False)
		self.plot_map_of_elites(iteration='initial')

		outter_bar = tqdm(total=self.iterations, desc="Iterations completed", position = 0, leave = True)
		outter_loop = range(0, self.iterations)
		inner_loop = range(0, 15)

		for i in outter_loop:
			# create the save folder for the generation
			folder = self.log_dir_path / 'genomes' / 'generation_{}'.format(i)
			folder.mkdir(parents=True, exist_ok=True)
			for j in inner_loop:
			# create more diverse mutations
				if (j%2 == 0):
					self.logger.debug(f"ITERATION {i} - Individual {j}")
					self.logger.debug("Select and mutate.")
					# get the number of elements that have already been initialized
					offspring_1, offspring_2 = self.crossover(individuals=2, iteration=i, mut_gen=0.25)#, curiosity=True)
					offspring_mut_1 = eaop.polynomial_bounded(offspring_1, self.cmap, eta=20.0, low=5.0, up=100.0, mut_pb=1/len(offspring_1.heights))
					offspring_mut_2 = eaop.polynomial_bounded(offspring_2, self.cmap, eta=20.0, low=5.0, up=100.0, mut_pb=1/len(offspring_2.heights))
					# place the new individual in the map of elites
					self.place_in_mapelites(offspring_mut_1, pbar=outter_bar.set_postfix(inner_loop=j, refresh=True), init=False, iteration=i)
					self.place_in_mapelites(offspring_mut_2, pbar=outter_bar.set_postfix(inner_loop=j, refresh=True), init=False, iteration=i)
				else:
					self.logger.debug(f"ITERATION {i} - Individual {j}")
					self.logger.debug("Select and mutate.")
					# get the number of elements that have already been initialized
					offspring_1, offspring_2 = self.crossover(individuals=2, iteration=i, mut_gen=-0.25)#, curiosity=True)
					offspring_mut_1 = eaop.polynomial_bounded(offspring_1, self.cmap, eta=20, low=5.0, up=100.0, mut_pb=1/len(offspring_1.heights))
					offspring_mut_2 = eaop.polynomial_bounded(offspring_2, self.cmap, eta=20, low=5.0, up=100.0, mut_pb=1/len(offspring_2.heights))
					# place the new individual in the map of elites
					self.place_in_mapelites(offspring_mut_1, pbar=outter_bar.set_postfix(inner_loop=j, refresh=True), init=False, iteration=i)
					self.place_in_mapelites(offspring_mut_2, pbar=outter_bar.set_postfix(inner_loop=j, refresh=True), init=False, iteration=i)

			#inner_bar.reset()
			self.plot_map_of_elites(iteration=i)
			outter_bar.update(1)

		# save results, display metrics and plot statistics
		end_time = time.time()
		self.elapsed_time = end_time - start_time
		self.save_logs()
		self.plot_map_of_elites(iteration=self.iterations)
