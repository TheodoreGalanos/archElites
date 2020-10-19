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
				 iterations,
				 descriptors,
				 bootstrap_individuals,
				 mutation_op,
				 mutation_args,
				 crossover_flag,
				 crossover_op,
				 crossover_fun,
				 crossover_args,
				 n_bins,
				 bins,
				 plot_args,
				 input_dir,
				 log_dir,
				 config_path,
				 seed,
				 minimization=True
				 ):
		"""
		:param iterations: Number of evolutionary iterations
		:param n_genes: The number of different parameters in the generative model that created the geometries
		:param n_pairs: The number of distant individuals that we want to find.
		:param bootstrap_individuals: Number of individuals randomly generated to bootstrap the algorithm
        :param mutation_op: Mutation function
        :param mutation_args: Mutation function arguments
        :param crossover_flag: Flag to activate crossover behavior
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
		self.color_file = color_file
		self.collection = Collection(input_dir, self.color_file)
		self.cmap = util.create_cmap(color_file)
		#set random seed
		self.seed = seed
		np.random.seed(self.seed)
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
		self.crossover_flag = crossover_flag
		self.crossover_op = crossover_op
		self.crossover_args = crossover_args
		self.crossover_fun = crossover_fun
		#get the number of bins for each feature dimension
		ft_bins = [len(bin)-1 for bin in self.bins.values()]
		ft_bins = [ft_bins[int(self.n_bins[0])], ft_bins[int(self.n_bins[1])]]

		#Map of Elites: Initialize data structures to store solutions and fitness values
		#self.solutions = np.full(ft_bins, -np.inf, dtype=(float))
		self.performances = np.full(ft_bins, np.inf)
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
		self.logger.info(f"Using random seed {self.seed}")
		print(f"\tUsing random seed {self.seed}")

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

		#random seed
		seed = config['mapelites'].getint('seed')
		if not seed:
			seed = np.random.randint(0, 100)

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
		steps = [step_fsi, step_gsi, step_osr, step_mh, step_tare]
		# substitute strings "inf" at start and end of bins with -np.inf and np.inf
		it=0
		for k,v in bins.items():
			b = v.split(',')
			edges = list(map(float, b[1:-1]))
			b = np.arange(edges[0], edges[1]+steps[it], steps[it], dtype=float)
			b = np.insert(b, (0,len(b)), [-np.inf, np.inf])
			bins[k] = b
			it +=1

		# EA OPERATORS
		ea_operators = [func for func in dir(eaop)
						if callable(getattr(eaop, func))
						and not func.startswith("__", 0, 2)
						]

		#mutation and crossover ops
		mutation_op = config['mutation']['type']
		mutation_fun = f"{str.lower(mutation_op)}_bounded"
		if mutation_fun not in ea_operators:
			raise ValueError(f"Mutation operator {mutation_op} not implemented.")
		mutation_fun = getattr(eaop, mutation_fun)

		mutation_args = None
		if mutation_op == "polynomial":
			mutation_args = {
				"eta": config['mutation'].getfloat('eta'),
				"low": config['mutation'].getfloat('low'),
				"up": config['mutation'].getfloat('up'),
				"mut_pb": config['mutation'].getfloat('mut_pb')
			}

		crossover_flag = config['crossover'].getboolean("crossover")
		crossover_op = config['crossover']['type']
		crossover_fun = f"{str.lower(crossover_op)}_crossover_geometry"
		if crossover_fun not in ea_operators:
			raise ValueError(f"Crossover operator {crossover_op} not implemented.")
		crossover_fun = getattr(eaop, crossover_fun)
		if crossover_op == "uniform":
			crossover_args = {
				"indpb": config['crossover'].getfloat('indpb'),
				"indgen": config['crossover'].getfloat('indgen')
			}

		return cls(
			n_genes=n_genes,
			n_pairs=n_pairs,
			size=size,
			input_dir=input_dir,
			color_file=color_file,
			iterations=iterations,
			descriptors=descriptors,
			bootstrap_individuals=bootstrap_individuals,
			mutation_op=mutation_fun,
			mutation_args=mutation_args,
			crossover_flag=crossover_flag,
			crossover_op=crossover_op,
			crossover_fun=crossover_fun,
			crossover_args=crossover_args,
			minimization=minimization,
			plot_args=plot_args,
			log_dir=log_dir,
			config_path=config_path,
			seed=seed,
			n_bins=n_bins,
			bins=bins
		)


	def generate_initial_population(self):

		self.logger.info("Generating initial population from the collection")
		population_ids = list(np.arange(0, (len(self.collection.points))))
		seed_ids = random.sample(population_ids, self.bootstrap_individuals)
		collection = Collection(self.input_dir, self.color_file)
		for id_ in seed_ids:
			ind = Individual(collection, id_, collection.cmap, self.size)
			self.place_in_mapelites(ind)

	def curiosity_crossover(self, individuals=2, iteration=None, mut_gen=None):
		"""
		Generate an offspring out of two individuals in the collection with high curiosity score
		"""

		#curiosity approach
		x_indxs, y_indxs = self.curiosity_score_selection(individuals=2)
		ind1_loc = self.current_locations[self.fpath_ids[x_indxs[0], y_indxs[0]]]
		ind2_loc = self.current_locations[self.fpath_ids[x_indxs[1], y_indxs[1]]]

		#random sampling apporach
		#idxs = self.random_selection(individuals=2)
		#ind1_loc = self.current_locations[self.fpath_ids[idxs[0]]]
		#ind2_loc = self.current_locations[self.fpath_ids[idxs[1]]]


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

		offspring = eaop.uniform_crossover_geometry(ind1, ind2, self.crossover_args["indgen"]+mut_gen, self.crossover_args["indpb"], parent_ids=[ind1.parent_ids, ind2.parent_ids])

		return offspring

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

	def curiosity_score_selection(self, individuals=2, k=5):
		"""
		Select an elite x from the current map of elites.
		The selection is based on  based on how many
		offspring it has added to the collection.
		:param individuals: The number of individuals to randomly select
		:return: A list of N random elites
		"""
		if(k<=individuals):
			raise Exception (f"Sampled population needs to be larger than selection")
		elif(k>= self.bootstrap_individuals):
			raise Exception (f"Sampled population needs to be smaller than the number of bootstrapped individuals during initialization")
		else:
			curiosity_flat = self.curiosity_score.flatten()
			curious_indxs = curiosity_flat.argsort()[-k:]

			# convert the idx_1d back into indices arrays for each dimension
			x_idx, y_idx = np.unravel_index(curious_indxs, self.curiosity_score.shape)
			# select individuals from the sample
			rnd_ind = random.sample((0, len(x_idx)-1), individuals)
			x_selected, y_selected = x_idx[rnd_ind], y_idx[rnd_ind]

		return x_selected, y_selected


	def similarity_based_crossover(self):
		"""
		Bootstrap the algorithm by generating `self.bootstrap_individuals` individuals from a collection of individuals.
		"""
		self.logger.info("Generating initial population from the collection")
		similarity_matrix = util.genetic_similarity(self.collection.points, self.n_genes)
		diverse_inds = util.diverse_pairs(similarity_matrix, self.n_pairs)
		population_ids = list(np.arange(0, (len(collection.points))))
		seed_ids = random.sample(population_ids, self.bootstrap_individuals)

		for id in seed_ids:
			pair_ids = diverse_inds[id]
			ind1 = Individual(collection, id, self.cmap)
			pairs = [Individual(collection, id_, self.cmap) for id_ in pair_ids]
			offsprings = eaop.uniform_crossover_geometry_similarity(ind1, pairs, self.crossover_args["indgen"], self.crossover_args["indpb"])
			for offspring in offsprings:
				self.place_in_mapelites(offspring)


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
		perf = self.performance_measure(x, fpath='./inferences_server/inferences/individual.png', exp_folder='F:\PhD_Research\Output\CaseStudies\MAP-Elites\pv_urban\inference_server\minimal-ml-serverCPU\inferences')

		# place operator performs either minimization or maximization
		if self.place_operator(perf[0][0], self.performances[x_bin, y_bin]):
			self.logger.debug(f"PLACE: Placing individual {x} at {x_bin, y_bin} with perf: {perf[0][0]}")

			# if this is during initalization simply update the grid
			if(init):
				# update performance score for the added individual
				self.performances[x_bin, y_bin] = perf[0][0]
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
				self.performances[x_bin, y_bin] = perf[0][0]
				fpath = self.log_dir_path / 'genomes' / 'generation_{}'.format(iteration) / 'individual_{}_{}.pkl'.format(x_bin, y_bin)
				x.save_to_disk(fpath)
				self.current_locations.append(str(fpath.absolute()))
				self.fpath_ids[x_bin, y_bin] = len(self.current_locations)-1
				# increase curiosity score for the parents
				for parents in x.parent_ids:
					self.curiosity_score[parents[0], parents[1]] += 1.0
		else:
			if(init):
				self.logger.debug(f"PLACE: Individual {x} rejected at {x_bin, y_bin} with perf: {perf[0][0]} in favor of {self.performances[x_bin, y_bin]}")
			else:
				self.logger.debug(f"PLACE: Offspring {x} rejected at {x_bin, y_bin} with perf: {perf[0][0]} in favor of {self.performances[x_bin, y_bin]}")
				for parents in x.parent_ids:
					self.curiosity_score[parents[0], parents[1]] -= 0.5

		if pbar is not None:
			sleep(0.5)
			pbar.update(1)

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
		# decide who are the behavioral dimensions (make this less hard coded)
		x_bin = np.digitize(osr, self.bins['bin_osr'], right=True)
		y_bin = np.digitize(mh, self.bins['bin_mh'], right=True)

		#x_bin = np.digitize(fsi, self.bins['bin_fsi'], right=True)
		#y_bin = np.digitize(gsi, self.bins['bin_gsi'], right=True)

		return x_bin, y_bin

	def performance_measure(self, x, fpath, exp_folder):
		"""
		Function to evaluate solution x and give a performance measure
		:param x: genotype of a solution
		:return: performance measure of that solution
		"""
		#NOTE: FIX THE TWO DIFFERENT RUN_INF SCRIPTS, one for init and one for a pair.

		image = x.draw_image()
		image.save(r'F:\PhD_Research\Output\CaseStudies\MAP-Elites\pv_urban\inference_server\minimal-ml-serverCPU\inferences\individual.png')
		run_inf()
		performance = evaluate_fn(exp_folder)

		return performance

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
		x_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[2])]]
		y_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[3])]]

#		x_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[0])]]
#		y_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[1])]]

		plot_heatmap(self.performances,
					 x_ax,
					 y_ax,
					 savefig_path=self.log_dir_path,
					 iteration=iteration,
					 title=f"Urban Density QD Map of Elites",
					 **self.plot_args)

	def get_elapsed_time(self):
		return self.elapsed_time

	def run(self):
		"""
		Main iteration loop of MAP-Elites
		"""
		start_time = time.time()
		# start by creating an initial set of random solutions
		self.generate_initial_population()
		self.plot_map_of_elites(iteration='initial')

		outter_bar = tqdm(total=self.iterations, desc="Iterations completed")
		outter_loop = range(0, self.iterations)
		inner_bar = tqdm(total=500, desc="Offsprings produced", leave=False)
		inner_loop = range(0, 500)

		# tqdm: progress bar
		for i in outter_loop:
	#with tqdm(total=self.iterations, desc="Iterations completed") as pbar:
		#for i in range(0, self.iterations):
			# create the save folder for the generation
			folder = fpath = self.log_dir_path / 'genomes' / 'generation_{}'.format(i)
			folder.mkdir(parents=True, exist_ok=True)
			for j in inner_loop:
			#for j in range(0, 100):
			# create more diverse mutations
				if (j%2 == 0):
					self.logger.debug(f"ITERATION {i} - Individual {j}")
					self.logger.debug("Select and mutate.")
					# get the number of elements that have already been initialized
					offspring = self.curiosity_crossover(individuals=2, iteration=i, mut_gen=0.25)
					offspring_mut = eaop.polynomial_bounded(offspring, self.cmap, eta=1.0, low=5.0, up=100.0, mut_pb=1/len(offspring.heights))
					# place the new individual in the map of elites
					self.place_in_mapelites(offspring_mut, pbar=inner_bar, init=False, iteration=i)
				else:
					self.logger.debug(f"ITERATION {i} - Individual {j}")
					self.logger.debug("Select and mutate.")
					# get the number of elements that have already been initialized
					offspring = self.curiosity_crossover(individuals=2, iteration=i, mut_gen=-0.25)
					offspring_mut = eaop.polynomial_bounded(offspring, self.cmap, eta=1.0, low=5.0, up=100.0, mut_pb=1/len(offspring.heights))
					# place the new individual in the map of elites
					self.place_in_mapelites(offspring_mut, pbar=inner_bar, init=False, iteration=i)

			inner_bar.reset()
			self.plot_map_of_elites(iteration=i)
			outter_bar.update(1)

		# save results, display metrics and plot statistics
		end_time = time.time()
		self.elapsed_time = end_time - start_time
		self.save_logs()
		self.plot_map_of_elites(iteration=self.iterations)
