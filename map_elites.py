import time
from time import sleep
import random
from random import sample
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
from ea_operators import uniform_crossover_geometry, polynomial_bounded
from population import Location, IndividualGrid, OffspringGrid, NeighbourGrid
from utilities import create_cmap, rotate_input
from evaluation import run_inf, evaluate_fn
from plot_utils import plot_heatmap

class MapElites(ABC):
	def __init__(self,
					size,
					color_file,
					wind_dir,
					wind_dir_names,
					iterations,
					descriptors,
					mutation_op,
					mutation_args,
					crossover_op,
					crossover_args,
					n_bins,
					bins,
					plot_args,
					input_dir,
					inference_dir,
					log_dir,
					config_path,
					minimization=True
					):
		"""
		:param iterations: Number of evolutionary iterations
		:param wind_dir: The different wind directions (in angles from South wind) included in the study.
		:param wind_dir_names: The names of the wind directions included in the study.
		:param n_pairs: The number of distant individuals that we want to find.
		:param mutation_args: Mutation function arguments
		:param crossover_op: Crossover function
		:param crossover_args: Crossover function arguments
		:param bins: Bins for feature dimensions
		:param minimization: True if solving a minimization problem. False if solving a maximization problem.
		"""

		# Location specific
		self.size = size
		self.color_file = color_file
		self.wind_dir = wind_dir
		self.wind_dir_names = wind_dir_names
		self.input_dir = input_dir
		self.inference_dir = inference_dir
		self.log_dir = log_dir
		self.location = Location(self.input_dir, self.color_file)
		self.cmap = create_cmap(color_file)

		# QD Specific
		self.mutation_op = mutation_op
		self.mutation_args = mutation_args
		self.crossover_op = crossover_op
		self.crossover_args = crossover_args
		self.n_bins = n_bins
		self.bins = bins
		if not (len(self.bins) == len(descriptors)):
			raise Exception(
				f"Length of bins needs to match the number of descriptor dimensions ")
		self.plot_args = plot_args
		self.iterations = iterations
		self.descriptors = descriptors

		self.minimization = minimization
		if self.minimization:
			self.place_operator = operator.lt
		else:
			self.place_operator = operator.ge
		self.elapsed_time = 0

		# Get the number of bins for each feature dimension
		ft_bins = [len(bin)-1 for bin in self.bins.values()]
		ft_bins = [ft_bins[int(self.n_bins[0])], ft_bins[int(self.n_bins[1])]]

		# Map of Elites: Initialize data structures to store solutions and fitness values
		if(self.minimization):
			self.performances = np.full(ft_bins, np.inf, dtype=(float))
		else:
			self.performances = np.full(ft_bins, -np.inf, dtype=(float))
		self.curiosity_score = np.full(ft_bins, -np.inf, dtype=(float))
		self.fpath_ids = np.full(ft_bins, 0, dtype=(int))
		self.previous_locations = []
		self.current_locations = []

		# Create log directory
		if log_dir:
			self.log_dir_path = Path(self.log_dir)
		else:
			now = datetime.now().strftime("%Y%m%d%H%M%S")
			self.log_dir_name = f"log_{now}"
			self.log_dir_path = Path(f'logs/{self.log_dir_name}')
		self.log_dir_path.mkdir(parents=True, exist_ok=True)

		# Create initial population directory
		genomes = self.log_dir_path / 'genomes' / 'initial_population'
		genomes.mkdir(parents=True, exist_ok=True)

		# Create performance directory
		performances = self.log_dir_path / 'performances'
		performances.mkdir(parents=True, exist_ok=True)

		# Save config file
		copyfile(config_path, self.log_dir_path / 'config.ini')

		# Setup logging
		self.logger = logging.getLogger('map elites')
		self.logger.setLevel(logging.DEBUG)

		# Create file handler which logs even debug messages
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
		input_dir = config['location'].get('input_dir')
		inference_dir = config['location'].get('inference_dir')
		color_file = config['location'].get('color_file')
		size_x = config['location'].getint('size_x')
		size_y = config['location'].getint('size_y')
		size = (size_x, size_y)
		wind_dir = config['location'].get('wind_dir').split(',')
		wind_dir_names = config['location'].get('wind_dir_names').split(',')

		#main mapelites config
		iterations = config['mapelites'].getint('iterations')
		minimization = config['mapelites'].getboolean('minimization')

		#plotting config
		plot_args = dict()
		plot_args['highlight_best'] = config['plotting'].getboolean('highlight_best')
		plot_args['interactive'] = config['plotting'].getboolean('interactive')

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
				"indgen": config['crossover'].getfloat('indgen')
			}

		return cls(
			wind_dir=wind_dir,
			wind_dir_names=wind_dir_names,
			size=size,
			input_dir=input_dir,
			inference_dir=inference_dir,
			color_file=color_file,
			iterations=iterations,
			descriptors=descriptors,
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

	def generate_initial_location(self):

		self.logger.info("Generating initial population from the location")
		self.logger.info("{} occupied grids in the location".format(self.location.num_occupied_grids))
		self.place_in_mapelites_with_inference(self.location, loc_mutations=None, init=True)

	def select_location_from_map(self):
		"""
		Select a location from the map at random
		"""

		#random sampling apporach
		loc = sample(self.current_locations, 1)[0]
		print(loc)
		with open(loc, 'rb') as loc_pickle:
			content = pickle.load(loc_pickle)

			polygons = content['polygons']
			heights = content['heights']
			colors = content['colors']
			grid_ids = content['grid_ids']
			neighbours = content['neighbours']
			status = content['status']
			grid_occupancy = content['grid_occupancy']
			grids_dangerous = content['grids_dangerous']
			grids_sitting = content['grids_sitting']
			dangerous = content['dangerous']
			sitting = content['sitting']
			evolved_buildings = content['evolved_buildings']
			evolved_grids = content['evolved_grids']

		location = Location(self.input_dir, self.color_file, size=(2500,2500), init=False,
							polygons=polygons, heights=heights, colors=colors, status=status, grid_ids=grid_ids,
							grid_occupancy=grid_occupancy, grids_dangerous=grids_dangerous,
							grids_sitting=grids_sitting, dangerous=dangerous, sitting=sitting,
							evolved_buildings=evolved_buildings, evolved_grids=evolved_grids,
							neighbours=neighbours)
		return location

	def place_in_mapelites_with_inference(self, x, loc_mutations, pbar=None, init=True, iteration=None):
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
		if(init):
			#get coordinates in the feature space from individual's performance
			grids_sitting, grids_dangerous = [], []
			for id_ in x.grid_occupancy:
				ind = IndividualGrid(x, id_, size=(250, 250))
				_, _, sitting_area, dangerous_area = self.performance_evaluation(ind, folder=self.inference_dir)
				grids_sitting.append(sitting_area)
				grids_dangerous.append(dangerous_area)

			x.grids_sitting = grids_sitting
			x.grids_dangerous = grids_dangerous
			x.sitting = np.sum(grids_sitting)
			x.dangerous = np.sum(grids_dangerous)
			x.calc_fitness()
		else:
			for id_ in x.evolved_grids[-loc_mutations:]:
				ind = IndividualGrid(x, id_, size=(250, 250))
				_, _, sitting_area, dangerous_area = self.performance_evaluation(ind, folder=self.inference_dir)
				x.grids_sitting[np.where(x.grid_occupancy == id_)[0].item()] = sitting_area
				x.grids_dangerous[np.where(x.grid_occupancy == id_)[0].item()] = dangerous_area

			x.sitting = np.sum(x.grids_sitting)
			x.dangerous = np.sum(x.grids_dangerous)
			x.calc_fitness()

		#get coordinates in the feature space
		x_bin, y_bin = self.map_x_to_b_inference(x.sitting_fitness, x.dangerous_fitness)
		# save the parent id of the individual
		x.grid_position = [x_bin, y_bin]
		# save individual performance


		# place operator performs either minimization or maximization
		if self.place_operator(x.features['FSI'], self.performances[x_bin, y_bin]):
			self.logger.debug(f"PLACE: Placing individual {x} at {x_bin, y_bin} with perf: {x.features['FSI']}")

			# if this is during initalization simply update the grid
			if(init):
				# update performance score for the added individual
				self.performances[x_bin, y_bin] = x.features['FSI']
				#save individual to disk for later use
				fpath = self.log_dir_path / 'genomes' / 'initialLocation_{}_{}.pkl'.format(x_bin, y_bin)
				#x.parent_ids = x.grid_position
				x.save_to_disk(fpath)
				# save individual's location on the disk
				self.current_locations.append(str(fpath.absolute()))
				# save individual's location id
				self.fpath_ids[x_bin, y_bin] = len(self.current_locations)-1
			else:
				self.performances[x_bin, y_bin] = x.features['FSI']
				fpath = self.log_dir_path / 'genomes' / 'Location_{}_{}_{}'.format(iteration, x_bin, y_bin)
				x.save_to_disk(fpath)
				self.current_locations.append(str(fpath.absolute()))
				self.fpath_ids[x_bin, y_bin] = len(self.current_locations)-1
		else:
			self.logger.debug(f"PLACE: Individual {x} rejected at {x_bin, y_bin} with perf: {x.features['FSI']} in favor of {self.performances[x_bin, y_bin]}")

		if pbar is not None:
			sleep(0.5)
			pbar.update()

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

	def performance_evaluation(self, x, folder):
		"""
		Function to evaluate solution x and give a performance measure
		:param x: genotype of a solution
		:return: performance measure of that solution
		"""

		# first generate all wind directions for the selected individual
		image = x.draw_image()
		for dir_ in self.wind_dir:
			img = rotate_input(image, int(dir_))
			img.save(folder + '\\individual_{}.png'.format(dir_))
		run_inf()
		sitting, dangerous, sitting_area, dangerous_area = evaluate_fn(self.inference_dir)

		return sitting, dangerous, sitting_area, dangerous_area

	def save_logs(self, iteration):
		"""
		Save logs, config file, individuals, and data structures to log folder.

		:param iteration: The current iteration of the algorithm.
		"""

		self.logger.info(f"Running time {time.strftime('%H:%M:%S', time.gmtime(self.elapsed_time))}")

		np.save(self.log_dir_path / 'performances' / 'performances_{}'.format(iteration), self.performances)
		#np.save(self.log_dir_path / "solutions", self.solutions)
		np.save(self.log_dir_path / 'curiosity', self.curiosity_score)

	def plot_map_of_elites(self, iteration):
		"""
		Plot a heatmap of elites
		"""
		# Stringify the bins to be used as strings in the plot axes
		x_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[5])]]
		y_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[6])]]

		#x_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[0])]]
		#y_ax = [str(d) for d in self.bins['bin_{}'.format(self.descriptors[1])]]

		plot_heatmap(self.performances,
						x_ax,
						y_ax,
						v_min=0.0,
						v_max=5.0,
						savefig_path=self.log_dir_path,
						iteration=iteration,
						title=f"MAP-Elites for the city of Boston",
						**self.plot_args)

	def get_elapsed_time(self):
		return self.elapsed_time

	def run(self):
		"""
		Main iteration loop of MAP-Elites
		"""
		start_time = time.time()
		# start by creating an initial set of random solutions
		self.generate_initial_location()
		#self.plot_map_of_elites(iteration='initial')

		outter_bar = tqdm(total=self.iterations, desc="Iterations completed", position = 0, leave = True)
		outter_loop = range(0, self.iterations)
		inner_loop = range(1, 101)
		total_mutations = 0

		for i in outter_loop:
			self.location = self.select_location_from_map()
			# create the save folder for the generation
			folder = self.log_dir_path / 'genomes'
			folder.mkdir(parents=True, exist_ok=True)
			for j in inner_loop:
				self.logger.debug(f"ITERATION {i} - Individual {j}")
				self.logger.debug("Select and mutate.")
				# create more diverse mutations
				try:
					offspring = uniform_crossover_geometry(self.location, self.crossover_args["indgen"])
					# place the individual in the map, if it is an elite
					mut_offspring = polynomial_bounded(offspring, self.cmap, eta=20.0, low=5.0, up=100.0, mut_pb=1/len(offspring.heights))
					self.location.evolve(mut_offspring)
					total_mutations+=1
					print(self.location.evolved_grids)
				except:
					pass
				if(j % 10 == 0):
					self.place_in_mapelites_with_inference(self.location, loc_mutations=total_mutations,
														  pbar=outter_bar.set_postfix(inner_loop=j, refresh=True),
														  init=False, iteration=i*100+j)
					total_mutations=0
			self.save_logs(iteration=i)
			#inner_bar.reset()
			#self.plot_map_of_elites(iteration=i)
			outter_bar.update()

		# save results, display metrics and plot statistics
		end_time = time.time()
		self.elapsed_time = end_time - start_time
		self.save_logs(iteration=i+1)
		#self.plot_map_of_elites(iteration=self.iterations)
