import numpy as np
import pickle
import aggdraw
from PIL import Image
from utilities import height_to_color, create_cmap, draw_polygons
from geometry import create_shapely_polygons, get_features, centroids, move_polygons, move_neighbour_polygons


class Location:
    def __init__(self, case_folder, color_file, size=(2500, 2500), init=True,
                 polygons=None, heights=None, colors=None, status=None, grid_ids=None,
                 grid_occupancy=None, grids_dangerous=None, grids_sitting=None, dangerous=None, sitting=None,
                 evolved_buildings=None, evolved_grids=None, neighbours=None):
        """
        A class to create an individual representing a whole geographical location, 
        including the information of all individual buildings.

        Args:
            folder (string): The folder where the genetic information of the location is stored.
            color_file (string): The location where the color file for the collection was stored.

        Raises:
            ValueError: I can't find anything in the selected folder.
            ValueError: A location requires the same number of points, heights, splits, and grid id values.
        """
        if(init):
            self.folder = case_folder
            self.color_file = color_file
            self.cmap = create_cmap(self.color_file)
            self.size = size
            self.evolved_buildings = []
            self.evolved_grids = []
            # get the location's initial genetic information
            self.points = np.load(self.folder  + r"\points\harvard_init.npy").flatten()
            self.heights = np.load(self.folder + r"\heights\harvard_init.npy").flatten()
            self.splits = np.load(self.folder +  r"\splits\harvard_init.npy").flatten()
            self.grid_ids = np.load(self.folder +  r"\grid_ids\harvard_init.npy").flatten()
            self.neighbour_ids = np.load(self.folder +  r"\neighbour_ids\harvard_init.npy").flatten()
            self.neighbour_id_splits = np.load(self.folder +  r"\neighbour_id_splits\harvard_init.npy").flatten()
            self.status = np.load(self.folder +  r"\status\harvard_init.npy").flatten()
            self.grid_occupancy = np.load(self.folder +  r"\grid_occupancy\harvard_init.npy").flatten()

            # Genotype
            self.colors  = np.array([height_to_color(self.cmap, height) for height in np.clip(self.heights, 0, 100)]).astype(int)
            self.polygons = create_shapely_polygons(self.points, self.splits, self.grid_ids)
            self.neighbours = np.split(self.neighbour_ids, np.cumsum(self.neighbour_id_splits))

            #couple of sanity checks to make sure a proper individual was passed
            if (len(self.points)) == len(self.heights) == len(self.splits) == len(self.grid_ids) == len(self.status) == 0:
                    raise ValueError(f"I can't find anything in the selected folder")

            if(len(self.points) != len(self.heights) != len(self.splits) != len(self.status) != len(self.grid_ids)):
                raise ValueError(f"A location requires the same number of points, heights, splits, \
                                 and grid id values.")
        else:
            self.polygons = polygons
            self.heights = heights
            self.grid_ids = grid_ids
            self.neighbours = neighbours
            self.status = status
            self.grid_occupancy = grid_occupancy
            self.colors  = colors
            self.color_file = color_file
            self.cmap = create_cmap(self.color_file)
            self.size = size
            self.evolved_buildings = evolved_buildings
            self.evolved_grids = evolved_grids

        # Features and descriptors
        self.footprints = np.array([polygon.area for polygon in self.polygons])
        self.feature_names = ['FSI', 'GSI', 'OSR', 'Mean_height', 'Tare']
        self.features = dict(zip(self.feature_names, get_features(self.footprints, self.heights)))
        self.centroids = np.array(centroids(self.polygons))
        self.num_buildings = len(self.heights)
        self.num_occupied_grids = len(self.grid_occupancy)
        self.grids_dangerous = grids_dangerous
        self.grids_sitting = grids_sitting
        self.dangerous = dangerous
        self.sitting = sitting

    def calc_fitness(self):

        try:
            self.dangerous_fitness = self.dangerous / (self.features['FSI'] * (self.size[0] * self.size[1]))
        except:
            pass
        try:
            self.sitting_fitness = self.sitting / (self.features['FSI'] * (self.size[0] * self.size[1]))
        except:
            pass

    def evolve(self, offspring):

        # Add/Replace the information to the location's genome

        # temporary values
        temp_heights = self.heights[offspring.added]
        temp_status = self.status[offspring.added]
        temp_colors = self.colors[offspring.added]

        if(len(offspring.dropped)):
            self.grid_ids = np.delete(self.grid_ids, offspring.dropped)
            self.heights = np.delete(self.heights, offspring.dropped)
            self.status = np.delete(self.status, offspring.dropped)
            self.colors = np.delete(self.colors, offspring.dropped, axis=0)
            self.polygons = np.delete(self.polygons, offspring.dropped)

        if(len(offspring.added)):
            self.grid_ids = np.append(self.grid_ids, np.array([offspring.grid_ids[0] for building in offspring.added]))
            self.heights = np.append(self.heights, temp_heights)
            self.status = np.append(self.status, temp_status)
            self.colors = np.append(self.colors, temp_colors, axis=0)

            indices = []
            for id_ in offspring.added:
                indx = np.where(offspring.building_ids == id_)[0].item()
                indices.append(indx)
            polygons = move_polygons(offspring.polygons[indices], from_grid=0, to_grid=offspring.grid_ids[0])
            self.polygons = np.append(self.polygons, polygons)

        self.evolved_grids.append(offspring.grid_ids[0])
        self.footprints = np.array([polygon.area for polygon in self.polygons])
        self.centroids = np.array(centroids(self.polygons))
        self.features = dict(zip(self.feature_names, get_features(self.footprints, self.heights)))
        self.num_buildings = len(self.heights)

    def draw_image(self, grid_id=None, fpath=None):
        """
        Draws a png image of the individual.

        Returns:
            image: A png image of the individual.
        """
        _, image = draw_polygons(self.polygons, self.grid_ids, self.colors, self.heights, self.size,
                                 grid_id=grid_id, fpath=fpath)

        return image
    def save_to_disk(self, fname):
        """
        Saves an individual and its properties to the hard drive.

        Args:
            fname (string): The filename the individual will be saved with.
        """
        data = {'polygons': self.polygons, 'heights': self.heights,
                'colors': self.colors, 'footprints:': self.footprints,
                'features': self.features, 'status': self.status,
                'grid_occupancy': self.grid_occupancy, 'grids_dangerous': self.grids_dangerous,
                'grids_sitting':self.grids_sitting, 'dangerous': self.dangerous,
                'sitting':self.sitting, 'evolved_buildings': self.evolved_buildings,
                'evolved_grids': self.evolved_grids, 'grid_ids': self.grid_ids,
                'neighbours': self.neighbours}

        with open(fname, 'wb') as file:
            pickle.dump(data, file)

class IndividualGrid:
    def __init__(self, location, grid_id, size=(250, 250)):
        """
        A class that creates the collection of grid-individuals out of a provided location.

        Args:
            collection (Collection): A collection of individuals.
            id_ (int): The indice of the individual in the collection.
            cmap (list): The color gradient map that represents heights into colors.
            size ([type]): The extend of the bounding box of an individual, in meters. Used to generate appropriate
                           image outputs.
        """
        self.grid_id = grid_id
        self.size = size
        self.grid_position = None

        # get individual's genome properties from collection
        self.building_ids = np.where(location.grid_ids == self.grid_id)
        self.polygons = move_polygons(location.polygons[self.building_ids], self.grid_id)
        self.heights = location.heights[self.building_ids]
        self.colors = location.colors[self.building_ids]
        self.footprints = location.footprints[self.building_ids]
        self.status = location.status[self.building_ids]
        self.centroids = location.centroids[self.building_ids]

        # calcualte features and descriptors
        self.feature_names = ['FSI', 'GSI', 'OSR', 'Mean_height', 'Tare']
        self.features = dict(zip(self.feature_names, get_features(self.footprints, self.heights)))
        self.dangerous = None
        self.sitting = None

    def draw_image(self, fpath=None):
        """
        Draws a png image of the individual.

        Returns:
            image: A png image of the individual.
        """
        image = Image.new("RGB", self.size, color="white")
        draw = aggdraw.Draw(image)

        for poly, color, height in zip(self.polygons, self.colors, self.heights):
            # get x, y sequence of coordinates for each polygon
            xy = poly.exterior.xy
            coords = np.dstack((xy[1], xy[0])).flatten()
            # create a brush according to each polygon color
            if(height == 0.0):
                brush = aggdraw.Brush((255, 255, 255), opacity=255)
            else:
                brush = aggdraw.Brush((color[0], color[1], color[2]), opacity=255)
            draw.polygon(coords, brush)

        image = Image.frombytes("RGB", self.size, draw.tobytes()).rotate(90)

        if(fpath):
            image.save(fpath)

        return image

    def save_to_disk(self, fname):
        """
        Saves an individual and its properties to the hard drive.

        Args:
            fname (string): The filename the individual will be saved with.
        """
        data = {'polygons': self.polygons, 'heights': self.heights,
                'colors': self.colors, 'footprints:': self.footprints,
                'features': self.features, 'building_ids': self.building_ids,
                'grid_id': self.grid_id, 'status': self.status,
                'dangerous': self.dangerous, 'sitting': self.sitting}

        with open(fname, 'wb') as file:
            pickle.dump(data, file)

class OffspringGrid:
    def __init__(self, polygons, colors, heights, grid_ids, status, building_ids, added, dropped, size=(250, 250)):
        """
        A class to create an offspring, along with its properties, out of the crossover or mutation
        of individuals.

        Args:
            polygons (list): The polygons of the evolved individual.
            colors (list): The colors of the evolved individual.
            heights (list): The heights of the evolved individual.
            size (tuple): The extend of the bounding box of an individual, in meters. Used to generate appropriate
                           image outputs.
        """
        # assign genome
        self.polygons = polygons
        self.colors  = colors
        self.heights = heights
        self.grid_ids = grid_ids
        self.status = status
        self.building_ids = building_ids
        self.added = added
        self.dropped = dropped
        self.size = size

        #calculate phenotype
        self.footprints = np.array([polygon.area for polygon in self.polygons])
        self.feature_names = ['FSI', 'GSI', 'OSR', 'Mean_height', 'Tare']
        self.features = dict(zip(self.feature_names, get_features(self.footprints, self.heights, boundary=self.size)))
        self.centroids = np.array(centroids(self.polygons))
        self.dangerous = None
        self.sitting = None
        self.fi_fitness = None

    def draw_image(self, fpath=None):
        """
        Draws a png image of the individual.

        Returns:
            image: A png image of the individual.
        """
        image = Image.new("RGB", (250, 250), color="white")
        draw = aggdraw.Draw(image)

        for poly, color, height in zip(self.polygons, self.colors, self.heights):
            # get x, y sequence of coordinates for each polygon
            xy = poly.exterior.xy
            coords = np.dstack((xy[1], xy[0])).flatten()
            # create a brush according to each polygon color
            if(height == 0.0):
                brush = aggdraw.Brush((255, 255, 255), opacity=255)
            else:
                brush = aggdraw.Brush((color[0], color[1], color[2]), opacity=255)
            draw.polygon(coords, brush)

        image = Image.frombytes("RGB", (250, 250), draw.tobytes()).rotate(90)

        if(fpath):
            image.save(fpath)

        return image

    def save_to_disk(self, fname):
        """
        Saves an individual and its properties to the hard drive.

        Args:
            fname (string): The filename the individual will be saved with.
        """
        data = {'polygons': self.polygons, 'heights': self.heights,
                'colors': self.colors, 'footprints:': self.footprints,
                'features': self.features, 'grid_id': self.grid_id,
                'status': self.status, 'building_ids': self.building_ids,
                'dangerous': self.dangerous, 'sitting': self.sitting, 'fi_fitness': self.fi_fitness}

        with open(fname, 'wb') as file:
            pickle.dump(data, file)

class NeighbourGrid:
    def __init__(self, location, grid_id, size=(750, 750)):
        """
        A class that creates the collection of grid-individuals out of a provided location.

        Args:
            collection (Collection): A collection of individuals.
            id_ (int): The indice of the individual in the collection.
            cmap (list): The color gradient map that represents heights into colors.
            size ([type]): The extend of the bounding box of an individual, in meters. Used to generate appropriate
                           image outputs.
        """
        self.grid_id = grid_id
        self.size = size
        self.grid_position = None

        # get genome properties from collection
        building_ids = []
        for id_ in location.neighbours[grid_id]:
            building_ids.append(np.where(location.grid_ids == id_))

        self.building_ids = np.concatenate(building_ids, axis=1).ravel()
        self.polygons = move_neighbour_polygons(location.polygons[self.building_ids], self.grid_id)
        self.heights = location.heights[self.building_ids]
        self.colors = location.colors[self.building_ids]
        self.footprints = location.footprints[self.building_ids]
        self.status = location.status[self.building_ids]
        self.centroids = location.centroids[self.building_ids]

        # calcualte features and descriptors
        self.feature_names = ['FSI', 'GSI', 'OSR', 'Mean_height', 'Tare']
        self.features = dict(zip(self.feature_names, get_features(self.footprints, self.heights)))
        self.dangerous = None
        self.sitting = None

    def draw_image(self, fpath=None):
        """
        Draws a png image of the individual.

        Returns:
            image: A png image of the individual.
        """
        image = Image.new("RGB", self.size, color="white")
        draw = aggdraw.Draw(image)

        for poly, color, height in zip(self.polygons, self.colors, self.heights):
            # get x, y sequence of coordinates for each polygon
            xy = poly.exterior.xy
            coords = np.dstack((xy[1], xy[0])).flatten()
            # create a brush according to each polygon color
            if(height == 0.0):
                brush = aggdraw.Brush((255, 255, 255), opacity=255)
            else:
                brush = aggdraw.Brush((color[0], color[1], color[2]), opacity=255)
            draw.polygon(coords, brush)

        image = Image.frombytes("RGB", self.size, draw.tobytes()).rotate(90)

        if(fpath):
            image.save(fpath)

        return image

    def save_to_disk(self, fname):
        """
        Saves an individual and its properties to the hard drive.

        Args:
            fname (string): The filename the individual will be saved with.
        """
        data = {'polygons': self.polygons, 'heights': self.heights,
                'colors': self.colors, 'footprints:': self.footprints,
                'features': self.features, 'building_ids': self.building_ids,
                'grid_id': self.grid_id, 'status': self.status,
                'dangerous': self.dangerous, 'sitting': self.sitting}

        with open(fname, 'wb') as file:
            pickle.dump(data, file)