import numpy as np
from math import ceil, floor
from random import randrange, uniform, sample
from PIL import Image
import aggdraw

def create_cmap(color_file):
    """
    Creates an RGB color map out of a list of color values

    Args:
        color_file (string): A csv file of the color gradient used to generate the heightmaps.

    Returns:
        array: An array of RGB values representing the colors for each height in the collection's range of heights.
    """
    colors = np.loadtxt(color_file, dtype=str)
    cmap = []
    for color in colors:
        cmap.append(color.split(',')[0])

    cmap = np.array(cmap, dtype=int)

    return cmap

def height_to_color(cmap, height):
    """
    Translates a building height value to a color, based on the given color map.

    Args:
        cmap (array): An array of RGB values representing the colors for each height in the collection's range of heights.
        height (float): A building height.

    Returns:
        tuple: The (R, G, B) value that corresponds to the specific input height.
    """
    if(height > len(cmap)-1):
        color_value = 0
    else:
        modulo = height % 1
        if(modulo) == 0:
            color_value = cmap[int(height)]
        else:
            minimum = floor(height)
            maximum = ceil(height)

            min_color = cmap[minimum+1]
            max_color = cmap[maximum+1]

            color_value = min_color + ((min_color-max_color) * modulo)

    return [color_value, color_value, color_value]

def get_poly_ids(polygons, status, indgen):
    """
    A function that selects random polygons from an input list of polygons.

    Args:
        polygons (list): A list of polygons
        random_genes (float, optional): Whether to select a random number of polygons, bypassing indgen.
        indgen (float): The minimum percentage of input polygons to select.

    Returns:
        list: A list of polygon ids.
    """
    mask = status.astype(bool)
    valid_ids = list(np.arange(0, len(polygons))[mask])
    poly_ids = sample(valid_ids, int(len(valid_ids) * indgen))

    return poly_ids

def draw_polygons(polygons, grid_ids, colors, heights, im_size=(2500, 2500), b_color="white", fpath=None, grid_id=None):

    if(grid_id):
        image = Image.new("RGB", (250, 250), color=b_color)
        draw = aggdraw.Draw(image)

        # get displacement from origin
        x_displacement = grid_id // 10
        y_displacement = grid_id % 10

        ids = np.where(grid_ids == grid_id)
        polygons = polygons[ids]
        colors = colors[ids]
        heights = heights[ids]

        for poly, color , height in zip(polygons, colors, heights):
            # get x, y sequence of coordinates for each polygon
            xy = poly.exterior.xy
            coords = np.dstack((xy[1], xy[0])).flatten()

            #bring everything to 250, 250
            # x-coordinates
            coords[1::2] = coords[1::2] - x_displacement*(im_size[0]//10)
            # y-coordinates
            coords[0::2] = coords[0::2] - y_displacement*(im_size[1]//10)
            # create a brush according to each polygon color
            if(height == 0.0):
                brush = aggdraw.Brush((255, 255, 255), opacity=255)
            else:
                brush = aggdraw.Brush((color[0], color[1], color[2]), opacity=255)
            draw.polygon(coords, brush)

        image = Image.frombytes("RGB", (250, 250), draw.tobytes()).rotate(90)

    else:
        image = Image.new("RGB", im_size, color="white")
        draw = aggdraw.Draw(image)

        for poly, color, height in zip(polygons, colors, heights):
            # get x, y sequence of coordinates for each polygon
            xy = poly.exterior.xy
            coords = np.dstack((xy[1], xy[0])).flatten()
            # create a brush according to each polygon color
            if(height == 0.0):
                brush = aggdraw.Brush((255, 255, 255), opacity=255)
            else:
                brush = aggdraw.Brush((color[0], color[1], color[2]), opacity=255)
            draw.polygon(coords, brush)

        image = Image.frombytes("RGB", im_size, draw.tobytes()).rotate(90)  

    if(fpath):
        image.save(fpath)

    return draw, image

def rotate_input(pil_img, degrees, interval=512):
    """
    Method to rotate image by `degrees` in a COUNTER-CLOCKWISE direction.
    As some rotations cause the corners of the original image to be cropped,
    the `interval` argument allows the image to expand in size.

    Args:
        pil_img (PIL.Image): A PIL image of the individual.
        degrees (int): The degrees of rotation.
        interval (int, optional): The interval to use while rotating the image. Defaults to 512.

    Returns:
        PIL image: A rotated image of the input individual.
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