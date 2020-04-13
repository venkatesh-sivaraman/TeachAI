import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from PIL import Image
import os

class Region:
    def __init__(self, values):
        self.values = values

    def save_image(self, rgb):
        template_image = np.ones((*self.values.shape, 4))
        template_image[:,:,:3] *= np.array(rgb)
        template_image[:,:,3] = self.values / np.max(self.values)
        img = Image.fromarray(np.uint8(template_image * 255))
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        id_num = np.random.randint(0, 1e10)
        path = os.path.join("tmp", "map_{}.png".format(id_num))
        img.save(path)
        return path

def resample_points(points):
    """Resamples the given set of points to be roughly evenly distributed in space."""
    # The function to be interpolated has distance traveled along the x axis, and either the
    # x or the y coordinates as the dependent variable.
    distances = []
    curr = 0.0
    for i, point in enumerate(points):
        if i != 0:
            curr += np.linalg.norm(np.array([point[0] - points[i - 1][0],
                                             point[1] - points[i - 1][1]]))
        distances.append(curr)

    f = interp1d(distances, points.T, kind='linear')
    return f(np.linspace(0, distances[-1], 20)).T

def detect_region(points, image_size):
    """
    Identifies a region of the image being highlighted by the given set of points, where the
    region is described by the probability of each pixel in the image being within the region
    being highlighted.

    points: A list of (x, y) tuples representing a loose definition of a region.
    image_size: A tuple (width, height) describing the dimensions of the image.

    Returns:
        A Region object containing an image of size (height, width) where the values at each
        position are the probabilities that each position is within the region.
    """

    # Loosely, we want P(point in region) to increase if the point is close to many of the
    # input points. The radius of "closeness" should be determined by the extent of the
    # region.
    point_array = resample_points(np.array(points))
    width, height = image_size

    prob_mat = np.zeros((height, width))
    var = ((np.max(point_array[:,1]) - np.min(point_array[:,1])) *
           (np.max(point_array[:,0]) - np.min(point_array[:,0]))) / 50.0

    for point in point_array:
        row_offsets = np.maximum(0, np.abs(np.arange(height) - point[1]) - 40)
        col_offsets = np.maximum(0, np.abs(np.arange(width) - point[0]) - 40)
        prob_mat += (np.exp(-0.5 * (row_offsets[:,np.newaxis] ** 2 / var)) *
                     np.exp(-0.5 * (col_offsets[np.newaxis,:] ** 2 / var)))

    # plt.figure()
    # orig_points = np.array(points)
    # plt.plot(orig_points[:,0], orig_points[:,1], 'bo')
    # plt.plot(point_array[:,0], point_array[:,1], 'ro')
    # plt.imshow(prob_mat)
    # plt.colorbar()
    # plt.show()

    # TODO How to know if the points are inside the convex hull defined by the gesture points?
    return Region(prob_mat)
