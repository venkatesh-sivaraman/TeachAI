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
        template_image = np.flip(template_image, axis=0)
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

def detect_region(points, image_size, mask_size):
    """Another method for detecting regions."""
    # Build a density map of input points
    bins = {}
    weighted_bins = {} # weighted by depth coordinate
    bin_size = (image_size[0] // mask_size[1], image_size[1] // mask_size[0])

    max_count = 0
    max_weight = 0
    for i, point in enumerate(points):
        x_bin = np.clip(int(np.round(point[0] / bin_size[0])), 0, mask_size[1] - 1)
        y_bin = np.clip(int(np.round(point[1] / bin_size[1])) + 1, 0, mask_size[0] - 1)
        bins.setdefault((x_bin, y_bin), []).append(i)
        weighted_bins[(x_bin, y_bin)] = weighted_bins.get((x_bin, y_bin), 0.0) + point[2]
        max_weight = max(max_weight, weighted_bins[(x_bin, y_bin)])
        max_count = max(max_count, len(bins[(x_bin, y_bin)]))
    # mat = np.zeros(mask_size)
    # for (x, y), idxs in bins.items():
    #     mat[y, x] = len(idxs)
    # print(mat)

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.scatter([x for x, y in points], [y for x, y in points])
    # plt.xlim(0, image_size[0])
    # plt.ylim(0, image_size[1])
    # plt.subplot(1, 2, 2)
    # plt.imshow(mat)
    # plt.show()

    # Get the bins that have several points
    def include_bin(bin_id):
        return (len(bins[bin_id]) > min(max_count * 0.5, len(points) * 0.1) or
                weighted_bins[bin_id] >= max_weight * 0.6)
    in_bins = [bin_id for bin_id in bins if include_bin(bin_id)]
    in_idxs = [i for bin_id in in_bins for i in bins[bin_id]]

    mask = np.zeros(mask_size)
    for i in range(min(in_idxs), max(in_idxs) + 1):
        x_bin = np.clip(int(np.round(points[i][0] / bin_size[0])), 0, mask_size[1] - 1)
        y_bin = np.clip(int(np.round(points[i][1] / bin_size[1])), 0, mask_size[0] - 1)
        mask[y_bin, x_bin] = 1

    return mask

def detect_region_probabilistic(points, image_size):
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
