from skimage import feature
import numpy as np
import mahotas
import cv2
from . import imputils


class HOG:
    """Hand writing image recgnoniztion"""

    def __init__(self, orientations=9, pixelsPerCell=(8, 9), cellsPerBlock=(3, 3), transform=False):

        super(HOG, self).__init__()
        self.orientations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.transform = transform

    def describe(self, image):
        hist = feature.hog(image, orientations=self.orienations, pixels_per_cell=self.pixelsPerCell,
                           cells_per_block=self.cellsPerBlock, transform_sqrt=self.transform)
        return hist


def load_digits(data_set_path):
    data = np.genfromtxt(data_set_path, delimiter=",", dtype="uint8")
    target = data[:, 0]
    data = data[:, 1:].reshape(data.shape[0], 28, 28)

    return (data, target)


def deskw(image, width):
    """A function to fix some of the "lean" of hand written digits"""
    # grabs the height and width of the image, then
    (h, w) = image.shape[:2]
    # the moments of the image are computed
    momnets = cv2.moments(image)
    # Skew is computed based on the momnets and the wraping matrix
    skew = moments["mu11"] / momnets["mu02"]
    M = np.float32([1, skew, -0.5 * w * skew], [0, 1, 0])
    # skwing the image takes place.
    # The first argument is the image that is going to be skewed, the
    # second is the matrix M that defines the “direction” in which
    # the image is going to be deskewed, and the third parameter
    # is the resulting width and height of the deskewed image.
    # the flags parameter controls how the image is going to be deskewed. In this case, we use linear inter-
    # polation.
    image = cv2.wrapAffine(
        image, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    image = imutils.resize(image, width=width)

    return image
