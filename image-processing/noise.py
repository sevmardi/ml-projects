import numpy as np
import random
import cv2
import argparse
import sys,  getopt
import imutils
from imutils import paths

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.randint(0, 1)
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to images directory")
args = vars(ap.parse_args())

for imagePath in paths.list_images(args["images"]):
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    # gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
    noise_img = sp_noise(image, 0.08)
cv2.imwrite('sp_noise.jpg', noise_img)
    
    # noise_img = sp_noise(image, 0.08)
    # cv2.imwrite('sp_noise.jpg', noise_img)


