# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import random
import json

# https://github.com/CityOfLosAngeles/count-machine/blob/5ed4176b8664b3343c0fcdc265733c1c5c6ca871/code/pedestrian_hog.py
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


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to images directory")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
for imagePath in paths.list_images(args["images"]):
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = cv2.imread(imagePath)

    print (image.shape)

    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()

    print (orig.shape)

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:

        print (x,y,w,h)
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

        c = 255 * np.random.randint(low=0, high=2, size=(h,w), dtype="i")
        image[y:(y+h),x:(x+w),0] = c
        image[y:(y+h),x:(x+w),1] = c
        image[y:(y+h),x:(x+w),2] = c

        # s0 = slice(y, y+h)
        # s1 = slice(x, x+w)

        # for i in range(3): 
        #     image[s0,s1,i] = c

        # sub_img = image[y:y + h, x:x + w]
        # sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        # shape = sub_img.shape
        # for row in range(shape[0]):
        #     for col in range(shape[1]):
        #         rnd = random.randint(0, 1)
        #         sub_img[row, col] = rnd * 255
                # noise_img = sp_noise(sub_img, 0.08)
    
    cv2.imwrite('sp_noise.jpg', image)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))

    # show the output images
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)

