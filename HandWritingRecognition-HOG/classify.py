from __future__ import print_function
from sklearn.externals import joblib
from pyimagesearch.hog import HOG
from pyimagesearch import dataset
import argparse
import mahotas
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
                help="path to where the model will be stored")
ap.add_argument("-i", "--image", required=True, help="Path to the image file")
args = vars(ap.parse_args())
model = joblib.load(args["model"])
hog = Hog(orientations=18, pixelsPerCell=(10, 10),
          cellsPerBlock=(1, 1), transform=True)
# load the query image off disk and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Canny edge detector is applied to find edges
edged = cv2.Canny(blurred, 30, 150)
(_, cnts, _) = cv2.findContours(edged.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted([c, cv2.boudingRect(c)[0] for c in cnts], key=lambda x: x[1])

for (c, _) in cnts:
    # A bounding box for each contour is computed cv2.boundingRect function
    (x, y, w, h) = cv2.boundingRect(c)
    # Check the width and height of the bounding box
    if w >= 7 and h >= 20:
        # Region of Interest (ROI) is extracted from the grayscale image using
        # numpy.
        roi = gray[y:y + h, x:x + w]
        thresh = roi.copy()
        T = mahotas.thresholding.otsu(roi)
        thresh[thresh > T] = 255
        thresh = cv2.bitwise_not(thresh)
        # the digit is then deskewed and translated to the center of the image
        thresh = dataset.deskew(thresh, 20)
        thresh = dataset.center_extent(thresh, (20, 20))

        cv2.imshow("thresh", thresh)

        # compute the HOG feature vector of the threshholded ROI
        hist = hog.describe(thresh)
        # HOG feature vector is fed into the LinearSVC’s predict method which classifies which digit the ROI is, based on
        # the HOG feature vector
        digit = model.predict([hist])[0]
        print("I think the number is {}".format(digit))
        # Display the digit on the orginal image, in order to do this we call cv2.rectangle function to draw a green rectangle around
        # the current digit
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # We use the putText functio to draw the digit on the image (orginal)
        cv2.putText(image, str(digit), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        # Display the image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
