from __future__ import print_function
from facedetector import FaceDetector 
import argparse
import cv2

# command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
                help="Path to where the face cascade resides")
ap.add_argument("-i", "--image", required=True,
                help="path to where the image file resides")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY)


# instantiates FaceDetector class
fd = FaceDetector(args["face"])

# detect the actual faces in the image by making a call to the detect method.
faceRects = fd.detect(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
# prints out the number of faces found in the image.
print("I found {} face(s)".fomat(len(faceRects)))

for (x, y, w, h) in faceRects:
	#draws a green box around the actual faces
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Faces", image)
cv2.waitKey(0)
