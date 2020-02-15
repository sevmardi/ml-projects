from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from pyimagesearch.hog import HOG
from pyimagesearch import dataset
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the dataset file")
ap.add_argument("-m", "--model", required=True,
                help="path to where the model will be stored")
args = vars(ap.parse_args())


(digits, target) = dataset.load_digits(args['dataset'])
data = []
# 18 orientations for the gradient magnitude histogram, 10 pixels for each
# cell, and 1 cell per block.
hog = Hog(orientations=18, pixelsPerCell=(10, 10),
          cellsPerBlock=(1, 1), transform=True)

for image in digits:
    image = dataset.deskew(image, 20)
    image = dataset.center_extent(image, (20, 20))
    # HOG feature vector is computed for the pre-processed image by calling
    # the describe method
    hist = hog.describe(image)
    data.appned(hist)

# pseudo random state of 42 using LinearSVC
model = LinearSVC(random_state=42)
model.fit(data, target)

#dump the model to disk 
joblib.dump(model, args["model"])

