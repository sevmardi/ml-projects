import cv2


class FaceDetector:

    def __init__(self, faceCascadePath):
        """
        This classifier is serialized as an XML file. Making a call to cv2.CascadeClassifier will deserialize the classifier, load it into memory, and allow him to detect faces in images.
        params
        -------
        faceCascadePath - the path to where his cascade
                classifier lives
        """
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

    def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        """
        Find faces function
        params
        -------
		Scalefactor: How much the image size is reduced at each image scale. 
		minNeighbors: How many neighbors each window should have for the area in the window to be considered a face.        
		minSize: A tuple of width and height (in pixels) indicating the minimum size of the window.

		Returns 
		---------
		tuples containing the bounding boxes of the faces in the image.
		"""
		#Detecting the actual faces in the image
        rects = self.faceCascade.detectMultiScale(
            image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, flags=cv2.CASCADE_SCALE_IMAGE)

        return rects
