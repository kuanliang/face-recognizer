import cv2

class FaceDetector:
    def __init__(self, faceCascadePath):
        # the path to the Cascade face classifier, which is serilized as an XML file
        # making a call to cv2.CascadeClassifier will deserialize the classifier,
        # load it into memory, and allow for detecting faces in images
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)
    
    # detect method 
    def detect(self, image, scaleFactor = 1.1, minNeighbors = 5, minSize=(30, 30)):
        '''face detect method 

        Args:
            image: the image being processed

            scaleFactor: How much the image size is reduced at each image scale

            minNeighbors: How many neighbors each window should have for the
                         for the area in the window to be considered a face.

            minSize:
        '''
        rects = self.faceCascade.detectMultiScale(image,
            scaleFactor = scaleFactor,
            minNeighbors = minNeighbors, minSize = minSize,
            flags = cv2.CASCADE_SCALE_IMAGE)

        return rects