width = 565
height = 584
numImages = 20
numClasses = 4
classes = ['Background', 'Artery', 'Vein', 'Overlap']	
classColors = ['black', 'red', 'blue', 'green']
classCMaps = ['Greys_r', 'Reds_r', 'Blues_r', 'Greens_r']
classRGBs = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)]
unknownRGB = (255, 255, 255)

from IO import LabelDataLoaderImages
import cv2
import numpy as np
def truthImageToLabelArray(combineOverlap, truthImage, stack=True):
	'''
	For loading RITE (Hu, Abramoff, Garvin) or Lincoln (Qureshi/Al'Diri) classification ground truth
	'''
	if combineOverlap:
		artery 	= np.all(truthImage==classRGBs[1], axis=-1)
		vein 	= np.all(truthImage==classRGBs[2], axis=-1)
		overlap = np.all(truthImage==classRGBs[3], axis=-1)
		unknown = np.all(truthImage==unknownRGB, axis=-1)
		artery = (artery + overlap + unknown) > 0
		vein = (vein + overlap + unknown) > 0
		labels = [artery, vein] 
	else:
		#labels = np.all(truthImage==classRGBs[:], axis=-1)
		#labels = [np.all(np.maximum((truthImage==i),(truthImage==unknownRGB if i>0 else 0)), axis=-1) for i in classRGBs]
		labels = [np.all((truthImage==i), axis=-1) for i in classRGBs]
	return np.stack(labels, 0) if stack else labels
	
class LabelDataLoaderImagesDRIVE(LabelDataLoaderImages):
	#def __init__(self, filePath, imageNamesFile, vesselOnlyMode):
	#	super(LabelDataLoaderImagesDRIVE, self).__init__(filePath, imageNamesFile)
	combineOverlap = False
		
	def loadImage(self, filePath):
		labelImage = cv2.imread(filePath, 1).astype(np.uint8)[:,:,::-1]
		return truthImageToLabelArray(self.combineOverlap, labelImage)
