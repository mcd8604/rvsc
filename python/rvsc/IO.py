import os
import numpy as np
import cv2

def loadImageNames(imageNamesFile):
	if not os.path.isfile(imageNamesFile):
		raise IOError('Cannot find imageNamesFile: {}'.format(imageNamesFile))
	imageNames = []
	with open(imageNamesFile) as f:
		imageNames = [i.strip() for i in f.readlines()]
	return imageNames

class LabelDataLoader:
	'''
	"Abstract" class for loading dict of classification label maps which have image 
	names for keys and blob data for values. Blob shape = (N, C, H, W).
	'''
	def __init__(self):
		pass
		
	def load(self):
		raise NotImplementedError, 'Must subclass this method!'

class LabelDataLoaderNumpy(LabelDataLoader):
	'''
	Load class label maps from Numpy file
	'''
	def __init__(self, filePath):
		self.filePath = filePath
		
	def load(self):
		return np.load(filePath)

class LabelDataLoaderImages(LabelDataLoader):	
	'''
	Load class label maps from images
	'''
	def __init__(self, imageDir, imageNames):
		self.imageDir = imageDir
		self.imageNames = imageNames
	
	def load(self):
		if not os.path.exists(self.imageDir):
			raise IOError("Label images directory does not exist: {}".format(self.imageDir))
		labels = {}		
		for imageName in self.imageNames:		
			labelImagePath = os.path.join(self.imageDir, imageName)
			assert os.path.exists(labelImagePath), "Label image does not exist: {}".format(labelImagePath)
			labels[imageName] = self.loadImage(labelImagePath)			
		return labels

	def loadImage(self, filePath):
		'''
		Loads a ground truth label image. Must be greyscale 8-bit and should contain 
		intensity values matching expected class labels. 0 is considered background label.
		Splits the image into an array of boolean maps for each class label.
		'''
		labelImage = cv2.imread(filePath, 0).astype(np.uint8)
		labelData = [labelImage==(i+1) for i in range(self.numClasses)]
		labelData = np.stack(labelData, 0)
		#labelData = labelData.reshape((self.numClasses,-1)) #this should be done only in metrics generators
		return labelData

def loadCombineAVScores(arteryScoreFile, veinScoreFile, outFile=None):
	'''
	This loads two separate score files (one for vein, one for artery) and combines them into a single score dict. 
	Score files should have same key set and value shapes should match.
	'''
	assert os.path.exists(arteryScoreFile), "arteryScoreFile does not exist: {}".format(arteryScoreFile)
	assert os.path.exists(veinScoreFile), "veinScoreFile does not exist: {}".format(veinScoreFile)
	
	a = np.load(arteryScoreFile)
	v = np.load(veinScoreFile)
	assert len(a.files) == len(v.files),\
		"Score files do not have the number of images: len(arteryScoreFile)={}, len(veinScoreFile)={}".format(len(a),len(v))
		
	outDict = {}
	for k in a.keys():
		scoreA = a[k]
		scoreV = v[k]
		assert k in v.files, "Key '{}' not found in veinScoreFile".format(k)
		assert scoreA.shape == scoreV.shape, "Scores shape mismatch: artery={}, vein={}".format(scoreA.shape, scoreV.shape)
		outDict[k] = np.concatenate([scoreA, scoreV])
	
	if outFile:
		np.savez(outFile, **outDict)
	return outDict