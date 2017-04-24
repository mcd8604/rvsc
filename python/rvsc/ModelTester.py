import cv2
import numpy as np
import caffe
import os
import IO

class ModelTester():
	def __init__(self):
		self.scoreBlobName = None
		self.imageNamesFile = None
		self.testImagesDir = None
		self.testScoresFile = None
		self.model = None
		self.weights = None
		self.gpuMode = None
		self.outputScoreFile = None
		
	# NOTE for refactoring: model 7/8 override this to load patches
	def getInputDict(self):
		inputDict = {}
		
		if(self.testImagesDir):
			# NOTE for refactoring: used for models 1,2,4,5,6
			# Read and preprocess image data
			imageNames = IO.loadImageNames(self.imageNamesFile)
			for imageName in imageNames:
				imagePath = os.path.join(self.testImagesDir, imageName)
				assert os.path.isfile(imagePath), 'Cannot find image: {}'.format(imagePath)
				image = cv2.imread(imagePath)
				image = np.array(image, dtype=np.float32)
				image = image[:,:,::-1] #BGR
				# TODO: externalize mean - specific to model and training dataset
				image -= np.array((171.0773,98.4333,58.8811)) #Mean substraction
				image = image.transpose((2,0,1))
				inputDict[imageName] = image[np.newaxis, ...]
		
		elif(self.testScoresFile):
			# NOTE for refactoring: used for model 3
			# Load scores from file
			inputDict = {}
			scoresDict = np.load(self.testScoresFile)
			# temp fix - model_1 scores are shape (C,H,W), so add the N axis to dim 1
			for i in scoresDict.files:
				inputDict[i] = scoresDict[i][np.newaxis,...]
			
		return inputDict
			
	def test(self):		
		if(self.gpuMode):
			caffe.set_device(0)
			caffe.set_mode_gpu()

		# load net
		self.net = caffe.Net(self.model, caffe.TEST, weights=self.weights)
		
		# load input data
		inputDict = self.getInputDict()
		
		# score
		scoresDict = {}
		for imageName, inputData in inputDict.iteritems():
			print("Testing image: " + imageName)
			
			#TEMP
			#import matplotlib.pyplot
			#matplotlib.pyplot.imshow(inputData[0][2])
			#matplotlib.pyplot.show()
			
			self.net.blobs['data'].reshape(*inputData.shape)
			self.net.blobs['data'].data[:] = inputData[:]
			self.net.forward()
			scoresDict[imageName] = self.net.blobs[self.scoreBlobName].data.copy()
			
			#TEMP
			#matplotlib.pyplot.imshow(self.net.blobs[self.scoreBlobName].data[0][0])
			#matplotlib.pyplot.show()
	
		# save
		np.savez(self.outputScoreFile, **scoresDict)
			
	
def parseArgs():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-b", "--scoreBlobName", help="Name of model blob to pull score from.", default="sigmoid-fuse")
	parser.add_argument("-f", "--imageNamesFile", help="File containing names of test images.")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("-i", "--testImagesDir", help="Directory containing images to test.")
	group.add_argument("-s", "--testScoresFile", help="File containing separate model output test scores to use as test input for current model.", default="model_1_testScores.npz")
	parser.add_argument("-m", "--model", help="Model testing proto definition file.", default="train.prototxt")
	parser.add_argument("-w", "--weights", help="Model weights file (.caffemodel)", default="_iter_20000.caffemodel")
	parser.add_argument("-g", "--gpuMode", help="GPU", action="store_true")
	parser.add_argument("-o", "--outputScoreFile", help="File path to save output scores to.", default="testScores.npz")

	return parser.parse_args()	
	
if __name__ == "__main__":
	args = parseArgs()
	tester = ModelTester()
	tester.scoreBlobName = args.scoreBlobName
	tester.imageNamesFile = args.imageNamesFile
	tester.testImagesDir = args.testImagesDir
	tester.testScoresFile = args.testScoresFile
	tester.model = args.model
	tester.weights = args.weights
	tester.gpuMode = args.gpuMode
	tester.outputScoreFile = args.outputScoreFile
	tester.test()