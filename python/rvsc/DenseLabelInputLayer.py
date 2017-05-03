import caffe

import os
import numpy as np
#import cv2
from PIL import Image

class DenseLabelInputLayer(caffe.Layer):
	def setup(self, bottom, top):
		# two tops: data and label
		if len(top) != 2:
			raise Exception("Need to define two tops: data and label.")
		# data layers have no bottoms
		if len(bottom) != 0:
			raise Exception("Do not define a bottom.")

		self.loadParams()
		self.loadImageNames()
		self.idx = 0
			
	def loadParams(self):
		self.params = eval(self.param_str)
		self.imageNamesFile = self.params['imageNamesFile']
		self.inputImageDir = self.params['inputImageDir']
		self.labelImageDir = self.params['labelImageDir']
		assert os.path.isfile(self.imageNamesFile),	"imageNamesFile doesn't exist: {}".format(self.imageNamesFile)
		assert os.path.isdir(self.inputImageDir), "inputImageDir doesn't exist: {}".format(self.inputImageDir)
		assert os.path.isdir(self.labelImageDir), "labelImageDir doesn't exist: {}".format(self.labelImageDir)
		self.mean = np.array(self.params['mean'])
		
	def loadImageNames(self):
		# load image file names
		self.imageNames = []
		with open(self.imageNamesFile) as f:
			self.imageNames = [i.strip() for i in f.readlines()]
		assert len(self.imageNames) > 0, "imageNamesFile {} is empty".format(self.imageNamesFile)
		self.checkImages()
	
	def checkImages(self):
		# check that all image files exists
		for imageName in self.imageNames:
			inputFilePath = os.path.join(self.inputImageDir, imageName)
			assert os.path.isfile(inputFilePath), "Input image file doesn't exist: {}".format(inputFilePath)
			labelFilePath = os.path.join(self.labelImageDir, imageName)
			assert os.path.isfile(labelFilePath), "Label image file doesn't exist: {}".format(labelFilePath)		

	def reshape(self, bottom, top):
		# load image + label image pair
		self.loadInput(self.idx)
		self.loadLabel(self.idx)
		# reshape tops to fit (leading 1 is for batch dimension)
		top[0].reshape(1, *self.data.shape)
		top[1].reshape(1, *self.label.shape)

	def forward(self, bottom, top):
		# assign output
		top[0].data[...] = self.data
		top[1].data[...] = self.label

		# pick next input
		self.idx += 1
		if self.idx >= len(self.imageNames):
			self.idx = 0

	def backward(self, top, propagate_down, bottom):
		pass

	def openImageFile(self, filePath):		
		# Loading via OpenCV
		#image = cv2.imread(inputFilePath)
		#image = np.array(image, dtype=np.float32)
		#return image[:,:,::-1] # Only swap BGR to RGB if loading with OpenCV
		
		# Loading via PIL
		image = Image.open(filePath)
		imageData = np.array(image.getdata(), np.float32)
		numChannels = len(image.getbands())
		return imageData.reshape(image.size[1], image.size[0], numChannels)		
		
	def loadInput(self, idx):
		inputFilePath = os.path.join(self.inputImageDir, self.imageNames[idx])		
		self.data = self.openImageFile(inputFilePath) 
		
		# Preprocess
		self.data -= self.mean 
		self.data = self.data.transpose((2,0,1)) 
				
	def loadLabel(self, idx):
		labelFilePath = os.path.join(self.labelImageDir, self.imageNames[idx])	
		labelImage = self.openImageFile(labelFilePath) 		
		self.prepareLabelFromImage(labelImage)
	
	def prepareLabelFromImage(self, image):
		"""
		Override this method for handling ground truth differently in a specific model
		Default behavior: multilabel classification
			label image should be greyscale, but is loaded as color..
			assume all channels are equivalent and take the first
			0 values are background and each unique int value is a different class label 
		"""
		numLabels = int(np.max(image))
		image = image.transpose((2,0,1)) # switch to (Channel, Height, Width)
		image = image[0] # just take first channel - assumes R,G,B all equal
		# create a channel for each label, including background
		self.label = np.stack([image==i for i in range(numLabels+1)], 0)		