import sys
import os.path
import csv
import numpy as np
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, classification_report, jaccard_similarity_score

class DataTransformer:
	'''
	Necessary for transforming scores and truth data to (C, totalNumPixels)
	The ResultsGenerator requires scores and truths as data dictionaries which have image 
	names for keys and various data for values. Each model and tests will require specific
	logic to transform the blob data to an ndarray with channels as first dimension and all
	higher dimensions to be flattened. Transformations must make sure to retain index order
	between scores and truths.
	# TODO add checks (where is best?) to enforce score and label array shapes 
	'''	
	def __init__(self):
		pass		
	def transformScoresPerClass(self, numChannels, scoreData):
		return scoreData
		#raise NotImplementedError, 'Must subclass this method!'			
	def transformTruthsPerClass(self, numChannels, truthData):
		return truthData
		#raise NotImplementedError, 'Must subclass this method!'

class ResultsGenerator:
	'''
	ResultsGenerator handles generation of classification model test results including:
		plotting precision recall curves
		plotting confusion matrix
		TODO: rendering output score images (currently in separate script)
	'''

	def __init__(self, labelDataLoader, dataTransformer=DataTransformer(), **kwargs):
		self.labelDataLoader = labelDataLoader
		self.dataTransformer = dataTransformer
		self.parseArgs(**kwargs)
		self.truthDict = labelDataLoader.load()
		self.transformData()
		
	def parseArgs(self, **kwargs):
		# TODO validate inputs / handle KeyErrors?
		#self.truthDir = kwargs['truthDir']
		self.scoreDict = kwargs['scoreDict']
		self.numClasses = kwargs['numClasses']
		self.classNames = kwargs['classNames']
		self.classColors = kwargs['classColors']
		#self.imageHeight = kwargs['imageHeight'] 
		#self.imageWidth = kwargs['imageWidth']
		#self.plotImageFile = kwargs['plotImageFile']
		#self.plotDataFile = kwargs['plotDataFile']
	
	def transformData(self):	
		'''
		Reshape data so class is first dimension and higher dimensions are flattened		
		Note:  Actual transformations are delegated to data transformer
		'''
		self.scores = np.zeros((self.numClasses, 0), dtype=np.float32)
		self.truths = np.zeros((self.numClasses, 0), dtype=np.uint8)
		for i in self.scoreDict.keys():
			# allow model-specific data transformation			
			scoresByClass = self.dataTransformer.transformScoresPerClass(self.numClasses, self.scoreDict[i])
			truthsByClass = self.dataTransformer.transformTruthsPerClass(self.numClasses, self.truthDict[i])
			# line em up!
			self.scores = np.concatenate((self.scores, scoresByClass), axis=1)	
			self.truths = np.concatenate((self.truths, truthsByClass), axis=1)
			
	def plotPrecisionRecall(self, title="Precision-Recall Curve", plotImageFile=None, figureSize=(8,8)):		
		plt.figure(figsize=figureSize)
		for c in range(self.numClasses):
			className = self.classNames[c]
			classColor = self.classColors[c]
			print 'Calculating Precision-Recall for class: {}'.format(className)
			
			truth = self.truths[c]#.flatten()
			score = self.scores[c]#.flatten()
				
			# Calculate
			precision, recall, thresholds = precision_recall_curve(truth, score)
			avgPR = average_precision_score(truth, score)
			
			# Plot
			plt.plot(precision, recall, color=classColor, label='[{:.3f}] {}'.format(avgPR, className))
			
		# Configure Plot
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title(title)
		plt.ylim([0.0, 1.0])
		plt.xlim([0.0, 1.0])
		plt.legend(loc="lower left")
		plt.tight_layout()
		if(plotImageFile):
			plt.savefig(plotImageFile)
		else:
			plt.show()
	
	def getConfusionmatrix(self):
		# shift labels, truths, and predictions up one index to 
		# prevent merging of first label (label value 0 moved to value 1) with background pixels (0 value)
		labels = range(1, self.numClasses+1)
		truths = np.insert(self.truths, obj=0, values=np.zeros_like(self.truths[0]), axis=0)
		scores = np.insert(self.scores, obj=0, values=np.zeros_like(self.scores[0]), axis=0)
		
		# get argmax predictions and calculate confusion matrix
		truth = np.argmax(truths, axis=0)
		prediction = np.argmax(scores, axis=0)
		return confusion_matrix(truth, prediction, labels)		
		
	def getCrossEntropy(self, classIndex=None):
		n = float(self.truths.sum())
		if classIndex:
			prediction = self.scores[classIndex]
			truth = self.truths[classIndex]
			n /= self.numClasses
		else:
			prediction = self.scores
			truth = self.truths
		np.seterr(divide='ignore')
		logPrediction = np.log2(prediction)
		np.seterr(divide='warn') 
		# consider log(0) as equivalent to 0 for practical purposes
		logPrediction[np.isneginf(logPrediction)] = 0 
		return -np.sum(truth * logPrediction) / n
		
	def printClassificationStats(self, outFile=None):			
		# column headers
		metricNames = ['Cross Entropy', 'Precision', 'Recall', 'F-Score']
		headers = ['{} {}'.format(m,c) for m,c in itertools.product(self.classNames, metricNames)] + ['Accuracy']
		
		# calculate metrics
		cm = self.getConfusionmatrix()
		metrics = []
		for i in range(self.numClasses):
			crossEntropy = self.getCrossEntropy(i)
			truePositive = cm[i,i]
			predictionPositive = float(np.sum(cm[i]))
			conditionPositive = float(np.sum(np.transpose(cm)[i]))
			precision = truePositive / predictionPositive
			recall = truePositive / conditionPositive
			fScore = 2 * precision * recall / (precision + recall)
			metrics += [crossEntropy, precision, recall, fScore]
		accuracy = np.trace(cm) / float(cm.sum())
		metrics.append(accuracy)
		
		# print
		out = sys.stdout
		if outFile:
			out = open(outFile, 'w')
		writer = csv.writer(out)
		writer.writerow(headers)
		writer.writerow(metrics)
		if outFile:
			out.close()

	def plotConfusionMatrix(self, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, plotImageFile=None, figureSize=(6,6)):
		cm = self.getConfusionmatrix()
		
		# below is from scikit learn website	
		
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.figure(figsize=figureSize)
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(self.classNames))
		plt.xticks(tick_marks, self.classNames, rotation=45)
		plt.yticks(tick_marks, self.classNames)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		print(cm)

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j],
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")

		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.tight_layout()
		if(plotImageFile):
			plt.savefig(plotImageFile)
		else:
			plt.show()