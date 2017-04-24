import numpy as np
from ResultsGenerator import ResultsGenerator, DataTransformer
import DRIVE_common as DRIVE

class DataTransformer1(DataTransformer):
	def  __init__(self, combineOverlap=False, ignoreBackground=False):
		self.combineOverlap = combineOverlap
		self.ignoreBackground = ignoreBackground
		
	def transformScoresPerClass(self, numClasses, scoreData):
		return self.transform(numClasses, scoreData)
		
	def transformTruthsPerClass(self, numClasses, truthData):
		return self.transform(numClasses, truthData)
		
	# input: truthData and scoreData shape: (C, H, W)
	# output: move classes/channels to first dimension and flatten all higher dimensions
	def transform(self, numClasses, data):
		classes = []
		if not self.ignoreBackground:
			classes.append(data[0])
		if self.combineOverlap:
			overlap = data[3]
			artery = np.maximum(data[1], overlap)
			vein = np.maximum(data[2], overlap)
			classes += [artery, vein]
		else:
			classes += [data[1], data[2], data[3]]
		data = np.stack(classes, axis=0)
		return data.reshape((numClasses, -1))

dataTransformer = DataTransformer1()

def getResultsGenerator(modelName='iter_20000', dataSet='test'):
	scoreFile =  '{}_{}Scores.npz'.format(modelName, dataSet)
	truthDir =  '../../../Images/DRIVE/{}/av'.format(dataSet)
	scoreDict = np.load(scoreFile)
	labelLoader = DRIVE.LabelDataLoaderImagesDRIVE(truthDir, scoreDict.files)
	s = 1 if dataTransformer.ignoreBackground else 0
	e = 3 if dataTransformer.combineOverlap else 4
	return ResultsGenerator(labelLoader, dataTransformer, scoreDict=scoreDict, 
		numClasses=e-s, classNames=DRIVE.classes[s:e], classColors=DRIVE.classColors[s:e])	
		
def run(modelName, dataSet):
	mode = 'NoBG' if dataTransformer.ignoreBackground else 'BG'
	mm = getResultsGenerator(modelName, dataSet)
	mm.plotPrecisionRecall(plotImageFile='{}PrecRecall_{}.pdf'.format(dataSet, mode), title="4 Class Model")
	mm.plotConfusionMatrix(plotImageFile='{}ConfusionMatrix_{}.pdf'.format(dataSet, mode), title="4 Class Model")
	mm.printClassificationStats(outFile='{}Stats_{}.csv'.format(dataSet, mode))
		
if __name__ == "__main__":	
	modelName = 'iter_20000'
	run(modelName, 'train')
	run(modelName, 'test')
	dataTransformer.ignoreBackground = True
	run(modelName, 'train')
	run(modelName, 'test')