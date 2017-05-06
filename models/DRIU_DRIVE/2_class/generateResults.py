import numpy as np
from ResultsGenerator import ResultsGenerator, DataTransformer
import DRIVE_common as DRIVE
import IO

class DataTransformer2Class(DataTransformer):
	# input: truthData and scoreData shape: (C, H, W)
	# output: move classes/channels to first dimension and flatten all higher dimensions
	def transformScoresPerClass(self, numClasses, scoreData):
		return scoreData.reshape((numClasses, -1))
		
	def transformTruthsPerClass(self, numClasses, truthData):
		# NOTE: moved this to DRIVE label loader: labelLoader.combineOverlap = True
		#overlap = truthData[3]
		#artery = np.maximum(truthData[1], overlap)
		#vein = np.maximum(truthData[2], overlap)
		#return np.stack([artery, vein], axis=0).reshape((numClasses, -1))
		return truthData.reshape((numClasses, -1))

# data-set is train or test split
# truth-set is RITE/lincoln/etc..		
def generate(scoreFile, dataSet, modelName, modelTitle, fileSuffix='', truthSet=''):
	truthDir = '../../../Images/DRIVE/{}/{}'.format(dataSet, truthSet)
	scoreDict = np.load(scoreFile)
	labelLoader = DRIVE.LabelDataLoaderImagesDRIVE(truthDir, scoreDict.files)
	labelLoader.combineOverlap = True
	s = 1
	e = 3
	n = e-s
	mm = ResultsGenerator(labelLoader, DataTransformer2Class(), scoreDict=scoreDict, 
		numClasses=n, classNames=DRIVE.classes[s:e], classColors=DRIVE.classColors[s:e])		
	mm.plotPrecisionRecall(plotImageFile='{}PrecRecall{}.png'.format(dataSet, fileSuffix), title=modelTitle)	
	mm.plotConfusionMatrix(plotImageFile='{}ConfMatrix{}.png'.format(dataSet, fileSuffix), title=modelTitle)
	mm.printClassificationStats(outFile='{}Stats{}.csv'.format(dataSet, fileSuffix))
	
	from renderTestOutput import TestRenderer
	outputDir = '{}Renderings{}'.format(dataSet, fileSuffix)
	renderer = TestRenderer(truthDir, scoreFile, outputDir)
	renderer.render()

if __name__ == "__main__":
	# NOTE: hardcoded script params
	def runModel(modelName, modelTitle, fileSuffix, truthSet):
		def runDataSet(dataSet):
			scoreFile = '{}_{}Scores.npz'.format(modelName, dataSet)	
			generate(scoreFile, dataSet, modelName, modelTitle, fileSuffix, truthSet)
		runDataSet('train')
		runDataSet('test')
	runModel('iter_20000', "2 Class Model", '_normal', 'av')
	runModel('aug_iter_20000', "2 Class Model - Rotate/Mirror", "_aug", 'av')
	runModel('control_iter_20000', "2 Class Model - Lincoln", "_control", 'lincoln')