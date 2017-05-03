import numpy as np
from ResultsGenerator import ResultsGenerator, DataTransformer
import DRIVE_common as DRIVE

class DataTransformer3Class(DataTransformer):
	# input: truthData and scoreData shape: (C, H, W)
	# output: move classes/channels to first dimension and flatten all higher dimensions
	def transformScoresPerClass(self, numClasses, scoreData):
		return scoreData.reshape((numClasses, -1))
		#return scoreData.transpose(1,0,2,3)
		
	def transformTruthsPerClass(self, numClasses, truthData):
		return truthData[1:4].reshape((numClasses, -1))

def run(modelName, dataSet):
	scoreFile =  '{}_{}Scores.npz'.format(modelName, dataSet)
	truthDir =  '../../../Images/DRIVE/{}/av'.format(dataSet)
	scoreDict = np.load(scoreFile)
	labelLoader = DRIVE.LabelDataLoaderImagesDRIVE(truthDir, scoreDict.files)
	mm = ResultsGenerator(labelLoader, DataTransformer3Class(), scoreDict=scoreDict, 
		numClasses=3, classNames=DRIVE.classes[1:4], classColors=DRIVE.classColors[1:4])		
	mm.plotPrecisionRecall(plotImageFile='{}PrecRecall.pdf'.format(dataSet), title="3 Class Model")
	mm.plotConfusionMatrix(plotImageFile='{}ConfusionMatrix.pdf'.format(dataSet), title="3 Class Model")
	mm.printClassificationStats(outFile='{}Stats_class.csv'.format(dataSet))
		
if __name__ == "__main__":	
	modelName = 'iter_20000'
	run(modelName, 'train')
	run(modelName, 'test')