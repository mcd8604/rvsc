import numpy as np
from sklearn.metrics import accuracy_score

def calculateMetrics(confusionMatrix):
	dims = confusionMatrix.shape
	assert len(dims) == 2 and dims[0] == dims[1], "Invalid confusion matrix shape: {}".format(dims)
	
	numClasses = dims[0]
	metrics = np.zeros((4))
		
	# Pixel Accuracy 
	totalCorrectPredictions = float(np.trace(confusionMatrix))
	totalPixelsPerClass = np.sum(confusionMatrix.transpose(), 0)
	totalPixels = np.sum(totalPixelsPerClass)
	metrics[0] = totalCorrectPredictions / totalPixels
	
	for i in range(numClasses):
		# Mean Accuracy
		correctPredictions = float(confusionMatrix[i,i])
		if totalPixelsPerClass[i] > 0:
			metrics[1] += (1.0 / numClasses) * correctPredictions / totalPixelsPerClass[i]
		
		# Mean IU
		iu = correctPredictions / (totalPixelsPerClass[i] - np.sum([confusionMatrix[j,i] - correctPredictions for j in range(numClasses)]))
		metrics[2] += (float(iu) / numClasses)
		
		# Frequency Weighted IU
		metrics[3] += totalPixelsPerClass[i] * iu
		
	metrics[3] /= totalPixels
	
	return metrics
	