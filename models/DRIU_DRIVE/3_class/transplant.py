import caffe
import numpy as np

oldProtoFile = '../base/deploy_DRIVE.prototxt'
oldWeightsFile = '../base/DRIU_DRIVE.caffemodel'

newProtoFile = 'train.prototxt'
newWeightsFile = 'train_start.caffemodel'

# Load original net
oldNet = caffe.Net(oldProtoFile, weights=oldWeightsFile, phase=1)
# Load original weights into new net - excludes the modified layers
newNet = caffe.Net(newProtoFile, weights=oldWeightsFile, phase=1)

# Initialize new-score-weighting_av layer using old weights for artery and vein
# leave overlap class empty 
oldWeights = oldNet.params['new-score-weighting'][0].data[0][0]
newNet.params['new-score-weighting_av'][0].data[0] = oldWeights
newNet.params['new-score-weighting_av'][0].data[1] = oldWeights
newNet.save(newWeightsFile)