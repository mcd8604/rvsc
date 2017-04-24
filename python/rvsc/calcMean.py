import numpy as np
import cv2

'''
DRIVE_dir = 'C:/caffe/caffe_27/DRIU/Images/DRIVE/train/images/'
images = []
for i in range(20,40):
	imgID = str(i+1).zfill(2)
	inFilePath = '{}/{}_training.png'.format(DRIVE_dir, imgID)
	images.append(cv2.imread(inFilePath)[:,:,::-1]	)
imgVol = np.stack(images, 0)
print np.mean(imgVol, (0,1,2))
'''

imageNames = []
with open('../Images/DRIVE/test/filenames_10.txt') as f:
	imageNames = [i.strip() for i in f.readlines()]
	
scoreDict = np.load('../models/DRIU_DRIVE/model_1/iter_20000_testScores.npz')
images = []
for imageName in imageNames:
	images.append(scoreDict[imageName].transpose(1,2,0))
imgVol = np.stack(images, 0)
print np.mean(imgVol, (0,1,2))