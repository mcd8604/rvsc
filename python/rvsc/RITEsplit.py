import os
import cv2
import numpy as np
import DRIVE_common

# temp flag - for each image, combine artery/vein into a single label file
singleFile = True

# Load all RITE ground truth images, 
# split into separate artery/vein images and save		
if __name__ == "__main__":
	# inputs
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--truthDir", default="../Images/DRIVE/test/av")
	parser.add_argument("--imageNamesFile", default="../Images/DRIVE/test/filenames.txt")
	args = parser.parse_args()
	
	# input validation
	if not os.path.exists(args.truthDir):
		raise IOError("Truth dir does not exist: {}".format(args.truthDir))
	if not os.path.isfile(args.imageNamesFile):
		raise IOError('Cannot find imageNamesFile: {}'.format(args.imageNamesFile))
	
	# set up dirs
	arteryDir = '{}/../artery'.format(args.truthDir)
	veinDir = '{}/../vein'.format(args.truthDir)
	if not os.path.exists(arteryDir):
		os.makedirs(arteryDir)
	if not os.path.exists(veinDir):
		os.makedirs(veinDir)
		
	# get image names
	imageNames = []
	with open(args.imageNamesFile) as f:
		imageNames = [i.strip() for i in f.readlines()]
	
	# load, split, save
	for imageName in imageNames:
		truthImageFilePath = '{}/{}'.format(args.truthDir, imageName)
		if not os.path.exists(truthImageFilePath):
			raise IOError("Truth image does not exist: {}".format(truthImageFilePath))	
		truthImage = cv2.imread(truthImageFilePath)[:,:,::-1]
		labelImage = DRIVE_common.truthImageToLabelImage(True, truthImage).astype(np.uint8)
		if singleFile:
			cv2.imwrite('{}/{}'.format('avLabel', imageName), labelImage[0] + (labelImage[1]*2))
		else:
			cv2.imwrite('{}/{}'.format(arteryDir, imageName), labelImage[0])
			cv2.imwrite('{}/{}'.format(veinDir, imageName), labelImage[1])