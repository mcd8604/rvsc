import os
import numpy as np
import cv2

'''
This script is for preprocessing the training and testing images by rotating 
at 90 degree intervals and mirroring over one axis, producing 8 unique-ish images.

Expected image directory structure:
DRIVE_DIR/
....train/
........filenames.txt
........images/
........av/
....test
........filenames.txt
........images/
........av/

Output images are named <imageName>_<n>.<imageExt> where n is 1:8 and saved to the same directory.
(<imageName>_pp_1.<imageExt> will always be identical to the input)
New image names files are created also: filenames_pp.txt (default)
'''

def parseArgs():	
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("DRIVE_DIR")
	args = parser.parse_args()		
	if not os.path.exists(args.DRIVE_DIR):
		raise IOError("DRIVE directory does not exist: {}".format(args.DRIVE_DIR))			
	return args
	
def getInputImageNames(imageNamesFilePath):	
	if not os.path.isfile(imageNamesFilePath):
		raise IOError('Cannot find image names file: {}'.format(imageNamesFilePath))		
	imageNames = []
	with open(imageNamesFilePath) as f:
		imageNames = [i.strip() for i in f.readlines()]		
	return imageNames

def preProcessImage(imageDir, imageName, image):		
	prefix, ext = os.path.splitext(imageName)
	rotatedImages = []
	newImageNames = []
	i = 0
	for i in range(4):
		# rotate
		r = np.rot90(image, i)
		rotatedImages.append(r)
		newImageName = '{}_{}{}'.format(prefix, i+1, ext)
		newImageNames.append(newImageName)
		cv2.imwrite(os.path.join(imageDir, newImageName), r)
	for r in rotatedImages:
		# mirror
		i += 1
		newImageName = '{}_{}{}'.format(prefix, i+1, ext)
		newImageNames.append(newImageName)
		cv2.imwrite(os.path.join(imageDir, newImageName), np.fliplr(r))
	return newImageNames
	
def writeNewImageNamesFile(imageNamesList, outDir, outFileName='filenames_pp.txt'):
	outFilePath = os.path.join(outDir, outFileName)
	with open(outFilePath, 'w') as f:
		for imageName in imageNamesList:
			f.write('{}\n'.format(imageName))
	
def preProcessImages(trainTestDir, subDirs=['images', 'av', 'artery', 'vein'], imageNamesFile='filenames.txt'):
	imageFileNames = getInputImageNames(os.path.join(trainTestDir, imageNamesFile))
	for subDir in subDirs:
		imageDir = os.path.join(trainTestDir, subDir)
		fileNamesList = []
		for imageName in imageFileNames:		
			# load and pre-process input image
			imagePath = os.path.join(imageDir, imageName)
			assert os.path.isfile(imagePath)
			image = cv2.imread(imagePath)
			newImageNames = preProcessImage(imageDir, imageName, image)
			fileNamesList.extend(newImageNames)
		# If first subDir run, create the new filenames file
		if(subDir == subDirs[0]):
			writeNewImageNamesFile(fileNamesList, trainTestDir)
	
if __name__ == "__main__":	
	args = parseArgs()		
	trainDir = os.path.join(args.DRIVE_DIR, 'train')
	testDir = os.path.join(args.DRIVE_DIR, 'test')	
	preProcessImages(trainDir)
	preProcessImages(testDir)		