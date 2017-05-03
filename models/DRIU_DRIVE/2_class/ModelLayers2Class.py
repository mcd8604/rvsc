import numpy as np
from DenseLabelInputLayer import DenseLabelInputLayer

class InputLayer2Class(DenseLabelInputLayer):	
	# Override this method for handling ground truth differently in a specific model
	def prepareLabelFromImage(self, image):
		# make sure image is 3-channel
		assert len(image.shape) == 3 and image.shape[2] == 3, "Label images must have 3 channels"
		
		#r,g,b = cv2.split(image)
		r = image[...,0]
		g = image[...,1]
		b = image[...,2]
		unknown = (np.minimum(b, r)) > 0
		background = ((b+g+r) == 0)
		overlap = (g - unknown) > 0
		artery = (r - unknown) > 0
		vein = (b - unknown) > 0
		
		# [1 x C x H x W]
		self.label = np.stack([artery+overlap, vein+overlap], 0)