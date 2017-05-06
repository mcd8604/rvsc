# 2-Class Models
This is an extension of the DRIU base model from a binary classification to 2 classes - artery+overlap, vein+overlap  
Weights are transplanted from the base model layer 'new-score-weighting' to 'new-score-weighting_av' for only artery and vein classes: indices (0,1)

To Train (on Linux/Slurm):
1.	Copy model weights from 3-class model: cp ../3_class/train_start.caffemodel
2.	Run train.job
3.	Run train_aug.job
4.	Run train_control.job

To Test (on Windows):
1.	Run test.bat
