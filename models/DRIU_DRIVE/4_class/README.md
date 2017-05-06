# 4-Class Model
This is an extension of the DRIU base model from a binary classification to 4 classes - background, artery, vein, overlap  
Weights are transplanted from the base model layer 'new-score-weighting' to 'new-score-weighting_av' for only artery and vein classes: indices (1,2)

To Train (on Linux/Slurm):
1.	Create base model weights: python transplant.py
2.	Run train.job

To Test (on Windows):
1.	Run test.bat
