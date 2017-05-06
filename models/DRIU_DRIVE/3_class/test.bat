@echo off 

SET REPO_ROOT=..\..\..
SET PYTHONPATH=%PYTHONPATH%;%REPO_ROOT%\python;
SET DRIVE=%REPO_ROOT%\Images\DRIVE
SET MODEL=iter_20000

REM Run Caffe: Generate Test Scores
python %REPO_ROOT%\rvsc\ModelTester.py -i %DRIVE%\train\images -f %DRIVE%\train\fileNames.txt -w _%MODEL%.caffemodel -o %MODEL%_trainScores.npz
python %REPO_ROOT%\rvsc\ModelTester.py -i %DRIVE%\test\images -f %DRIVE%\test\fileNames.txt -w _%MODEL%.caffemodel -o %MODEL%_testScores.npz

python generateResults.py