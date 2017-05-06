@echo off 

SET REPO_ROOT=..\..\..
SET PYTHONPATH=%PYTHONPATH%;%REPO_ROOT%\python;
SET DRIVE=%REPO_ROOT%\Images\DRIVE

REM Run Caffe: Generate Test Scores
SET MODEL=iter_20000
python %REPO_ROOT%\rvsc\ModelTester.py -i %DRIVE%\train\images -f %DRIVE%\train\fileNames.txt -w _%MODEL%.caffemodel -o %MODEL%_trainScores.npz
python %REPO_ROOT%\rvsc\ModelTester.py -i %DRIVE%\test\images -f %DRIVE%\test\fileNames.txt -w _%MODEL%.caffemodel -o %MODEL%_testScores.npz

SET MODEL=aug_iter_20000
python %REPO_ROOT%\rvsc\ModelTester.py -i %DRIVE%\train\images -f %DRIVE%\train\fileNames_160.txt -w %MODEL%.caffemodel -o %MODEL%_trainScores.npz
python %REPO_ROOT%\rvsc\ModelTester.py -i %DRIVE%\test\images -f %DRIVE%\test\fileNames_160.txt -w %MODEL%.caffemodel -o %MODEL%_testScores.npz

SET MODEL=control_iter_20000
python %REPO_ROOT%\rvsc\ModelTester.py -i %DRIVE%\train\images -f %DRIVE%\train\fileNames.txt -w %MODEL%.caffemodel -o %MODEL%_trainScores.npz
python %REPO_ROOT%\rvsc\ModelTester.py -i %DRIVE%\test\images -f %DRIVE%\test\fileNames.txt -w %MODEL%.caffemodel -o %MODEL%_testScores.npz

python generateResults.py