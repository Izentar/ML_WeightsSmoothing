import sys, os
from pathlib import Path

PrimaryWorkingDir = None
newDir = None

def run():
    global PrimaryWorkingDir
    global newDir

    if(os.path.basename(os.path.abspath(__file__)) == 'smoothing' and PrimaryWorkingDir is None):
        PrimaryWorkingDir = os.getcwd()

    if(PrimaryWorkingDir is None):
        PrimaryWorkingDir = os.getcwd()
        print(PrimaryWorkingDir)

        newDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(newDir)
        # zmień working directory na inny folder.
        os.chdir(newDir)
        #print(os.getcwd())