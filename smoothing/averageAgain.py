import sys, os
from pathlib import Path
sys.path.append(os.path.abspath(str(Path(__file__).parents[1])))

import torch
import torchvision
import torch.optim as optim
from framework import smoothingFramework as sf
from framework import defaultClasses as dc
from experiments import experiments

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os

def average(paths):
    stats = []
    for p in paths:
        #os.path.join(sf.StaticData.LOG_FOLDER
        stat = torch.load(p)['stat']
        if(not isinstance(stat, sf.Statistics)):
            raise Exception("The object loaded is not a Statistics class")
        stats.append(stat)

    experiments.printAvgStats(stats, 
        os.path.join("custom_avg_" + sf.Output.getTimeStr()), runningAvgSize=10)

if(__name__ == '__main__'):
    paths = sys.argv[1:]

    if(len(paths) == 0):
        print("Paths not provided.")
        exit(1)

    average(paths)

    '''s = sf.Statistics()
    torch.save(s, "tmpStat")

    
    stat = torch.load("tmpStat")'''
