import experiments as experiments

import torch
import torchvision
import torch.optim as optim
from framework import smoothingFramework as sf
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from framework import defaultClasses as dc

if(__name__ == '__main__'):
    sf.StaticData.TEST_MODE = True

    metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
    dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, fromGrayToRGB=False)
    loop = 5
    modelName = "alexnet"

    smoothingMetadata = None
    if(sf.test_mode.isActive()):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationEWMA_Metadata(test_movingAvgParam=0.15,
            test_epsilon=1, test_hardEpsilon=1e-7, test_weightsEpsilon=1000)
    else:
        smoothingMetadata = dc.DefaultSmoothingOscilationEWMA_Metadata(movingAvgParam=0.15, 
        epsilon=1e-5, hardEpsilon=1e-7, weightsEpsilon=1e-6, batchPercentMaxStart=0.98)
    '''
    #####################
    types = ('predefModel', 'CIFAR10', 'movingMean')
    try:
        stats = []
        rootFolder = sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        for r in range(loop):
            obj = models.alexnet()
            metadata.resetOutput()

            smoothingMetadata.movingAvgParam = 0.1
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=1, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                rootFolder=rootFolder)
            stats.append(stat)
        experiments.printStats(stats, metadata)

    except Exception as ex:
        experiments.printException(ex, types)

    '''
    #####################
    types = ('predefModel', 'CIFAR10', 'movingMean')
    modelName = "wide_resnet"
    try:
        stats = []
        rootFolder = sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        for r in range(loop):
            obj = models.wide_resnet50_2()
            metadata.resetOutput()
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=15, fromGrayToRGB=False)
            
            smoothingMetadata.movingAvgParam = 0.15
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=1, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                modelPredefObjName=modelName, rootFolder=rootFolder)
            stats.append(stat.pop()) # weź pod uwagę tylko ostatni wynik (najlepiej wyćwiczony)
        experiments.printAvgStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)
'''
    #####################
    types = ('predefModel', 'CIFAR10', 'movingMean')
    try:
        stats = []
        rootFolder = sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        for r in range(loop):
            obj = models.alexnet()
            metadata.resetOutput()
            
            smoothingMetadata.movingAvgParam = 0.2
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=1, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                rootFolder=rootFolder)
            stats.append(stat)
        experiments.printStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('predefModel', 'CIFAR10', 'movingMean')
    try:
        stats = []
        rootFolder = sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        for r in range(loop):
            obj = models.alexnet()
            metadata.resetOutput()
            
            smoothingMetadata.movingAvgParam = 0.25
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=1, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                rootFolder=rootFolder)
            stats.append(stat)
        experiments.printStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)
'''