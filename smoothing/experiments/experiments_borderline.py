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
    #sf.StaticData.TEST_MODE = True

    metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
    dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, fromGrayToRGB=False)
    loop = 5

    #####################
    types = ('predefModel', 'CIFAR10', 'borderline')
    try:
        stats = []
        rootFolder = sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        for r in range(loop):
            obj = models.alexnet()
            metadata.resetOutput()
            
            smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=2000)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=1, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                rootFolder=rootFolder)
            stats.append(stat)
        experiments.printStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)


    #####################
    types = ('predefModel', 'CIFAR10', 'borderline')
    try:
        stats = []
        rootFolder = sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        for r in range(loop):
            obj = models.alexnet()
            metadata.resetOutput()
            
            smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=2500)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=1, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                rootFolder=rootFolder)
            stats.append(stat)
        experiments.printStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('predefModel', 'CIFAR10', 'borderline')
    try:
        stats = []
        rootFolder = sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        for r in range(loop):
            obj = models.alexnet()
            metadata.resetOutput()
            
            smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=2800)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=1, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                rootFolder=rootFolder)
            stats.append(stat)
        experiments.printStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('predefModel', 'CIFAR10', 'borderline')
    try:
        stats = []
        rootFolder = sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        for r in range(loop):
            obj = models.alexnet()
            metadata.resetOutput()
            
            smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=3000)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=1, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                rootFolder=rootFolder)
            stats.append(stat)
        experiments.printStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)