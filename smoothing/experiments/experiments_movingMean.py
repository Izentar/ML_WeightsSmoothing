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
    dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, fromGrayToRGB=False)

    smoothingMetadata = None
    if(sf.test_mode.isActive()):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationMovingMean_Metadata(test_movingAvgParam=0.15)
    else:
        smoothingMetadata = dc.DefaultSmoothingOscilationMovingMean_Metadata(movingAvgParam=0.15, 
        epsilon=1e-4, hardEpsilon=1e-7, weightsEpsilon=1e-6, batchPercentMaxStart=0.98)

    #####################
    types = ('predefModel', 'CIFAR100', 'movingMean')
    try:
        obj = models.alexnet()
        metadata.resetOutput()

        smoothingMetadata.movingAvgParam = 0.1
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(numbOfRepetition=5, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)
        experiments.printStats(stat, metadata)

    except Exception as ex:
        experiments.printException(ex, types)


    #####################
    types = ('predefModel', 'CIFAR100', 'movingMean')
    try:
        obj = models.alexnet()
        metadata.resetOutput()
        
        smoothingMetadata.movingAvgParam = 0.15
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(numbOfRepetition=5, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)
        experiments.printStats(stat, metadata)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('predefModel', 'CIFAR100', 'movingMean')
    try:
        obj = models.alexnet()
        metadata.resetOutput()
        
        smoothingMetadata.movingAvgParam = 0.2
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(numbOfRepetition=5, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)
        experiments.printStats(stat, metadata)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('predefModel', 'CIFAR100', 'movingMean')
    try:
        obj = models.alexnet()
        metadata.resetOutput()
        
        smoothingMetadata.movingAvgParam = 0.25
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(numbOfRepetition=5, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)
        experiments.printStats(stat, metadata)
    except Exception as ex:
        experiments.printException(ex, types)