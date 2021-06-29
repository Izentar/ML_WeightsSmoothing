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
    dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=2, fromGrayToRGB=True)
    loop = 5
    modelName = "wide_resnet50_2"
    prefix = "epsilons_"

    types = ('predefModel', 'MNIST', 'movingMean')
    try:
        stats = []
        rootFolder = prefix + sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        smoothingMetadata = dc.DefaultSmoothingOscilationMovingMean_Metadata(movingAvgParam=0.15, 
            epsilon=1e-1, hardEpsilon=1e-3, weightsEpsilon=1e-2, batchPercentMaxStart=0.98)

        for r in range(loop):
            obj = models.wide_resnet50_2()
            metadata.resetOutput()
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=2, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                modelPredefObjName=modelName, rootFolder=rootFolder)
            stats.append(stat.pop()) # weź pod uwagę tylko ostatni wynik (najlepiej wyćwiczony)
        experiments.printAvgStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)

    try:
        stats = []
        rootFolder = prefix + sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        smoothingMetadata = dc.DefaultSmoothingOscilationMovingMean_Metadata(movingAvgParam=0.15, 
            epsilon=1e-2, hardEpsilon=1e-4, weightsEpsilon=1e-3, batchPercentMaxStart=0.98)

        for r in range(loop):
            obj = models.wide_resnet50_2()
            metadata.resetOutput()
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=2, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                modelPredefObjName=modelName, rootFolder=rootFolder)
            stats.append(stat.pop()) # weź pod uwagę tylko ostatni wynik (najlepiej wyćwiczony)
        experiments.printAvgStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)

    try:
        stats = []
        rootFolder = prefix + sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        smoothingMetadata = dc.DefaultSmoothingOscilationMovingMean_Metadata(movingAvgParam=0.15, 
            epsilon=1e-3, hardEpsilon=1e-5, weightsEpsilon=1e-4, batchPercentMaxStart=0.98)

        for r in range(loop):
            obj = models.wide_resnet50_2()
            metadata.resetOutput()
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=2, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                modelPredefObjName=modelName, rootFolder=rootFolder)
            stats.append(stat.pop()) # weź pod uwagę tylko ostatni wynik (najlepiej wyćwiczony)
        experiments.printAvgStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)



    try:
        stats = []
        rootFolder = prefix + sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        smoothingMetadata = dc.DefaultSmoothingOscilationMovingMean_Metadata(movingAvgParam=0.15, 
            epsilon=1e-4, hardEpsilon=1e-6, weightsEpsilon=1e-5, batchPercentMaxStart=0.98)

        for r in range(loop):
            obj = models.wide_resnet50_2()
            metadata.resetOutput()
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=2, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                modelPredefObjName=modelName, rootFolder=rootFolder)
            stats.append(stat.pop()) # weź pod uwagę tylko ostatni wynik (najlepiej wyćwiczony)
        experiments.printAvgStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)


    try:
        stats = []
        rootFolder = prefix + sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        smoothingMetadata = dc.DefaultSmoothingOscilationMovingMean_Metadata(movingAvgParam=0.15, 
            epsilon=1e-5, hardEpsilon=1e-7, weightsEpsilon=1e-6, batchPercentMaxStart=0.98)

        for r in range(loop):
            obj = models.wide_resnet50_2()
            metadata.resetOutput()
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=2, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                modelPredefObjName=modelName, rootFolder=rootFolder)
            stats.append(stat.pop()) # weź pod uwagę tylko ostatni wynik (najlepiej wyćwiczony)
        experiments.printAvgStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)


    try:
        stats = []
        rootFolder = prefix + sf.Output.getTimeStr() +  ''.join(x + "_" for x in types) + "set"
        smoothingMetadata = dc.DefaultSmoothingOscilationMovingMean_Metadata(movingAvgParam=0.15, 
            epsilon=1e-6, hardEpsilon=1e-8, weightsEpsilon=1e-7, batchPercentMaxStart=0.98)

        for r in range(loop):
            obj = models.wide_resnet50_2()
            metadata.resetOutput()
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=2, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj,
                modelPredefObjName=modelName, rootFolder=rootFolder)
            stats.append(stat.pop()) # weź pod uwagę tylko ostatni wynik (najlepiej wyćwiczony)
        experiments.printAvgStats(stats, metadata)
    except Exception as ex:
        experiments.printException(ex, types)