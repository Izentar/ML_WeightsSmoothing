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
import experiments as experiments

if(__name__ == '__main__'):
    #sf.StaticData.TEST_MODE = True

    #####################
    types = ('simpleConvModel', 'MINST', 'weightedMean')
    try:
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(1.05), epsilon=1e-4, weightsEpsilon=1e-6*5)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)


    #####################
    types = ('simpleConvModel', 'MINST', 'weightedMean')
    try:
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(1.1), epsilon=1e-4, weightsEpsilon=1e-6*5)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('simpleConvModel', 'MINST', 'weightedMean')
    try:
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(1.15), epsilon=1e-4, weightsEpsilon=1e-6*5)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('simpleConvModel', 'MINST', 'weightedMean')
    try:
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(1.2), epsilon=1e-4, weightsEpsilon=1e-6*5)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('simpleConvModel', 'MINST', 'weightedMean')
    try:
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(1.25), epsilon=1e-4, weightsEpsilon=1e-6*5)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('simpleConvModel', 'MINST', 'weightedMean')
    try:
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(1.3), epsilon=1e-4, weightsEpsilon=1e-6*5)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)