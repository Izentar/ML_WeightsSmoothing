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
    types = ('simpleConvModel', 'MINST', 'borderline')
    try:
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=2000)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)


    #####################
    types = ('simpleConvModel', 'MINST', 'borderline')
    try:
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=2500)

        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('simpleConvModel', 'MINST', 'borderline')
    try:
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=2800)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('simpleConvModel', 'MINST', 'borderline')
    try:
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=3000)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)