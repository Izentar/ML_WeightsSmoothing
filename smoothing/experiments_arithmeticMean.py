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
    types = ('predefModel', 'MINST', 'generalizedMean')
    try:
        obj = models.alexnet()
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=3)

        smoothingMetadata = dc.DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, hardEpsilon=1e-6, weightsEpsilon=1e-5)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)


    #####################
    types = ('predefModel', 'MINST', 'generalizedMean')
    try:
        obj = models.alexnet()
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)

        smoothingMetadata = dc.DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-3*5, weightsEpsilon=1e-4*5)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('predefModel', 'MINST', 'generalizedMean')
    try:
        obj = models.alexnet()
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-4, weightsEpsilon=1e-5)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('predefModel', 'MINST', 'generalizedMean')
    try:
        obj = models.alexnet()
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-4*5, weightsEpsilon=1e-5*5)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)

    #####################
    types = ('predefModel', 'MINST', 'generalizedMean')
    try:
        obj = models.alexnet()
        metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
        dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
        
        smoothingMetadata = dc.DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, weightsEpsilon=1e-6)
        modelMetadata = dc.DefaultModel_Metadata()

        stat=dc.run(modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
            modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)
        stat.printPlots(startAt=-10)
    except Exception as ex:
        experiments.printException(ex, types)