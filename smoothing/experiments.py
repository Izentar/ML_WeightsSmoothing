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
    obj = models.alexnet()
    metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
    dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True)
    smoothingMetadata = dc.DefaultSmoothingOscilationMovingMean_Metadata()
    modelMetadata = dc.DefaultModel_Metadata()

    stat=dc.run(modelType='predefModel', dataType='MINST', smoothingType='movingMean', metadataObj=metadata, 
        modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)
    stat.printPlots(startAt=-10)
