import experiments as experiments

import torch
import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from framework import smoothingFramework as sf
from framework import defaultClasses as dc

# wzorowane na pracy https://paperswithcode.com/paper/wide-residual-networks

if(__name__ == '__main__'):
    modelDevice = 'cuda:0'
    if(sf.test_mode().isActive()):
        modelDevice="cuda:0"

    metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
    dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=False, pin_memoryTrain=False, epoch=200, fromGrayToRGB=False,
        batchTrainSize=128, batchTestSize=128)
    optimizerDataDict={"learning_rate":0.1, "momentum":0.9, "weight_decay":0.0005}
    modelMetadata = dc.DefaultModel_Metadata(device=modelDevice, lossFuncDataDict={}, optimizerDataDict=optimizerDataDict)
    loop = 5
    modelName = "wide_resnet"
    prefix = "set_copyOfExper_"
    runningAvgSize = 10


    types = ('predefModel', 'CIFAR10', 'pytorchSWA')
    try:
        stats = []
        rootFolder = prefix + sf.Output.getTimeStr() + ''.join(x + "_" for x in types)
        smoothingMetadata = dc.DefaultPytorchAveragedSmoothing_Metadata(device='cuda:0')

        for r in range(loop):

            obj = models.resnext50_32x4d()

            data = dc.DefaultDataCIFAR10(dataMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)
            smoothing = dc.DefaultPytorchAveragedSmoothing(smoothingMetadata,  model=model)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                weight_decay=optimizerDataDict['weight_decay'], momentum=optimizerDataDict['momentum'])
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=0.2, verbose=True)
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, rootFolder=rootFolder,
                schedulers=[([60, 120, 160], scheduler)])

            stat.saveSelf(name="stat")

            stats.append(stat)
        experiments.printAvgStats(stats, metadata, runningAvgSize=runningAvgSize)
    except Exception as ex:
        experiments.printException(ex, types)



    types = ('predefModel', 'CIFAR10', 'EWMA')
    try:
        stats = []
        rootFolder = prefix + sf.Output.getTimeStr() +  ''.join(x + "_" for x in types)
        smoothingMetadata = dc.DefaultSmoothingOscilationEWMA_Metadata(movingAvgParam=0.05, 
            epsilon=1e-5, hardEpsilon=1e-7, weightsEpsilon=1e-6, batchPercentMaxStart=0.98, device='cuda:0')

        for r in range(loop):
            obj = models.resnext50_32x4d()

            data = dc.DefaultDataCIFAR10(dataMetadata)
            smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                weight_decay=optimizerDataDict['weight_decay'], momentum=optimizerDataDict['momentum'])
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=0.2, verbose=True)
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, rootFolder=rootFolder,
                schedulers=[([60, 120, 160], scheduler)])

            stat.saveSelf(name="stat")

            stats.append(stat)
        experiments.printAvgStats(stats, metadata, runningAvgSize=runningAvgSize)
    except Exception as ex:
        experiments.printException(ex, types)