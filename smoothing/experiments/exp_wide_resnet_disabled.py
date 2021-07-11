import experiments as experiments

import torch
import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models.resnet as modResnet
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from framework import smoothingFramework as sf
from framework import defaultClasses as dc

# wzorowane na pracy https://paperswithcode.com/paper/wide-residual-networks
# model wzorowany na resnet18 https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/module.py

if(__name__ == '__main__'):
    modelDevice = 'cuda:0'
    if(sf.test_mode().isActive()):
        modelDevice="cuda:0"

    metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
    dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=False, pin_memoryTrain=False, epoch=100, fromGrayToRGB=False,
        batchTrainSize=125, batchTestSize=125, startTestAtEpoch=[0, 24, 44, 74, 99])
    optimizerDataDict={"learning_rate":0.1, "momentum":0.9, "weight_decay":0.001}
    modelMetadata = dc.DefaultModel_Metadata(device=modelDevice, lossFuncDataDict={}, optimizerDataDict=optimizerDataDict)
    loop = 5
    modelName = "wide_resnet"
    prefix = "set_copyOfExper_"
    runningAvgSize = 10
    num_classes = 10
    layers = [2, 2, 2, 2]
    block = modResnet.BasicBlock


    types = ('predefModel', 'CIFAR10', 'disabled')
    try:
        stats = []
        rootFolder = prefix + sf.Output.getTimeStr() + ''.join(x + "_" for x in types)
        smoothingMetadata = dc.DisabledSmoothing_Metadata()

        for r in range(loop):

            obj = models.ResNet(block, layers, num_classes=num_classes)

            data = dc.DefaultDataCIFAR10(dataMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)
            smoothing = dc.DisabledSmoothing(smoothingMetadata)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                weight_decay=optimizerDataDict['weight_decay'], momentum=optimizerDataDict['momentum'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, rootFolder=rootFolder,
                schedulers=[([30, 60, 90, 120, 150, 180], scheduler)])

            stat.saveSelf(name="stat")

            stats.append(stat)
        experiments.printAvgStats(stats, metadata, runningAvgSize=runningAvgSize)
    except Exception as ex:
        experiments.printException(ex, types)
