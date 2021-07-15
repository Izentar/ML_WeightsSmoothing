import experiments as experiments

import torch
import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.models.resnet as modResnet

from framework import smoothingFramework as sf
from framework import defaultClasses as dc
from framework.utils import Cutout

import numpy as np
import math

if(__name__ == '__main__'):
    modelDevice = 'cuda:0'
    if(sf.test_mode().isActive()):
        modelDevice="cuda:0"
        
    block = modResnet.BasicBlock

    otherData = {
        "IMG_MEAN":[125.3, 123.0, 113.9],
        "IMG_STD":[63.0, 62.1, 66.7],
        "loop":5,
        "modelName":"wide_resnet",
        "prefix":"set_copyOfExper_",
        "runningAvgSize":10,
        "num_classes":10,
        "schedulerEpoches":[35, 50, 65, 80, 95],
        "lr_sched_gamma":0.2,
        "block":"torchvision.models.resnet.BasicBlock"
    }

    normalize = transforms.Normalize(mean=[x / 255.0 for x in otherData["IMG_MEAN"]],
                                     std=[x / 255.0 for x in otherData["IMG_STD"]])

    metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
    dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=False, pin_memoryTrain=False, epoch=100,
        batchTrainSize=128, batchTestSize=100, startTestAtEpoch=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
        transformTrain=transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            torchvision.transforms.GaussianBlur(5),
            transforms.ToTensor(),
            normalize,
            Cutout(n_holes=1, length=5)
        ]),
        transformTest=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
    optimizerDataDict={"learning_rate":0.1, "momentum":0.9, "weight_decay":0.0005, "nesterov":True}
    modelMetadata = dc.DefaultModel_Metadata(device=modelDevice, lossFuncDataDict={}, optimizerDataDict=optimizerDataDict)


    types = ('predefModel', 'CIFAR10', 'disabled')
    try:
        stats = []
        rootFolder = otherData["prefix"] + sf.Output.getTimeStr() + ''.join(x + "_" for x in types)
        smoothingMetadata = dc.DisabledSmoothing_Metadata()

        for r in range(otherData["loop"]):
            obj = models.densenet169(num_classes=otherData["num_classes"])

            data = dc.DefaultDataCIFAR10(dataMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=otherData["modelName"])
            smoothing = dc.DisabledSmoothing(smoothingMetadata)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                weight_decay=optimizerDataDict['weight_decay'], momentum=optimizerDataDict['momentum'], nesterov=optimizerDataDict['nesterov'])
            scheduler = sf.MultiplicativeLR(optimizer, gamma=otherData["lr_sched_gamma"])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, rootFolder=rootFolder,
                schedulers=[(otherData["schedulerEpoches"], scheduler)], logData=otherData)

            stat.saveSelf(name="stat")

            stats.append(stat)
        experiments.printAvgStats(stats, metadata, runningAvgSize=otherData["runningAvgSize"])
    except Exception as ex:
        experiments.printException(ex, types)