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
import argparse

def getParser():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training', add_help=True)
    parser.add_argument('--optim', default='SGD', choices=["SGD", "Adam"], help='choose optimizer')
    parser.add_argument('--dataset', default='CIFAR10', choices=["CIFAR10", "CIFAR100"], help='choose dataset')
    parser.add_argument('--loops', default=5, type=int, help='how many times test must repeat (default 5)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate (default 0.1)')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='optimizer weight decay (default 0.01)')
    parser.add_argument('--model', default="wide_resnet", choices=["vgg19", "wide_resnet", "densenet169"], 
        help='model type (default wide_resnet)')
    parser.add_argument('--test', help='debug / test mode', action='store_true')
    parser.add_argument('--debug', help='debug / test mode', action='store_true')
    

    return parser

if(__name__ == '__main__'):
    args = getParser().parse_args()
    print("Arguments passed:\n{}".format(args))

    modelDevice = 'cuda:0'
    if(sf.test_mode().isActive()):
        modelDevice="cuda:0"

    otherData = {
        "IMG_MEAN":[125.3, 123.0, 113.9],
        "IMG_STD":[63.0, 62.1, 66.7],
        "loop":args.loops,
        "modelName":"wide_resnet",
        "prefix":"set_copyOfExper_",
        "runningAvgSize":10,
        "num_classes":10,
        "schedulerEpoches":[50, 110, 160],
        "lr_sched_gamma":0.2,
        "optim": args.optim,
        "model": args.model,
        "dataset": args.dataset
    }

    normalize = transforms.Normalize(mean=[x / 255.0 for x in otherData["IMG_MEAN"]],
                                     std=[x / 255.0 for x in otherData["IMG_STD"]])

    metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
    dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=False, pin_memoryTrain=False, epoch=100,
        batchTrainSize=128, batchTestSize=100, startTestAtEpoch=list(range(0, 171, 10)) + [1], 
        transformTrain=transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            torchvision.transforms.GaussianBlur(5),
            transforms.ToTensor(),
            normalize,
            Cutout(n_holes=2, length=5)
        ]),
        transformTest=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
    optimizerDataDict={
        "learning_rate":args.lr,
        "momentum":0.9, 
        "weight_decay":args.weight_decay, 
        "nesterov":True}
    modelMetadata = dc.DefaultModel_Metadata(device=modelDevice, lossFuncDataDict={}, optimizerDataDict=optimizerDataDict)


    types = ('VGG19', 'predefModel', args.dataset, 'disabled', args.optim)
    try:
        stats = []
        rootFolder = otherData["prefix"] + sf.Output.getTimeStr() + ''.join(x + "_" for x in types)
        smoothingMetadata = dc.DisabledSmoothing_Metadata()

        for r in range(otherData["loop"]):
            obj = None
            if(args.model == "vgg19"):
                obj = models.vgg19_bn(num_classes=otherData["num_classes"])
            elif(args.model == "wide_resnet"):
                obj = models.wide_resnet50_2(num_classes=otherData["num_classes"])
            elif(args.model == "densenet169"):
                obj = models.densenet169(num_classes=otherData["num_classes"])
            else:
                raise Exception()

            data = None
            if(args.dataset == "CIFAR10"):
                data = dc.DefaultDataCIFAR10(dataMetadata)
            elif(args.dataset == "CIFAR100"):
                data = dc.DefaultDataCIFAR100(dataMetadata)
            else:
                raise Exception()

            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=otherData["modelName"])
            smoothing = dc.DisabledSmoothing(smoothingMetadata)

            optimizer = None
            if(args.optim == "SGD"):
                optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                    weight_decay=optimizerDataDict['weight_decay'], momentum=optimizerDataDict['momentum'], nesterov=optimizerDataDict['nesterov'])
            elif(args.optim == "Adam"):
                optimizer = optim.Adam(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                    weight_decay=optimizerDataDict['weight_decay'])
            else:
                raise Exception()
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