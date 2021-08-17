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
from framework.models.densenet import DenseNet
from framework.models.vgg import vgg19_bn
from framework.models.wideResNet import WideResNet

import numpy as np
import math
import argparse
import sys

VGG = "vgg19_bn"
DENSENET = "densenet"
WRESNET = "wide_resnet"

def getParser():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training', add_help=True)
    parser.add_argument('--optim', default='SGD', choices=["SGD", "Adam"], help='choose optimizer')
    parser.add_argument('--sched', default='multiplic', choices=["multiplic", "adapt"], help='choose scheduler')
    parser.add_argument('--dataset', default='CIFAR10', choices=["CIFAR10", "CIFAR100"], help='choose dataset')
    parser.add_argument('--loops', default=5, type=int, help='how many times test must repeat (default 5)')
    parser.add_argument('--model', default=VGG, choices=[VGG, WRESNET, DENSENET], 
        help='model type (default {})'.format(WRESNET))
    parser.add_argument('--teststep', type=int, default=10, help='A number specifying for which multiples of epochs to call the test')
    parser.add_argument('--dev', default='cuda:0', choices=['cpu', 'cuda:0'], type=str, help='Device of the model. Choose where to \
        store model weights.')
    parser.add_argument('--test', help='debug / test mode', action='store_true')
    parser.add_argument('--debug', help='debug / test mode', action='store_true')
    parser.add_argument('--testarg', help='Use the arguments from the command line. If not, then use special default arguments prepared for \
        test mode.', action='store_true')

    parser.add_argument('--swindow', type=int, default=20, help='sliding window size. To disable set less than 1.')
    parser.add_argument('--savgwindow', type=int, default=10, help='sliding window size for averaged logs. To disable set less than 1.')


    parser.add_argument('--epochs', default=300, type=int, metavar='N',
        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0, type=float,
            metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[],
        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
        metavar='W', help='weight decay (default: 5e-4)')

    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
    parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')    

    parser.add_argument('--smsched', default='swa', choices=["swa"], help='choose smoothing scheduler')
    parser.add_argument('--smoothing', default='disabled', choices=["disabled", "pytorch", "ewma", "generMean", "simplemean"], help='choose smoothing mode')
    parser.add_argument('--smstart', default=0.8, type=float, help='when to start smoothing, exact location ([0;1])')
    parser.add_argument('--smsoftstart', default=0.02, type=float, help='when to enable smoothing, it does not mean it will start calculating average weights ([0;1])')
    parser.add_argument('--smhardend', default=0.99, type=float, help='when to end smoothing and training definitely ([0;1])')
    
    parser.add_argument('--smdev', default='cuda:0', choices=['cpu', 'cuda:0'], type=str, help='Device of the smoothing algorithm. Choose where to \
        store averaged weights.')
    parser.add_argument('--smstartAt', default=1, type=int, help='')
    parser.add_argument('--smlossWarmup', default=293, type=int, help='')
    parser.add_argument('--smlossPatience', default=293, type=int, help='')
    parser.add_argument('--smlossThreshold', default=1e-4, type=float, help='')
    parser.add_argument('--smweightWarmup', default=150, type=int, help='')
    parser.add_argument('--smweightPatience', default=150, type=int, help='')
    parser.add_argument('--smweightThreshold', default=1e-4, type=float, help='')
    parser.add_argument('--smlossThresholdMode', default='rel', choices=['rel', 'abs'], type=str, help='')
    parser.add_argument('--smweightThresholdMode', default='rel', choices=['rel', 'abs'], type=str, help='')


    parser.add_argument('--smlosscontainer', default=195, type=int, help='')
    parser.add_argument('--smweightsumcontsize', default=100, type=int, help='the size of the sum container')
    parser.add_argument('--smmovingparam', default=0.05, type=float, help='moving parameter for the moving mean')
    parser.add_argument('--smgeneralmeanpow', default=1.0, type=float, help='the power of general mean')
    parser.add_argument('--smschedule', type=int, nargs='+', default=[],
        help='Invoke SWALR scheduler at these epochs.')
    parser.add_argument('--smlr', default=0.01, type=float, help='value that SWALR schedule sets as learning rate')
    parser.add_argument('--smoffsched', action='store_true', help='choose if smoothing scheduler should be created')
    parser.add_argument('--smannealEpochs', default=5, type=int, help='')

    parser.add_argument('--factor', default=0.1, type=float, help='')
    parser.add_argument('--patience', default=6, type=int, help='')
    parser.add_argument('--threshold', default=0.0001, type=float, help='')
    parser.add_argument('--minlr', default=0.0, type=float, help='')
    parser.add_argument('--cooldown', default=25, type=int, help='')


    return parser

def createSmoothing(args, model):
    smoothing = None
    smoothingMetadata = None

    index = 0
    if(sf.test_mode.isActive()):
        index = 1

    if(args.smoothing == "disabled"):
        smoothingMetadata = dc.DisabledSmoothing_Metadata()
        smoothing = dc.DisabledSmoothing(smoothingMetadata)

    elif(args.smoothing == "pytorch"):
        if(sf.test_mode.isActive()):
            if(args.testarg):
                smoothingMetadata = dc.Test_DefaultPytorchAveragedSmoothing_Metadata(device=args.smdev, smoothingStartPercent=args.smstart)
            else:
                smoothingMetadata = dc.Test_DefaultPytorchAveragedSmoothing_Metadata(device=args.smdev)    
        else:
            smoothingMetadata = dc.DefaultPytorchAveragedSmoothing_Metadata(device=args.smdev, smoothingStartPercent=args.smstart)
        smoothing = dc.DefaultPytorchAveragedSmoothing(smoothingMetadata=smoothingMetadata, model=model)

    elif(args.smoothing == "ewma"):
        if(sf.test_mode.isActive()):
            if(args.testarg):
                smoothingMetadata = dc.Test_DefaultSmoothingOscilationEWMA_Metadata(device=args.smdev,
                    batchPercentMaxStart=args.smhardend, batchPercentMinStart=args.smsoftstart, startAt=args.smstartAt,
                    lossPatience = args.smlossPatience, lossThreshold = args.smlossThreshold, weightPatience = args.smweightPatience, 
                    weightThreshold = args.smweightThreshold, lossThresholdMode = args.smlossThresholdMode, weightThresholdMode = args.smweightThresholdMode,
                    lossContainerSize=args.smlosscontainer, lossWarmup=args.smlossWarmup, weightWarmup=args.smweightWarmup,
                    weightSumContainerSize=args.smweightsumcontsize,
                    movingAvgParam=args.smmovingparam)
            else:
                smoothingMetadata = dc.Test_DefaultSmoothingOscilationEWMA_Metadata(device=args.smdev)
        else:
            smoothingMetadata = dc.DefaultSmoothingOscilationEWMA_Metadata(device=args.smdev,
                batchPercentMaxStart=args.smhardend, batchPercentMinStart=args.smsoftstart, startAt=args.smstartAt,
                lossPatience = args.smlossPatience, lossThreshold = args.smlossThreshold, weightPatience = args.smweightPatience, 
                weightThreshold = args.smweightThreshold, lossThresholdMode = args.smlossThresholdMode, weightThresholdMode = args.smweightThresholdMode,
                lossContainerSize=args.smlosscontainer, lossWarmup=args.smlossWarmup, weightWarmup=args.smweightWarmup,
                weightSumContainerSize=args.smweightsumcontsize,
                movingAvgParam=args.smmovingparam)

        smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata)

    elif(args.smoothing == "generMean"):
        if(sf.test_mode.isActive()):
            if(args.testarg):
                smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(device=args.smdev,
                    batchPercentMaxStart=args.smhardend, batchPercentMinStart=args.smsoftstart, startAt=args.smstartAt,
                    lossPatience = args.smlossPatience, lossThreshold = args.smlossThreshold, weightPatience = args.smweightPatience, 
                    weightThreshold = args.smweightThreshold, lossThresholdMode = args.smlossThresholdMode, weightThresholdMode = args.smweightThresholdMode,
                    lossContainerSize=args.smlosscontainer, lossWarmup=args.smlossWarmup, weightWarmup=args.smweightWarmup,
                    weightSumContainerSize=args.smweightsumcontsize,
                    generalizedMeanPower=args.smgeneralmeanpow)
            else:
                smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(device=args.smdev)
        else:
            smoothingMetadata = dc.DefaultSmoothingOscilationGeneralizedMean_Metadata(device=args.smdev,
                batchPercentMaxStart=args.smhardend, batchPercentMinStart=args.smsoftstart, startAt=args.smstartAt,
                lossPatience = args.smlossPatience, lossThreshold = args.smlossThreshold, weightPatience = args.smweightPatience, 
                weightThreshold = args.smweightThreshold, lossThresholdMode = args.smlossThresholdMode, weightThresholdMode = args.smweightThresholdMode,
                lossContainerSize=args.smlosscontainer, lossWarmup=args.smlossWarmup, weightWarmup=args.smweightWarmup,
                weightSumContainerSize=args.smweightsumcontsize,
                generalizedMeanPower=args.smgeneralmeanpow)
        smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata)

    elif(args.smoothing == "simplemean"):
        if(sf.test_mode.isActive()):
            if(args.testarg):
                smoothingMetadata = dc.Test_DefaultSmoothingSimpleMean_Metadata(device=args.smdev, batchPercentStart=args.smstart)
            else:
                smoothingMetadata = dc.Test_DefaultSmoothingSimpleMean_Metadata(device=args.smdev)
        else:
            smoothingMetadata = dc.DefaultSmoothingSimpleMean_Metadata(device=args.smdev, batchPercentStart=args.smstart)
        smoothing = dc.DefaultSmoothingSimpleMean(smoothingMetadata)
    else:
        raise Exception("Unknown smoothing type")

    return smoothing, smoothingMetadata

def createData(args, dataMetadata):
    data = None
    if(args.dataset == "CIFAR10"):
        data = dc.DefaultDataCIFAR10(dataMetadata)
    elif(args.dataset == "CIFAR100"):
        data = dc.DefaultDataCIFAR100(dataMetadata)
    else:
        raise Exception()
    return data

def createModel(args, modelMetadata):
    obj = None
    if(args.model == VGG):
        obj = vgg19_bn(num_classes=otherData["num_classes"])
    elif(args.model == WRESNET):
        obj = WideResNet(
            depth=args.depth, 
            widen_factor=args.widen_factor, 
            dropRate=args.drop, 
            num_classes=otherData["num_classes"]
            )
    elif(args.model == DENSENET):
        obj = DenseNet(
            num_classes=otherData["num_classes"],
            depth=args.depth,
            growthRate=args.growthRate,
            compressionRate=args.compressionRate,
            dropRate=args.drop)
    else:
        raise Exception()

    model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=otherData["model"])

    return obj, model

def createOptimizer(args, model, optimizerDataDict):
    optimizer = None
    if(args.optim == "SGD"):
        optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=args.lr, 
            weight_decay=args.weight_decay, momentum=args.momentum, nesterov=optimizerDataDict['nesterov'])
    elif(args.optim == "Adam"):
        optimizer = optim.Adam(model.getNNModelModule().parameters(), lr=args.lr, 
            weight_decay=args.weight_decay)
    else:
        raise Exception()
    return optimizer

def createScheduler(args, optimizer):
    sched = None
    if(args.sched == 'multiplic'):
        sched = sf.MultiplicativeLR(optimizer, gamma=args.gamma)
    elif(args.sched == 'adapt'):
        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, 
            threshold=args.threshold, min_lr=args.minlr, cooldown=args.cooldown)
    else:
        raise Exception()
    return sched, True

def createSmScheduler(args, optimizer):
    smsched = None
    if(args.smsched == 'swa'):
        smsched = torch.optim.swa_utils.SWALR(optimizer, swa_lr=args.smlr, anneal_epochs=args.smannealEpochs, anneal_strategy='linear')
    else:
        raise Exception()
    return smsched, False

def validateArgs(args):
    if(args.sched == 'adapt' and args.teststep != 1):
        raise Exception("Bad combination for '{}' and '{}'.".format(args.sched, args.teststep))

if(__name__ == '__main__'):
    args = getParser().parse_args()
    print("Arguments passed:\n{}".format(args))

    validateArgs(args)

    otherData = {
        "prefix":"set_copyOfExper_",
        "Input parameters" : str(args).split(),
        "num_classes":10 if args.dataset == "CIFAR10" else 100,
        "bash_input": ' '.join(sys.argv),

        "optim": args.optim,
        "model": args.model,
        "dataset": args.dataset,
        VGG + "_params": "num_classes",
        DENSENET + "_params": "num_classes depth growthRate compressionRate drop",
        WRESNET + "_params": "num_classes depth widen_factor drop",
        "SGD_param": "learning_rate weight-decay momentum nesterov",
        "Adam_param": "learning_rate weight-decay"
    }

    CIFAR10_normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    CIFAR100_normalize = transforms.Normalize(mean=(0.5070,  0.4865,  0.4409), std=(0.2673,  0.2564,  0.2761))

    metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
    dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=False, pin_memoryTrain=False, epoch=args.epochs,
        batchTrainSize=128, batchTestSize=100, startTestAtEpoch=list(range(0, args.epochs+args.teststep + 1, args.teststep)) + [1], 
        transformTrain=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            #transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #torchvision.transforms.GaussianBlur(5),
            transforms.ToTensor(),
            CIFAR10_normalize if args.dataset == 'CIFAR10' else CIFAR100_normalize,
            #Cutout(n_holes=1, length=5)
        ]),
        transformTest=transforms.Compose([
            transforms.ToTensor(),
            CIFAR10_normalize if args.dataset == 'CIFAR10' else CIFAR100_normalize
        ]))
    optimizerDataDict={
        "learning_rate":args.lr,
        "momentum":args.momentum, 
        "weight_decay":args.weight_decay, 
        "nesterov":True}
    modelMetadata = dc.DefaultModel_Metadata(device=args.dev, lossFuncDataDict={}, optimizerDataDict=optimizerDataDict)


    types = (args.model, 'predefModel', args.dataset, args.smoothing, args.optim)
    try:
        stats = []
        rootFolder = otherData["prefix"] + sf.Output.getTimeStr() + ''.join(x + "_" for x in types)

        for r in range(args.loops):
            obj, model = createModel(args, modelMetadata)
            optimizer = createOptimizer(args, model=model, optimizerDataDict=optimizerDataDict)
            sched, schedMetric = createScheduler(args, optimizer)
            smsched, smschedMetric = createSmScheduler(args, optimizer)

            loss_fn = nn.CrossEntropyLoss()     

            data = createData(args, dataMetadata)
            smoothing, smoothingMetadata = createSmoothing(args=args, model=model)

            schedSmoothing = sf.SchedulerContainer(schedType='smoothing', importance=1).add(schedule=args.smschedule, scheduler=smsched, metric=smschedMetric)
            schedNormal = sf.SchedulerContainer(schedType='normal', importance=2).add(schedule=list(args.schedule), scheduler=sched, metric=schedMetric)

            schedulers = None
            if(args.smoffsched):
                schedulers = [schedNormal]
            else:
                schedulers = [schedSmoothing, schedNormal]

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, rootFolder=rootFolder,
                schedulers=schedulers, logData=otherData, fileFormat='.png', dpi=300, resolutionInches=6.5, widthTickFreq=0.15, runningAvgSize=args.swindow)

            stat.saveSelf(name="stat")

            stats.append(stat)
        experiments.printAvgStats(stats, metadata, runningAvgSize=args.savgwindow, fileFormat='.png', dpi=300, 
            resolutionInches=6.5, widthTickFreq=0.15)
    except Exception as ex:
        experiments.printException(ex, types)