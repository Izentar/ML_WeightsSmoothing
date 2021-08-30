from framework import defaultClasses as dc
from framework import smoothingFramework as sf
from experiments import exp_pytorch as ex
import pandas as pd
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from framework.test import utils as ut
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import gc
import sys

"""
    Sprawdza jedynie występowanie błędów składni oraz wywołania metod / funkcji przy wywołaniu przez parser.
"""

class Test_RunExperiment(ut.Utils):
    RESIZE_TO = 64
    MAP = {
        'optim': None,
        'sched': None,
        'dataset': None,
        'loops': None,
        'model': 'name',
        'teststep': None,
        'dev': 'device',
        'test': None,
        'debug': None,
        'testarg': None,
        'swindow': None,
        'savgwindow': None,
        'epochs': 'epoch',
        'lr': None,
        'learning-rate': None,
        'drop': 'dropRate',
        'dropout': None,
        'schedule': 'schedule',
        'gamma': 'gamma',
        'momentum': 'momentum',
        'weight-decay': 'weight_decay',
        'wd': None,
        'depth': 'depth',
        'widen-factor': None,
        'growthRate': 'growthRate',
        'compressionRate': 'compressionRate',
        'smsched': None,
        'smoothing': None,
        'smstart': 'smoothingStartPercent',
        'smsoftstart': 'batchPercentMinStart',
        'smhardend': 'batchPercentMaxStart',
        'smdev': 'device',
        'smstartAt': 'startAt',
        'smlossWarmup': 'lossWarmup',
        'smlossPatience': 'lossPatience',
        'smlossThreshold': 'lossThreshold',
        'smweightWarmup': 'weightWarmup',
        'smweightPatience': 'weightPatience',
        'smweightThreshold': 'weightThreshold',
        'smlossThresholdMode': 'lossThresholdMode',
        'smweightThresholdMode': 'weightThresholdMode',
        'smlosscontainer': 'lossContainerSize',
        'smweightsumcontsize': 'weightSumContainerSize',
        'smmovingparam': 'movingAvgParam',
        'smgeneralmeanpow': 'generalizedMeanPower',
        'smschedule': 'schedule',
        'smlr': 'swa_lr',
        'smoffsched': None,
        'smannealEpochs': 'anneal_epochs',
        'factor': 'factor',
        'patience': 'patience',
        'threshold': 'threshold',
        'minlr': 'min_lr',
        'cooldown': 'cooldown'
    }

    def setUp(self):
        # czyszczeie pamięci
        torch.cuda.empty_cache()
        gc.collect()
        print("cuda_allocated: {} Mb".format(torch.cuda.memory_allocated()))

        self.initArgs = '--loops 1 --sched multiplic --epochs 164 --gamma 0.001 --patience 0 --cooldown 0 --teststep 1 --testarg'.split()

        self.MNIST_normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.trainTrans = transforms.Compose([
            transforms.Resize(Test_RunExperiment.RESIZE_TO),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.repeat(3, 1, 1)),
            self.MNIST_normalize
        ])
        self.testTrans = transforms.Compose([
            transforms.Resize(Test_RunExperiment.RESIZE_TO),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.repeat(3, 1, 1)),
            self.MNIST_normalize
        ])

        self.CIFAR10_trainTrans = transforms.Compose([
            transforms.Resize(Test_RunExperiment.RESIZE_TO),
            transforms.ToTensor(),
            self.MNIST_normalize
        ])
        self.CIFAR10_testTrans = transforms.Compose([
            transforms.Resize(Test_RunExperiment.RESIZE_TO),
            transforms.ToTensor(),
            self.MNIST_normalize
        ])

        self.CIFAR100_trainTrans = transforms.Compose([
            transforms.Resize(Test_RunExperiment.RESIZE_TO),
            transforms.ToTensor(),
            self.MNIST_normalize
        ])
        self.CIFAR100_testTrans = transforms.Compose([
            transforms.Resize(Test_RunExperiment.RESIZE_TO),
            transforms.ToTensor(),
            self.MNIST_normalize
        ])
    
    def getRevMap(self):
        newDict = {}
        for k, v in Test_RunExperiment.MAP.items():
            if(v is not None and v in newDict):
                if(isinstance(newDict[v], list)):
                    newDict[v].append(k)
                else:
                    tmp = newDict[v]
                    newDict[v] = [k]
                    newDict[v].append(tmp)
            else:
                newDict[v] = [k]
        return newDict

    def runMain(self, arg):
        parArgs = ex.getParser().parse_args(args=arg)
        ex.main(args=parArgs, raiseExc=True)

    def compareParsed(self, args, obj):
        objDict = vars(obj)
        namespace = vars(args)
        count = 0
        countInMap = 0
        mm = self.getRevMap()
        for okey, oval in objDict.items():
            if(okey in mm): # czy dana zmienna obiektu posiada wartość z parsera
                countInMap = countInMap + len(mm[okey])
                for key, val in namespace.items():
                    if(key in mm[okey]): # tylko pasujące klucze
                        self.cmpPandas(oval, "var: '{}' \nfrom '{}'".format(okey, str(obj)), val)
                        count = count + 1
        self.cmpPandas(count, "var number '{}'; from map '{}'".format(count, countInMap), countInMap)

class Test_parser(Test_RunExperiment):
    def test_1(self):
        with sf.test_mode():
            arguments = '--testarg --loops 3 --teststep 2 --dataset CIFAR10 --optim Adam --smoothing pytorch --sched multiplic \
                --smhardend 0.2 --smstartAt 2 --smstart 0.0001 --model vgg19_bn'.split()
            args = ex.getParser().parse_args(args=arguments)
            smoothingM = ex.createSmoothingMetadata(args)
            dataM = ex.createDataMetadata(args)
            modelM = ex.createModelMetadata(args)[0]
            modelM.prepare(lossFunc="lossFunc", optimizer="optimizer") # dummy values

            self.compareParsed(args=args, obj=smoothingM)
            self.compareParsed(args=args, obj=dataM)
            self.compareParsed(args=args, obj=modelM)

    def test_2(self):
        with sf.test_mode():
            arguments = '--testarg --loops 4 --teststep 2 --dataset CIFAR10 --optim SGD --smoothing pytorch --sched multiplic \
                --smhardend 0.2 --smstartAt 2 --smstart 0.0001 --model vgg19_bn --smgeneralmeanpow 0.05 --gamma 0.74 \
                --growthRate 3 --smlr 0.07 --drop 0.5 --epochs 888'.split()
            args = ex.getParser().parse_args(args=arguments)
            smoothingM = ex.createSmoothingMetadata(args)
            dataM = ex.createDataMetadata(args)
            modelM = ex.createModelMetadata(args)[0]
            modelM.prepare(lossFunc="lossFunc", optimizer="optimizer") # dummy values

            self.compareParsed(args=args, obj=smoothingM)
            self.compareParsed(args=args, obj=dataM)
            self.compareParsed(args=args, obj=modelM)

    def test_3(self):
        with sf.test_mode():
            arguments = '--testarg --loops 3 --teststep 2 --dataset CIFAR10 --optim Adam --smoothing pytorch --sched multiplic \
                --smhardend 0.2 --smstartAt 2 --smstart 0.0001 --model vgg19_bn'.split()
            args = ex.getParser().parse_args(args=arguments)
            smoothingM = ex.createSmoothingMetadata(args)
            dataM = ex.createDataMetadata(args)
            modelM = ex.createModelMetadata(args)[0]
            modelM.prepare(lossFunc="lossFunc", optimizer="optimizer") # dummy values

            self.compareParsed(args=args, obj=smoothingM)
            self.compareParsed(args=args, obj=dataM)
            self.compareParsed(args=args, obj=modelM)


class Test_generMean_CIFAR10_vgg(Test_RunExperiment):
    def test(self):
        with sf.test_mode():
            arguments = self.initArgs + '--dataset CIFAR10 --optim SGD --smoothing generMean --sched multiplic \
                --smhardend 0.2 --smstartAt 2 --smstart 0.0001 --model vgg19_bn --depth 28'.split()
            self.runMain(arg=arguments)


class Test_movingMean_CIFAR100_predefModel_wrn(Test_RunExperiment):
    def test(self):
        with sf.test_mode():
            arguments = self.initArgs + '--dataset CIFAR100 --optim SGD --smoothing generMean --sched multiplic \
                --smhardend 0.2 --smstartAt 2 --smstart 0.0001 --model wide_resnet --depth 28'.split()
            self.runMain(arg=arguments)

class Test_pytorch_CIFAR100_predefModel_vgg(Test_RunExperiment):
    def test(self):
        with sf.test_mode():
            arguments = self.initArgs + '--dataset CIFAR100 --optim SGD --smoothing pytorch --sched multiplic \
                --smhardend 0.2 --smstartAt 2 --smstart 0.0001 --model vgg19_bn --depth 28'.split()
            self.runMain(arg=arguments)

class Test_pytorch_CIFAR100_predefModel_densenet(Test_RunExperiment):
    def test(self):
        with sf.test_mode():
            arguments = self.initArgs + '--dataset CIFAR100 --optim SGD --smoothing pytorch --sched multiplic \
                --smhardend 0.2 --smstartAt 2 --smstart 0.0001 --model densenet --depth 34'.split()
            self.runMain(arg=arguments)

class Test_pytorch_CIFAR10_predefModel_vgg_Adam(Test_RunExperiment):
    def test(self):
        with sf.test_mode():
            arguments = self.initArgs + '--dataset CIFAR10 --optim Adam --smoothing pytorch --sched multiplic \
                --smhardend 0.2 --smstartAt 2 --smstart 0.0001 --model vgg19_bn --depth 28'.split()
            self.runMain(arg=arguments)

