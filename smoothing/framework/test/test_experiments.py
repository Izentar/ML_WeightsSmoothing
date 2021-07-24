from framework import defaultClasses as dc
from framework import smoothingFramework as sf
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

"""
    Sprawdź jedynie występowanie błędów składni oraz wywołania metod / funkcji.
    Nie wykonuje żadnych porównań wartości.
"""

class Test_RunExperiment(unittest.TestCase):
    RESIZE_TO = 64

    def setUp(self):
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
    
    def test_experiment_movingMean_MNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                howOftenPrintTrain=2, transformTrain=self.trainTrans, transformTest=self.testTrans)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationEWMA_Metadata(movingAvgParam=0.1, device='cuda:0',
                weightSumContainerSize=3, weightSumContainerSizeStartAt=1, lossContainer=20, lossContainerDelayedStartAt=10)
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
    
    
    def test_experiment_weightedMean_MNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                howOftenPrintTrain=2, transformTrain=self.trainTrans, transformTest=self.testTrans)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(1.05), device='cpu', 
                epsilon=1e-5, hardEpsilon=1e-7, weightsEpsilon=1e-6, weightSumContainerSize=3, weightSumContainerSizeStartAt=1, 
                lossContainer=20, lossContainerDelayedStartAt=10)
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
    
    def test_experiment_borderline_MNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                howOftenPrintTrain=2, transformTrain=self.trainTrans, transformTest=self.testTrans)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingSimpleMean_Metadata(numbOfBatchAfterSwitchOn=5, device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingSimpleMean(smoothingMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)

    def test_experiment_borderline_MNIST_predefModel_wide_resnet(self):
        with sf.test_mode():
            modelName = "wide_resnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                howOftenPrintTrain=2, transformTrain=self.trainTrans, transformTest=self.testTrans)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.wide_resnet50_2()
            smoothingMetadata = dc.Test_DefaultSmoothingSimpleMean_Metadata(numbOfBatchAfterSwitchOn=5, device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingSimpleMean(smoothingMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
    
    def test_experiment_generalizedMean_MNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                howOftenPrintTrain=2, transformTrain=self.trainTrans, transformTest=self.testTrans)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, hardEpsilon=1e-6, weightsEpsilon=1e-5,
                weightSumContainerSize=3, weightSumContainerSizeStartAt=1, lossContainer=20, lossContainerDelayedStartAt=10, device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
    
    def test_experiment_generalizedMean_CIFAR10_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1,
                howOftenPrintTrain=2, transformTrain=self.CIFAR10_trainTrans, transformTest=self.CIFAR10_testTrans)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, hardEpsilon=1e-6, weightsEpsilon=1e-5,
                weightSumContainerSize=3, weightSumContainerSizeStartAt=1, lossContainer=20, lossContainerDelayedStartAt=10, device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataCIFAR10(dataMetadata)
            smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
    
    def test_experiment_generalizedMean_CIFAR100_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1,
                howOftenPrintTrain=2, transformTrain=self.CIFAR100_trainTrans, transformTest=self.CIFAR100_testTrans)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, hardEpsilon=1e-6, weightsEpsilon=1e-5,
                weightSumContainerSize=3, weightSumContainerSizeStartAt=1, lossContainer=20, lossContainerDelayedStartAt=10, device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataCIFAR100(dataMetadata)
            smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)

    def test_experiment_generalizedMean_EMNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                howOftenPrintTrain=2, transformTrain=self.trainTrans, transformTest=self.testTrans)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, hardEpsilon=1e-6, weightsEpsilon=1e-5,
                weightSumContainerSize=3, weightSumContainerSizeStartAt=1, lossContainer=20, lossContainerDelayedStartAt=10, device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataEMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
    
    def test_experiment_generalizedMean_MNIST_SimpleModel(self):
        with sf.test_mode():
            modelName = "alexnet"

            self.trainTrans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.repeat(3, 1, 1)),
            self.MNIST_normalize
            ])
            self.testTrans = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Lambda(lambda x : x.repeat(3, 1, 1)),
                self.MNIST_normalize
            ])

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                howOftenPrintTrain=2, transformTrain=self.trainTrans, transformTest=self.testTrans)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, hardEpsilon=1e-6, weightsEpsilon=1e-5,
                weightSumContainerSize=3, weightSumContainerSizeStartAt=1, lossContainer=20, lossContainerDelayedStartAt=10, device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata)
            model = dc.DefaultModelSimpleConv(modelMetadata=modelMetadata)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
    
    def test_experiment_pytorchSWA_CIFAR10_SimpleModel(self):
        with sf.test_mode():
            modelName = "simpleConv"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                howOftenPrintTrain=2, transformTrain=self.CIFAR10_trainTrans, transformTest=self.CIFAR10_testTrans)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            smoothingMetadata = dc.Test_DefaultPytorchAveragedSmoothing_Metadata(device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataCIFAR10(dataMetadata)
            model = dc.DefaultModelSimpleConv(modelMetadata=modelMetadata)
            smoothing = dc.DefaultPytorchAveragedSmoothing(smoothingMetadata, model=model)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
    


if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
    
