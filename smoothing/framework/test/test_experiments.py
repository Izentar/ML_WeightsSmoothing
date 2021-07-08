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

"""
    Sprawdź jedynie występowanie błędów składni oraz wywołania metod / funkcji.
    Nie wykonuje żadnych porównań wartości.
"""

class Test_RunExperiment(unittest.TestCase):

    MNIST_RESIZE = (64, 64)
    
    def test_experiment_movingMean_MNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3, resizeTo=Test_RunExperiment.MNIST_RESIZE)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationEWMA_Metadata(test_movingAvgParam=0.1, test_device='cuda:0',
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10)
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
                test_howOftenPrintTrain=2, howOftenPrintTrain=3, resizeTo=Test_RunExperiment.MNIST_RESIZE)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(test_weightIter=dc.DefaultWeightDecay(1.05), test_device='cpu', 
                test_epsilon=1e-5, test_hardEpsilon=1e-7, test_weightsEpsilon=1e-6, test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, 
                test_lossContainer=20, test_lossContainerDelayedStartAt=10)
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
                test_howOftenPrintTrain=2, howOftenPrintTrain=3, resizeTo=Test_RunExperiment.MNIST_RESIZE)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingBorderline_Metadata(test_numbOfBatchAfterSwitchOn=5, test_device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingBorderline(smoothingMetadata)
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
                test_howOftenPrintTrain=2, howOftenPrintTrain=3, resizeTo=Test_RunExperiment.MNIST_RESIZE)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.wide_resnet50_2()
            smoothingMetadata = dc.Test_DefaultSmoothingBorderline_Metadata(test_numbOfBatchAfterSwitchOn=5, test_device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingBorderline(smoothingMetadata)
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
                test_howOftenPrintTrain=2, howOftenPrintTrain=3, resizeTo=Test_RunExperiment.MNIST_RESIZE)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(test_epsilon=1e-5, test_hardEpsilon=1e-6, test_weightsEpsilon=1e-5,
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10, test_device='cuda:0')
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
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, fromGrayToRGB=False,
                test_howOftenPrintTrain=2, howOftenPrintTrain=3, resizeTo=(64, 64))
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(test_epsilon=1e-5, test_hardEpsilon=1e-6, test_weightsEpsilon=1e-5,
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10, test_device='cuda:0')
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
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, fromGrayToRGB=False,
                test_howOftenPrintTrain=2, howOftenPrintTrain=3, resizeTo=Test_RunExperiment.MNIST_RESIZE)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(test_epsilon=1e-5, test_hardEpsilon=1e-6, test_weightsEpsilon=1e-5,
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10, test_device='cuda:0')
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
                test_howOftenPrintTrain=2, howOftenPrintTrain=3, resizeTo=Test_RunExperiment.MNIST_RESIZE)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            obj = models.alexnet()
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(test_epsilon=1e-5, test_hardEpsilon=1e-6, test_weightsEpsilon=1e-5,
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10, test_device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataEMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata)
            model = dc.DefaultModelPredef(obj=obj, modelMetadata=modelMetadata, name=modelName)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)

    def test_experiment_generalizedMean_MNIST_predefModel_SiplmeModel(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(test_epsilon=1e-5, test_hardEpsilon=1e-6, test_weightsEpsilon=1e-5,
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10, test_device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataMNIST(dataMetadata)
            smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata)
            model = dc.DefaultModelSimpleConv(modelMetadata=modelMetadata)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)


    def test_experiment_pytorchSWA_CIFAR10_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)
            optimizerDataDict={"learning_rate":1e-3, "momentum":0.9}

            smoothingMetadata = dc.Test_DefaultPytorchAveragedSmoothing_Metadata(test_device='cuda:0')
            modelMetadata = dc.DefaultModel_Metadata(lossFuncDataDict={}, optimizerDataDict=optimizerDataDict, device='cuda:0')

            data = dc.DefaultDataMNIST(dataMetadata)
            smoothing = dc.DefaultPytorchAveragedSmoothing(smoothingMetadata)
            model = dc.DefaultModelSimpleConv(modelMetadata=modelMetadata)

            optimizer = optim.SGD(model.getNNModelModule().parameters(), lr=optimizerDataDict['learning_rate'], 
                momentum=optimizerDataDict['momentum'])
            loss_fn = nn.CrossEntropyLoss()     

            stat=dc.run(metadataObj=metadata, data=data, model=model, smoothing=smoothing, optimizer=optimizer, lossFunc=loss_fn,
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata)
    


if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
    
