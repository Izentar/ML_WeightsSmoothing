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
    def test_experiment_movingMean_MNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'MNIST', 'movingMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationMovingMean_Metadata(test_movingAvgParam=0.1, 
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelPredefObjName=modelName, modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)


    def test_experiment_weightedMean_MNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'MNIST', 'weightedMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(test_weightIter=dc.DefaultWeightDecay(1.05), 
                test_epsilon=1e-5, test_hardEpsilon=1e-7, test_weightsEpsilon=1e-6, test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, 
                test_lossContainer=20, test_lossContainerDelayedStartAt=10)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelPredefObjName=modelName, modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)

    def test_experiment_borderline_MNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'MNIST', 'borderline')
            obj = models.alexnet()
            
            smoothingMetadata = dc.Test_DefaultSmoothingBorderline_Metadata(test_numbOfBatchAfterSwitchOn=5)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelPredefObjName=modelName, modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)

    def test_experiment_borderline_MNIST_predefModel_wide_resnet(self):
        with sf.test_mode():
            modelName = "wide_resnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'MNIST', 'borderline')
            obj = models.wide_resnet50_2()
            
            smoothingMetadata = dc.Test_DefaultSmoothingBorderline_Metadata(test_numbOfBatchAfterSwitchOn=5)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelPredefObjName=modelName, modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)


    def test_experiment_generalizedMean_MNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'MNIST', 'generalizedMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(test_epsilon=1e-5, test_hardEpsilon=1e-6, test_weightsEpsilon=1e-5,
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelPredefObjName=modelName, modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)

    def test_experiment_generalizedMean_CIFAR10_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, fromGrayToRGB=False,
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'CIFAR10', 'generalizedMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(test_epsilon=1e-5, test_hardEpsilon=1e-6, test_weightsEpsilon=1e-5,
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelPredefObjName=modelName, modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)

    def test_experiment_generalizedMean_CIFAR100_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, fromGrayToRGB=False,
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'CIFAR100', 'generalizedMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(test_epsilon=1e-5, test_hardEpsilon=1e-6, test_weightsEpsilon=1e-5,
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelPredefObjName=modelName, modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)

    def test_experiment_generalizedMean_EMNIST_predefModel_alexnet(self):
        with sf.test_mode():
            modelName = "alexnet"

            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'EMNIST', 'generalizedMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(test_epsilon=1e-5, test_hardEpsilon=1e-6, test_weightsEpsilon=1e-5,
                test_weightSumContainerSize=3, test_weightSumContainerSizeStartAt=1, test_lossContainer=20, test_lossContainerDelayedStartAt=10)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelPredefObjName=modelName, modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)


if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
    
