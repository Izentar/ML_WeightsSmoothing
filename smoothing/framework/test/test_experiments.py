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
    def test_experiment_movingMean_MINST_predefModel(self):
        with sf.test_mode():
            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'MINST', 'movingMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.Test_DefaultSmoothingOscilationMovingMean_Metadata(test_movingAvgParam=0.1)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)


    def test_experiment_weightedMean_MINST_predefModel(self):
        with sf.test_mode():
            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'MINST', 'weightedMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(1.05), 
                epsilon=1e-5, hardEpsilon=1e-7, weightsEpsilon=1e-6)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)

    def test_experiment_borderline_MINST_predefModel(self):
        with sf.test_mode():
            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'MINST', 'borderline')
            obj = models.alexnet()
            
            smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=5)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)


    def test_experiment_generalizedMean_MINST_predefModel(self):
        with sf.test_mode():
            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'MINST', 'generalizedMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, hardEpsilon=1e-6, weightsEpsilon=1e-5)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)

    def test_experiment_generalizedMean_CIFAR10_predefModel(self):
        with sf.test_mode():
            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, fromGrayToRGB=False,
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'CIFAR10', 'generalizedMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, hardEpsilon=1e-6, weightsEpsilon=1e-5)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)

    def test_experiment_generalizedMean_CIFAR100_predefModel(self):
        with sf.test_mode():
            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, fromGrayToRGB=False,
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'CIFAR100', 'generalizedMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, hardEpsilon=1e-6, weightsEpsilon=1e-5)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)

    def test_experiment_generalizedMean_EMNIST_predefModel(self):
        with sf.test_mode():
            metadata = sf.Metadata(testFlag=True, trainFlag=True, debugInfo=True)
            dataMetadata = dc.DefaultData_Metadata(pin_memoryTest=True, pin_memoryTrain=True, epoch=1, 
                test_howOftenPrintTrain=2, howOftenPrintTrain=3)

            types = ('predefModel', 'EMNIST', 'generalizedMean')
            obj = models.alexnet()
            
            smoothingMetadata = dc.DefaultSmoothingOscilationGeneralizedMean_Metadata(epsilon=1e-5, hardEpsilon=1e-6, weightsEpsilon=1e-5)
            modelMetadata = dc.DefaultModel_Metadata()

            stat=dc.run(numbOfRepetition=3, modelType=types[0], dataType=types[1], smoothingType=types[2], metadataObj=metadata, 
                modelMetadata=modelMetadata, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, modelPredefObj=obj)


if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
    
