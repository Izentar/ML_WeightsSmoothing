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

init_weights = {
    'linear1.weight': [[5., 5., 5.]], 
    'linear1.bias': [7.], 
    'linear2.weight': [[5.], [5.], [5.]], 
    'linear2.bias': [7., 7., 7.]
}

class Test_Data(unittest.TestCase):
    def test_updateTotalNumbLoops_testMode(self):
        with sf.test_mode():
            dataMetadata = dc.DefaultData_Metadata(epoch=7)
            data = dc.DefaultDataMNIST(dataMetadata)
            data.epochHelper = sf.EpochDataContainer()

            data._updateTotalNumbLoops(dataMetadata)

            ut.testCmpPandas(data.epochHelper.maxTrainTotalNumber, "max_loops_train", 7 * sf.StaticData.MAX_DEBUG_LOOPS * 1)
            ut.testCmpPandas(data.epochHelper.maxTestTotalNumber, "max_loops_test", 7 * sf.StaticData.MAX_DEBUG_LOOPS * 2)
        
    def test_updateTotalNumbLoops(self):
        dataMetadata = dc.DefaultData_Metadata(epoch=7)
        data = dc.DefaultDataMNIST(dataMetadata)
        data.epochHelper = sf.EpochDataContainer()

        data._updateTotalNumbLoops(dataMetadata)

        ut.testCmpPandas(data.epochHelper.maxTrainTotalNumber, "max_loops_train", 7 * 1 * len(data.trainloader))
        ut.testCmpPandas(data.epochHelper.maxTestTotalNumber, "max_loops_test", 7 * 2 * len(data.testloader))

class TestModel_Metadata(sf.Model_Metadata):
    def __init__(self):
        super().__init__()
        self.device = 'cpu:0'
        self.learning_rate = 1e-3
        self.momentum = 0.9

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Learning rate:\t{}\n'.format(self.learning_rate))
        tmp_str += ('Momentum:\t{}\n'.format(self.momentum))
        tmp_str += ('Model device :\t{}\n'.format(self.device))
        return tmp_str

class TestModel(sf.Model):
    def __init__(self, modelMetadata):
        super().__init__(modelMetadata)
        self.linear1 = nn.Linear(3, 1)
        self.linear2 = nn.Linear(1, 3)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.getNNModelModule().parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)

        self.getNNModelModule().to(modelMetadata.device)
        self.__initializeWeights__()

    def setConstWeights(self, weight, bias):
        for m in self.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.constant_(m.weight, weight)
                nn.init.constant_(m.bias, bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def __update__(self, modelMetadata):
        self.getNNModelModule().to(modelMetadata.device)
        self.optimizer = optim.SGD(self.getNNModelModule().parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)

    def __initializeWeights__(self):
        for m in self.modules():
            if(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d))):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 3)
            elif(isinstance(m, nn.Linear)):
                nn.init.constant_(m.weight, 5)
                nn.init.constant_(m.bias, 7)

class Test_DefaultSmoothing(ut.Utils):
    """
    Utility class
    """

    def checkSmoothedWeights(self, smoothing, helperEpoch, smoothingMetadata, dataMetadata, helper, model, metadata, w, b):
        weights = self.setWeightDict(w, b)
        smoothing(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata)
        self.compareDictToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata), numpyDict=weights)

    def checkOscilation__isSmoothingGoodEnough__(self, avgLoss, helperEpoch, avgKLoss, dataMetadata, smoothing, smoothingMetadata, helper, model, metadata, booleanIsGood):
        smoothing(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata)
        ut.testCmpPandas(smoothing.lossContainer.getAverage(), 'average', avgLoss)
        ut.testCmpPandas(smoothing.lossContainer.getAverage(smoothingMetadata.lossContainerDelayedStartAt), 'average', avgKLoss)
        ut.testCmpPandas(smoothing.__isSmoothingGoodEnough__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata), 'isSmoothingGoodEnough', booleanIsGood)

class Test__SmoothingOscilationBase(Test_DefaultSmoothing):
    def setUp(self):
        self.metadata = sf.Metadata()
        self.metadata.debugInfo = True
        self.metadata.debugOutput = 'debug'
        self.metadata.logFolderSuffix = str(time.time())
        self.metadata.prepareOutput()
        self.modelMetadata = TestModel_Metadata()
        self.model = TestModel(self.modelMetadata)
        self.helper = sf.TrainDataContainer()
        self.dataMetadata = dc.DefaultData_Metadata()

        self.helperEpoch = sf.EpochDataContainer()
        self.helperEpoch.trainTotalNumber = 3
        self.helperEpoch.maxTrainTotalNumber = 5

    def test__isSmoothingGoodEnough__(self):
        self.smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(epsilon=1.0, hardEpsilon=1e-9, smoothingEndCheckType='wgsum',
        weightsEpsilon=2.0, softMarginAdditionalLoops=0,
        lossContainer=3, lossContainerDelayedStartAt=1, weightsArraySize=3)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=self.smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())
        self.helper.loss = torch.Tensor([1.0])

        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=1.0, avgKLoss=0, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
            smoothing=smoothing, smoothingMetadata=self.smoothingMetadata,
            helper=self.helper, model=self.model, metadata=self.metadata, booleanIsGood=False)
        self.helper.loss = torch.Tensor([0.5])
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=1.5/2, avgKLoss=1.0, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
        smoothing=smoothing, smoothingMetadata=self.smoothingMetadata,
            helper=self.helper, model=self.model, metadata=self.metadata, booleanIsGood=False)
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=2/3, avgKLoss=1.5/2, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
        smoothing=smoothing, smoothingMetadata=self.smoothingMetadata,
            helper=self.helper, model=self.model, metadata=self.metadata, booleanIsGood=False)

        self.helper.loss = torch.Tensor([1.5])
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=2.5/3, avgKLoss=1/2, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
        smoothing=smoothing, smoothingMetadata=self.smoothingMetadata,
            helper=self.helper, model=self.model, metadata=self.metadata, booleanIsGood=False)
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=3.5/3, avgKLoss=2/2, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
        smoothing=smoothing, smoothingMetadata=self.smoothingMetadata,
            helper=self.helper, model=self.model, metadata=self.metadata, booleanIsGood=False)
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=4.5/3, avgKLoss=3/2, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
        smoothing=smoothing, smoothingMetadata=self.smoothingMetadata,
            helper=self.helper, model=self.model, metadata=self.metadata, booleanIsGood=False)

        self.helper.loss = torch.Tensor([1.3])
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=4.3/3, avgKLoss=3/2, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
        smoothing=smoothing, smoothingMetadata=self.smoothingMetadata,
            helper=self.helper, model=self.model, metadata=self.metadata, booleanIsGood=False)
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=4.1/3, avgKLoss=2.8/2, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
            smoothing=smoothing, smoothingMetadata=self.smoothingMetadata,
            helper=self.helper, model=self.model, metadata=self.metadata, booleanIsGood=True)
    
    def test__smoothingGoodEnoughCheck(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(softMarginAdditionalLoops=0, weightsEpsilon = 1.0,
            smoothingEndCheckType='wgsum')

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())
        smoothing.countWeights = 1
        a = 2.0
        b = 1.5

        ut.testCmpPandas(smoothing._smoothingGoodEnoughCheck(abs(a - b), smoothingMetadata=smoothingMetadata), 'bool', False)
        ut.testCmpPandas(smoothing._smoothingGoodEnoughCheck(abs(a - b), smoothingMetadata=smoothingMetadata), 'bool', True)
        b = 3.1
        ut.testCmpPandas(smoothing._smoothingGoodEnoughCheck(abs(a - b), smoothingMetadata=smoothingMetadata), 'bool', False)
    
    def test__sumAllWeights(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum')

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())
        smoothing.countWeights = 1

        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        sumWg = smoothing._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=self.metadata)
        ut.testCmpPandas(sumWg, 'weight_sum', 58.0)

class Test_DefaultSmoothingOscilationWeightedMean(Test_DefaultSmoothing):
    def setUp(self):
        self.metadata = sf.Metadata()
        self.metadata.debugInfo = True
        self.metadata.logFolderSuffix = str(time.time())
        self.metadata.debugOutput = 'debug'
        self.metadata.prepareOutput()
        self.modelMetadata = TestModel_Metadata()
        self.model = TestModel(self.modelMetadata)
        self.helper = sf.TrainDataContainer()
        self.dataMetadata = dc.DefaultData_Metadata()
        self.helperEpoch = sf.EpochDataContainer()
        self.helperEpoch.trainTotalNumber = 3
        self.helperEpoch.maxTrainTotalNumber = 1

    def test__getSmoothedWeights__(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(2), smoothingEndCheckType='wgsum', 
        epsilon=1.0, weightsEpsilon=1.0, hardEpsilon=1e-9, 
        softMarginAdditionalLoops=0, lossContainer=3, lossContainerDelayedStartAt=1, weightsArraySize=2)

        self.helperEpoch.maxTrainTotalNumber = 1000

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())
        self.helper.loss = torch.Tensor([1.0])

        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, 
        metadata=self.metadata, smoothingMetadata=smoothingMetadata)
        self.compareDictToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata), {})
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
        modelMetadata=None, metadata=self.metadata, smoothingMetadata=smoothingMetadata) # aby zapisać wagi
        self.compareDictToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata), init_weights)

        self.model.setConstWeights(weight=17, bias=19)
        w = (17+5/2)/1.5
        b = (19+7/2)/1.5
        self.checkSmoothedWeights(smoothing=smoothing, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, smoothingMetadata=smoothingMetadata, 
        helper=self.helper, model=self.model, metadata=self.metadata, w=w, b=b)

        self.model.setConstWeights(weight=23, bias=27)
        w = (23+17/2)/1.5
        b = (27+19/2)/1.5
        self.checkSmoothedWeights(smoothing=smoothing, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, smoothingMetadata=smoothingMetadata, 
        helper=self.helper, model=self.model, metadata=self.metadata, w=w, b=b)

        self.model.setConstWeights(weight=31, bias=37)
        w = (31+23/2)/1.5
        b = (37+27/2)/1.5
        self.checkSmoothedWeights(smoothing=smoothing, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, smoothingMetadata=smoothingMetadata, 
        helper=self.helper, model=self.model, metadata=self.metadata, w=w, b=b)

    def test_calcMean(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(2), smoothingEndCheckType='wgsum')

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())

        weights = dict(self.model.named_parameters())
        self.compareDictToNumpy(iterator=weights, numpyDict=init_weights)


        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        smoothedWg = smoothing.weightsArray.array
        weights = dict(self.model.named_parameters())
        i = smoothedWg[0]
        self.compareDictToNumpy(iterator=weights, numpyDict=init_weights)
        self.compareDictToNumpy(iterator=i, numpyDict=init_weights)

        #########

        second_weights = {
            'linear1.weight': [[11., 11., 11.]], 
            'linear1.bias': [13.], 
            'linear2.weight': [[11.], [11.], [11.]], 
            'linear2.bias': [13., 13., 13.]
        }

        self.model.setConstWeights(weight=11, bias=13)  # change model weights
        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata) 
        weights = dict(self.model.named_parameters())
        i = smoothedWg[1]
        self.compareDictToNumpy(iterator=i, numpyDict=second_weights)
        self.compareDictToNumpy(iterator=weights, numpyDict=second_weights)

    
        ########

        third_weights = {
            'linear1.weight': [[9., 9., 9.]], 
            'linear1.bias': [11.], 
            'linear2.weight': [[9.], [9.], [9.]], 
            'linear2.bias': [11., 11., 11.]
        }

        smoothing.countWeights = 2
        sm_weights = smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=None)
        weights = dict(self.model.named_parameters())
        self.compareDictToNumpy(iterator=sm_weights, numpyDict=third_weights)
        self.compareDictToNumpy(iterator=weights, numpyDict=second_weights)
 
    def test__sumWeightsToArrayStd(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(2),
        smoothingEndCheckType='std',
        epsilon=1.0, weightsEpsilon=1.0, hardEpsilon=1e-9, 
        softMarginAdditionalLoops=0, lossContainer=3, lossContainerDelayedStartAt=1, weightsArraySize=2)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())

        self.helperEpoch.epochNumber = 3
        self.dataMetadata.epoch = 4
        self.helper.loss = torch.Tensor([1.0])
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, 
        metadata=self.metadata, smoothingMetadata=smoothingMetadata)
        wg = smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata)
        std = smoothing._sumWeightsToArrayStd(wg)
        ut.testCmpPandas(std.item(), 'std', dc.ConfigClass.STD_NAN)


        second_weights = {
            'linear1.weight': [[11., 11., 11.]], 
            'linear1.bias': [13.], 
            'linear2.weight': [[11.], [11.], [11.]], 
            'linear2.bias': [13., 13., 13.]
        }

        self.model.setConstWeights(weight=11, bias=13)
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, 
        metadata=self.metadata, smoothingMetadata=smoothingMetadata)
        wg = smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata)
        std = smoothing._sumWeightsToArrayStd(wg)
        ut.testCmpPandas(std.item(), 'std', torch.std(torch.Tensor([(11 - 9.0)*6+(13 - 11.0)*4,  (9.0 - 5)*6+(11.0 - 7)*4])).item()) # smoothed weights = saved weights -> 0


class Test_DefaultSmoothingOscilationEWMA(Test_DefaultSmoothing):
    def setUp(self):
        self.metadata = sf.Metadata()
        self.metadata.debugInfo = True
        self.metadata.logFolderSuffix = str(time.time())
        self.metadata.debugOutput = 'debug'
        self.metadata.prepareOutput()
        self.modelMetadata = TestModel_Metadata()
        self.model = TestModel(self.modelMetadata)
        self.helper = sf.TrainDataContainer()
        self.dataMetadata = dc.DefaultData_Metadata()
        self.helperEpoch = sf.EpochDataContainer()
        self.helperEpoch.trainTotalNumber = 3
        self.helperEpoch.maxTrainTotalNumber = 1000

    def test_calcMean(self):
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        smoothingMetadata = dc.DefaultSmoothingOscilationEWMA_Metadata(movingAvgParam=0.5)

        smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())

        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)

        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictToNumpy(wg, init_weights)
        self.compareDictToNumpy(smoothedWg, init_weights)

        #############

        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)
        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictToNumpy(wg, init_weights)
        self.compareDictToNumpy(smoothedWg, init_weights)

        ############

        second_base_weights = {
            'linear1.weight': [[17., 17., 17.]], 
            'linear1.bias': [19.], 
            'linear2.weight': [[17.], [17.], [17.]], 
            'linear2.bias': [19., 19., 19.]
        }

        second_smth_weights = {
            'linear1.weight': [[11., 11., 11.]], 
            'linear1.bias': [13.], 
            'linear2.weight': [[11.], [11.], [11.]], 
            'linear2.bias': [13., 13., 13.]
        }

        model.setConstWeights(weight=17, bias=19)
        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)
        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictToNumpy(iterator=wg, numpyDict=second_base_weights)
        self.compareDictToNumpy(iterator=smoothedWg, numpyDict=second_smth_weights)

    def test__getSmoothedWeights__(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationEWMA_Metadata(movingAvgParam=0.5, epsilon=1.0,
        weightsEpsilon=1.0, softMarginAdditionalLoops=0, hardEpsilon=1e-9,
        lossContainer=3, lossContainerDelayedStartAt=1)

        self.helper.loss = torch.Tensor([1.0])

        smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())

        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, metadata=self.metadata, 
        smoothingMetadata=smoothingMetadata)
        self.compareDictToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata), {})
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, metadata=self.metadata, 
        smoothingMetadata=smoothingMetadata) # zapisanie wag
        self.compareDictToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata), init_weights)

        self.model.setConstWeights(weight=17, bias=19)
        w = (17/2+5/2)
        b = (19/2+7/2)
        self.checkSmoothedWeights(smoothing=smoothing, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
        smoothingMetadata=smoothingMetadata, helper=self.helper, model=self.model, metadata=self.metadata, w=w, b=b)

        self.model.setConstWeights(weight=23, bias=27)
        w = (23/2+w/2)
        b = (27/2+b/2)
        self.checkSmoothedWeights(smoothing=smoothing, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
        smoothingMetadata=smoothingMetadata, helper=self.helper, model=self.model, metadata=self.metadata, w=w, b=b)

        self.model.setConstWeights(weight=31, bias=37)
        w = (31/2+w/2)
        b = (37/2+b/2)
        self.checkSmoothedWeights(smoothing=smoothing, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, 
        smoothingMetadata=smoothingMetadata, helper=self.helper, model=self.model, metadata=self.metadata, w=w, b=b)

class Test_DefaultSmoothingSimpleMean(Test_DefaultSmoothing):
    def setUp(self):
        self.metadata = sf.Metadata()
        self.metadata.debugInfo = True
        self.metadata.logFolderSuffix = str(time.time())
        self.metadata.debugOutput = 'debug'
        self.metadata.prepareOutput()
        self.modelMetadata = TestModel_Metadata()
        self.model = TestModel(self.modelMetadata)
        self.helper = sf.TrainDataContainer()
        self.smoothingMetadata = dc.Test_DefaultSmoothingSimpleMean_Metadata(numbOfBatchAfterSwitchOn=2)
        self.dataMetadata = dc.DefaultData_Metadata()
        self.helperEpoch = sf.EpochDataContainer()
        self.helperEpoch.trainTotalNumber = 3


    def utils_checkSmoothedWeights(self, model, helperEpoch, dataMetadata, smoothing, smoothingMetadata, helper, metadata, w, b, sumW, sumB, count):
        # utils
        model.setConstWeights(weight=w, bias=b)
        w = (w+sumW)/count
        b = (b+sumB)/count
        wg = self.setWeightDict(w=w, b=b)
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, 
        metadata=self.metadata, smoothingMetadata=self.smoothingMetadata)
        self.compareDictToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=self.smoothingMetadata, metadata=self.metadata), numpyDict=wg)

    def test___isSmoothingGoodEnough__(self):
        self.helper.loss = torch.Tensor([1.0])

        smoothing = dc.DefaultSmoothingSimpleMean(smoothingMetadata=self.smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=self.smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())

        tmp = smoothing.__isSmoothingGoodEnough__(helperEpoch=None, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
        modelMetadata=None, metadata=self.metadata, smoothingMetadata=self.smoothingMetadata)
        assert tmp == False

    def test__getSmoothedWeights__(self):
        self.helper.loss = torch.Tensor([1.0])

        self.helperEpoch.trainTotalNumber = 0
        self.helperEpoch.maxTrainTotalNumber = 50000

        smoothing = dc.DefaultSmoothingSimpleMean(smoothingMetadata=self.smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=self.smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())

        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, 
        metadata=self.metadata, smoothingMetadata=self.smoothingMetadata)
        self.compareDictToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=self.smoothingMetadata, metadata=self.metadata), numpyDict={})
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, 
        metadata=self.metadata, smoothingMetadata=self.smoothingMetadata)
        self.compareDictToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=self.smoothingMetadata, metadata=self.metadata), numpyDict={})
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, 
        metadata=self.metadata, smoothingMetadata=self.smoothingMetadata) # zapisanie wag
        self.compareDictToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=self.smoothingMetadata, metadata=self.metadata), numpyDict=init_weights)

        self.utils_checkSmoothedWeights(model=self.model, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, smoothing=smoothing, 
        smoothingMetadata=self.smoothingMetadata, helper=self.helper, metadata=self.metadata, w=17, b=19, sumW=5, sumB=7, count=2)
        self.utils_checkSmoothedWeights(model=self.model, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, smoothing=smoothing, 
        smoothingMetadata=self.smoothingMetadata, helper=self.helper, metadata=self.metadata, w=23, b=29, sumW=5+17, sumB=7+19, count=3)
        self.utils_checkSmoothedWeights(model=self.model, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, smoothing=smoothing, 
        smoothingMetadata=self.smoothingMetadata, helper=self.helper, metadata=self.metadata, w=45, b=85, sumW=5+17+23, sumB=7+19+29, count=4) 

def run():
    inst = Test_DefaultSmoothingOscilationWeightedMean()
    inst.test__sumWeightsToArrayStd()

if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
    
