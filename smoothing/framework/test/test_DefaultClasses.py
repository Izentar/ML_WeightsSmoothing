from framework import defaultClasses as dc
from framework import smoothingFramework as sf
import pandas as pd
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sf.StaticData.LOG_FOLDER = './smoothing/framework/test/dump/'

init_weights = {
    'linear1.weight': [[5., 5., 5.]], 
    'linear1.bias': [7.], 
    'linear2.weight': [[5.], [5.], [5.]], 
    'linear2.bias': [7., 7., 7.]
}

def testCmpPandas(obj_1, name_1, obj_2, name_2 = None):
    if(name_2 is None):
        pd.testing.assert_frame_equal(pd.DataFrame([{name_1: obj_1}]), pd.DataFrame([{name_1: obj_2}]))
    else:
        pd.testing.assert_frame_equal(pd.DataFrame([{name_1: obj_1}]), pd.DataFrame([{name_2: obj_2}]))

class Test_CircularList(unittest.TestCase):

    def test_pushBack(self):
        inst = dc.CircularList(2)
        inst.pushBack(1)
        testCmpPandas(inst.array[0], 'array_value', 1)
        inst.pushBack(2)
        testCmpPandas(inst.array[0], 'array_value', 1)
        testCmpPandas(inst.array[1], 'array_value', 2)
        inst.pushBack(3)
        testCmpPandas(inst.array[0], 'array_value', 3)
        testCmpPandas(inst.array[1], 'array_value', 2)
        inst.pushBack(4)
        testCmpPandas(inst.array[0], 'array_value', 3)
        testCmpPandas(inst.array[1], 'array_value', 4)

        inst.reset()
        testCmpPandas(len(inst.array), 'array_length', 0)
        inst.pushBack(10)
        testCmpPandas(inst.array[0], 'array_value', 10)
        testCmpPandas(len(inst.array), 'array_length', 1)

    def test_getAverage(self):
        inst = dc.CircularList(3)
        testCmpPandas(inst.getAverage(), 'average', 0)
        inst.pushBack(1)
        testCmpPandas(inst.getAverage(), 'average', 1.0)
        inst.pushBack(2)
        testCmpPandas(inst.getAverage(), 'average', 1.5)
        inst.pushBack(3)
        testCmpPandas(inst.getAverage(), 'average', 2.0)
        testCmpPandas(inst.getAverage(startAt=1), 'average', 1.5)
        inst.pushBack(4)
        testCmpPandas(inst.getAverage(), 'average', 3.0)
        inst.pushBack(5)
        testCmpPandas(inst.getAverage(), 'average', 4.0)

        testCmpPandas(inst.getAverage(startAt=1), 'average', 3.5)
        testCmpPandas(inst.getAverage(startAt=2), 'average', 3.0)

        inst.reset()
        testCmpPandas(len(inst.array), 'array_length', 0)
        inst.pushBack(10)
        testCmpPandas(inst.getAverage(), 'average', 10.0)
        testCmpPandas(len(inst.array), 'array_length', 1)

        testCmpPandas(inst.getAverage(startAt=1), 'average', 0)

    def test_iteration(self):
        inst = dc.CircularList(3)
        inst.pushBack(1)
        inst.pushBack(2)
        inst.pushBack(3)

        i = iter(inst)
        testCmpPandas(i.indexArray, 'array', [2, 1, 0])

        testCmpPandas(next(i), 'iter_next', 3)
        testCmpPandas(next(i), 'iter_next', 2)
        testCmpPandas(next(i), 'iter_next', 1)

        self.assertRaises(StopIteration, lambda : next(i))

        inst.pushBack(4)
        i = iter(inst)
        testCmpPandas(next(i), 'iter_next', 4)
        testCmpPandas(next(i), 'iter_next', 3)
        testCmpPandas(next(i), 'iter_next', 2)
        self.assertRaises(StopIteration, lambda : next(i))

        inst.pushBack(5)
        i = iter(inst)
        testCmpPandas(next(i), 'iter_next', 5)
        testCmpPandas(next(i), 'iter_next', 4)
        testCmpPandas(next(i), 'iter_next', 3)
        self.assertRaises(StopIteration, lambda : next(i))
        
    def test_len(self):
        inst = dc.CircularList(3)
        testCmpPandas(len(inst), 'array_length', 0)
        inst.pushBack(1)
        testCmpPandas(len(inst), 'array_length', 1)
        inst.pushBack(2)
        testCmpPandas(len(inst), 'array_length', 2)
        inst.pushBack(3)
        testCmpPandas(len(inst), 'array_length', 3)
        inst.pushBack(4)
        testCmpPandas(len(inst), 'array_length', 3)
        inst.pushBack(5)
        testCmpPandas(len(inst), 'array_length', 3)

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

class Test_DefaultSmoothing(unittest.TestCase):
    """
    Utility class
    """
    def compareArraysTensorNumpy(self, iterator, numpyArray: list):
        if(len(numpyArray) == 0):
            test = False
            for ar in iterator:
                test = True
            if(test):
                self.fail("Comparing empty to non-empty array. Iter:\n{}\n:Dict:\n{}".format(iterator, numpyArray)) 
        idx = 0
        for ar in iterator:
            if(idx >= len(numpyArray)):
                self.fail("Arrays size not equals.") 
            ar = ar.detach().numpy()
            testCmpPandas(ar, 'array', numpyArray[idx])
            idx += 1

    def compareDictTensorToNumpy(self, iterator, numpyDict: dict):
        if(len(numpyDict) == 0):
            test = False
            for ar in iterator:
                test = True
            if(test):
                self.fail("Comparing empty to non-empty dicttionary. Iter:\n{}\n:Dict:\n{}".format(iterator, numpyDict)) 

        idx = 0
        for key, ar in iterator.items():
            if(idx >= len(numpyDict)):
                self.fail("Arrays size not equals.") 
            if(key not in numpyDict):
                self.fail("Dictionary key not found.") 
            ar = ar.detach().numpy()
            testCmpPandas(ar, 'array', numpyDict[key])
            idx += 1
            
    def setWeightDict(self, w, b):
        return {
            'linear1.weight': [[w, w, w]], 
            'linear1.bias': [b], 
            'linear2.weight': [[w], [w], [w]], 
            'linear2.bias': [b, b, b]
        }

    def checkSmoothedWeights(self, smoothing, smoothingMetadata, helper, model, metadata, w, b):
        weights = self.setWeightDict(w, b)
        smoothing(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata)
        self.compareDictTensorToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata), numpyDict=weights)

    def checkOscilation__isSmoothingGoodEnough__(self, avgLoss, avgKLoss, smoothing, smoothingMetadata, helper, model, metadata, booleanIsGood):
        smoothing(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', avgLoss)
        testCmpPandas(smoothing.lossContainer.getAverage(smoothingMetadata.lossContainerDelayedStartAt), 'average', avgKLoss)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata), 'isSmoothingGoodEnough', booleanIsGood)

class Test__SmoothingOscilationBase(Test_DefaultSmoothing):
    def test__isSmoothingGoodEnough__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_1'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(epsilon=1.0, hardEpsilon=1e-9, 
        weightsEpsilon=1.0, numbOfBatchMinStart=1, softMarginAdditionalLoops=1,
        lossContainer=3, lossContainerDelayedStartAt=1, weightsArraySize=3)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())
        helper.loss = torch.Tensor([1.0])

        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=1.0, avgKLoss=0, smoothing=smoothing, smoothingMetadata=smoothingMetadata,
            helper=helper, model=model, metadata=metadata, booleanIsGood=False)
        helper.loss = torch.Tensor([0.5])
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=1.5/2, avgKLoss=1.0, smoothing=smoothing, smoothingMetadata=smoothingMetadata,
            helper=helper, model=model, metadata=metadata, booleanIsGood=False)
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=2/3, avgKLoss=1.5/2, smoothing=smoothing, smoothingMetadata=smoothingMetadata,
            helper=helper, model=model, metadata=metadata, booleanIsGood=False)

        helper.loss = torch.Tensor([1.5])
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=2.5/3, avgKLoss=1/2, smoothing=smoothing, smoothingMetadata=smoothingMetadata,
            helper=helper, model=model, metadata=metadata, booleanIsGood=False)
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=3.5/3, avgKLoss=2/2, smoothing=smoothing, smoothingMetadata=smoothingMetadata,
            helper=helper, model=model, metadata=metadata, booleanIsGood=False)
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=4.5/3, avgKLoss=3/2, smoothing=smoothing, smoothingMetadata=smoothingMetadata,
            helper=helper, model=model, metadata=metadata, booleanIsGood=False)

        helper.loss = torch.Tensor([1.3])
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=4.3/3, avgKLoss=3/2, smoothing=smoothing, smoothingMetadata=smoothingMetadata,
            helper=helper, model=model, metadata=metadata, booleanIsGood=False)
        self.checkOscilation__isSmoothingGoodEnough__(avgLoss=4.1/3, avgKLoss=2.8/2, smoothing=smoothing, smoothingMetadata=smoothingMetadata,
            helper=helper, model=model, metadata=metadata, booleanIsGood=True)
    
    def test__smoothingGoodEnoughCheck(self):
        metadata = sf.Metadata()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(softMarginAdditionalLoops=1, weightsEpsilon = 1.0)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())
        smoothing.countWeights = 1
        a = 2.0
        b = 1.5

        testCmpPandas(smoothing._smoothingGoodEnoughCheck(a, b, smoothingMetadata=smoothingMetadata), 'bool', False)
        testCmpPandas(smoothing._smoothingGoodEnoughCheck(a, b, smoothingMetadata=smoothingMetadata), 'bool', True)
        testCmpPandas(smoothing._smoothingGoodEnoughCheck(b, a, smoothingMetadata=smoothingMetadata), 'bool', True)
        b = 3.1
        testCmpPandas(smoothing._smoothingGoodEnoughCheck(a, b, smoothingMetadata=smoothingMetadata), 'bool', False)

    def test__sumAllWeights(self):
        metadata = sf.Metadata()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata()

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())
        smoothing.countWeights = 1

        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)
        sumWg = smoothing._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=metadata)
        testCmpPandas(sumWg, 'weight_sum', 58.0)

class Test_DefaultSmoothingOscilationWeightedMean(Test_DefaultSmoothing):
    def test__getSmoothedWeights__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_3'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(2), 
        epsilon=1.0, weightsEpsilon=1.0, numbOfBatchMinStart=1, hardEpsilon=1e-9, 
        softMarginAdditionalLoops=1, lossContainer=3, lossContainerDelayedStartAt=1, weightsArraySize=2)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())
        helper.loss = torch.Tensor([1.0])

        smoothing(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata)
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata), {})
        smoothing(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata) # aby zapisać wagi
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata), init_weights)

        model.setConstWeights(weight=17, bias=19)
        w = (17+5/2)/1.5
        b = (19+7/2)/1.5
        self.checkSmoothedWeights(smoothing=smoothing, smoothingMetadata=smoothingMetadata, helper=helper, model=model, metadata=metadata, w=w, b=b)

        model.setConstWeights(weight=23, bias=27)
        w = (23+17/2)/1.5
        b = (27+19/2)/1.5
        self.checkSmoothedWeights(smoothing=smoothing, smoothingMetadata=smoothingMetadata, helper=helper, model=model, metadata=metadata, w=w, b=b)

        model.setConstWeights(weight=31, bias=37)
        w = (31+23/2)/1.5
        b = (37+27/2)/1.5
        self.checkSmoothedWeights(smoothing=smoothing, smoothingMetadata=smoothingMetadata, helper=helper, model=model, metadata=metadata, w=w, b=b)

    def test_calcMean(self):
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        smoothingMetadata = dc.DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(2))

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())

        weights = dict(model.named_parameters())
        self.compareDictTensorToNumpy(iterator=weights, numpyDict=init_weights)


        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)
        smoothedWg = smoothing.weightsArray.array
        weights = dict(model.named_parameters())
        i = smoothedWg[0]
        self.compareDictTensorToNumpy(iterator=weights, numpyDict=init_weights)
        self.compareDictTensorToNumpy(iterator=i, numpyDict=init_weights)

        #########

        second_weights = {
            'linear1.weight': [[11., 11., 11.]], 
            'linear1.bias': [13.], 
            'linear2.weight': [[11.], [11.], [11.]], 
            'linear2.bias': [13., 13., 13.]
        }

        model.setConstWeights(weight=11, bias=13)  # change model weights
        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata) 
        weights = dict(model.named_parameters())
        i = smoothedWg[1]
        self.compareDictTensorToNumpy(iterator=i, numpyDict=second_weights)
        self.compareDictTensorToNumpy(iterator=weights, numpyDict=second_weights)

    
        ########

        third_weights = {
            'linear1.weight': [[9., 9., 9.]], 
            'linear1.bias': [11.], 
            'linear2.weight': [[9.], [9.], [9.]], 
            'linear2.bias': [11., 11., 11.]
        }

        smoothing.countWeights = 2
        sm_weights = smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=None)
        weights = dict(model.named_parameters())
        self.compareDictTensorToNumpy(iterator=sm_weights, numpyDict=third_weights)
        self.compareDictTensorToNumpy(iterator=weights, numpyDict=second_weights)
 
class Test_DefaultSmoothingOscilationMovingMean(Test_DefaultSmoothing):
    def test_calcMean(self):
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        smoothingMetadata = dc.DefaultSmoothingOscilationMovingMean_Metadata(movingAvgParam=0.5)

        smoothing = dc.DefaultSmoothingOscilationMovingMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())

        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)

        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictTensorToNumpy(wg, init_weights)
        self.compareDictTensorToNumpy(smoothedWg, init_weights)

        #############

        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)
        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictTensorToNumpy(wg, init_weights)
        self.compareDictTensorToNumpy(smoothedWg, init_weights)

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
        self.compareDictTensorToNumpy(iterator=wg, numpyDict=second_base_weights)
        self.compareDictTensorToNumpy(iterator=smoothedWg, numpyDict=second_smth_weights)

    def test__getSmoothedWeights__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_4'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()
        smoothingMetadata = dc.DefaultSmoothingOscilationMovingMean_Metadata(movingAvgParam=0.5, epsilon=1.0,
        weightsEpsilon=1.0, numbOfBatchMinStart=1, softMarginAdditionalLoops=1, hardEpsilon=1e-9,
        lossContainer=3, lossContainerDelayedStartAt=1)

        helper.loss = torch.Tensor([1.0])

        smoothing = dc.DefaultSmoothingOscilationMovingMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())

        smoothing(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata)
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata), {})
        smoothing(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata) # zapisanie wag
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata), init_weights)

        model.setConstWeights(weight=17, bias=19)
        w = (17/2+5/2)
        b = (19/2+7/2)
        self.checkSmoothedWeights(smoothing=smoothing, smoothingMetadata=smoothingMetadata, helper=helper, model=model, metadata=metadata, w=w, b=b)

        model.setConstWeights(weight=23, bias=27)
        w = (23/2+w/2)
        b = (27/2+b/2)
        self.checkSmoothedWeights(smoothing=smoothing, smoothingMetadata=smoothingMetadata, helper=helper, model=model, metadata=metadata, w=w, b=b)

        model.setConstWeights(weight=31, bias=37)
        w = (31/2+w/2)
        b = (37/2+b/2)
        self.checkSmoothedWeights(smoothing=smoothing, smoothingMetadata=smoothingMetadata, helper=helper, model=model, metadata=metadata, w=w, b=b)

class Test_DefaultSmoothingBorderline(Test_DefaultSmoothing):
    def utils_checkSmoothedWeights(self, model, smoothing, smoothingMetadata, helper, metadata, w, b, sumW, sumB, count):
        # utils
        model.setConstWeights(weight=w, bias=b)
        w = (w+sumW)/count
        b = (b+sumB)/count
        wg = self.setWeightDict(w=w, b=b)
        smoothing(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata)
        self.compareDictTensorToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata), numpyDict=wg)

    def test___isSmoothingGoodEnough__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_5'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()
        smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=2)

        helper.loss = torch.Tensor([1.0])

        smoothing = dc.DefaultSmoothingBorderline(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())

        tmp = smoothing.__isSmoothingGoodEnough__(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata)
        assert tmp == False

    def test__getSmoothedWeights__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_6'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()
        smoothingMetadata = dc.DefaultSmoothingBorderline_Metadata(numbOfBatchAfterSwitchOn=2)

        helper.loss = torch.Tensor([1.0])

        smoothing = dc.DefaultSmoothingBorderline(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())

        smoothing(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata)
        self.compareDictTensorToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata), numpyDict={})
        smoothing(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata)
        self.compareDictTensorToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata), numpyDict={})
        smoothing(helperEpoch=None, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata) # zapisanie wag
        self.compareDictTensorToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata), numpyDict=init_weights)

        self.utils_checkSmoothedWeights(model=model, smoothing=smoothing, smoothingMetadata=smoothingMetadata, helper=helper, metadata=metadata, w=17, b=19, sumW=5, sumB=7, count=2)
        self.utils_checkSmoothedWeights(model=model, smoothing=smoothing, smoothingMetadata=smoothingMetadata, helper=helper, metadata=metadata, w=23, b=29, sumW=5+17, sumB=7+19, count=3)
        self.utils_checkSmoothedWeights(model=model, smoothing=smoothing, smoothingMetadata=smoothingMetadata, helper=helper, metadata=metadata, w=45, b=85, sumW=5+17+23, sumB=7+19+29, count=4) 


if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
