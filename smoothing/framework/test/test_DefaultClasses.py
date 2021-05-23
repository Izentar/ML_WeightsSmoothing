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
        inst.pushBack(4)
        testCmpPandas(inst.getAverage(), 'average', 3.0)
        inst.pushBack(5)
        testCmpPandas(inst.getAverage(), 'average', 4.0)

        inst.reset()
        testCmpPandas(len(inst.array), 'array_length', 0)
        inst.pushBack(10)
        testCmpPandas(inst.getAverage(), 'average', 10.0)
        testCmpPandas(len(inst.array), 'array_length', 1)

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
                self.fail("Comparing empty to non-empty array.") 
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
                self.fail("Comparing empty to non-empty dicttionary.") 

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

    def checkSmoothedWeights(self, smoothing, helper, model, metadata, w, b):
        weights = self.setWeightDict(w, b)
        for _ in range(smoothing.avgOfAvgUpdateFreq): # aby przejść przez obszar w którym nie wykonuje się aktualizacji wag wygładzania
            smoothing(None, helper, model, None, None, metadata) 
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), weights)

    def checkOscilation__isSmoothingGoodEnough__(self, avgLoss, avgKLoss, smoothing, helper, model, metadata, booleanIsGood):
        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', avgLoss)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', avgKLoss)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', booleanIsGood)

class Test__SmoothingOscilationBase(Test_DefaultSmoothing):

    def test__saveAvgSum(self):
        metadata = sf.Metadata()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean()
        smoothing.__setDictionary__(model.getNNModelModule().named_parameters())

        sumAvg = 5
        smoothing.divisionCounter += 1
        a, b = smoothing._saveAvgSum(sumAvg)
        testCmpPandas(a, 'weight_sum', 0)
        testCmpPandas(b, 'weight_sum', 5.0)
        sumAvg = 7
        smoothing.divisionCounter += 1
        a, b = smoothing._saveAvgSum(sumAvg)
        testCmpPandas(a, 'weight_sum', 7.0)
        testCmpPandas(b, 'weight_sum', 5.0)
        sumAvg = 11
        smoothing.divisionCounter += 1
        a, b = smoothing._saveAvgSum(sumAvg)
        testCmpPandas(a, 'weight_sum', 7.0)
        testCmpPandas(b, 'weight_sum', 8.0)
        sumAvg = 13
        smoothing.divisionCounter += 1
        a, b = smoothing._saveAvgSum(sumAvg)
        testCmpPandas(a, 'weight_sum', 10.0)
        testCmpPandas(b, 'weight_sum', 8.0)

    def test__isSmoothingGoodEnough__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_1'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(epsilon=1.0, avgOfAvgUpdateFreq=2, whenCheckCanComputeWeights=1,
        weightsEpsilon=1.0, numbOfBatchMinStart=1, endSmoothingFreq=2, softMarginAdditionalLoops=1,
        lossContainer=3, lastKLossAverage=2, weightsArraySize=3)
        smoothing.__setDictionary__(model.getNNModelModule().named_parameters())
        helper.loss = torch.Tensor([1.0])

        self.checkOscilation__isSmoothingGoodEnough__(1.0, 0, smoothing, helper, model, metadata, False)
        helper.loss = torch.Tensor([0.5])
        self.checkOscilation__isSmoothingGoodEnough__(1.5/2, 0.5, smoothing, helper, model, metadata, False)
        self.checkOscilation__isSmoothingGoodEnough__(2/3, 0.5, smoothing, helper, model, metadata, False)

        helper.loss = torch.Tensor([1.5])
        self.checkOscilation__isSmoothingGoodEnough__(2.5/3, 2/2, smoothing, helper, model, metadata, False)
        self.checkOscilation__isSmoothingGoodEnough__(3.5/3, 2/2, smoothing, helper, model, metadata, False)
        self.checkOscilation__isSmoothingGoodEnough__(4.5/3, 3/2, smoothing, helper, model, metadata, False)

        helper.loss = torch.Tensor([1.3])
        self.checkOscilation__isSmoothingGoodEnough__(4.3/3, 3/2, smoothing, helper, model, metadata, False)
        self.checkOscilation__isSmoothingGoodEnough__(4.1/3, 2.8/2, smoothing, helper, model, metadata, True)
    
    def test__smoothingGoodEnoughCheck(self):
        metadata = sf.Metadata()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(softMarginAdditionalLoops=1, weightsEpsilon = 1.0)
        smoothing.__setDictionary__(model.getNNModelModule().named_parameters())
        smoothing.countWeights = 1
        a = 2.0
        b = 1.5

        testCmpPandas(smoothing._smoothingGoodEnoughCheck(a, b), 'bool', False)
        testCmpPandas(smoothing._smoothingGoodEnoughCheck(a, b), 'bool', True)
        testCmpPandas(smoothing._smoothingGoodEnoughCheck(b, a), 'bool', True)
        b = 3.1
        testCmpPandas(smoothing._smoothingGoodEnoughCheck(a, b), 'bool', False)

    def test__sumAllWeights(self):
        metadata = sf.Metadata()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean()
        smoothing.__setDictionary__(model.getNNModelModule().named_parameters())
        smoothing.countWeights = 1

        smoothing.calcMean(model)
        sumWg = smoothing._sumAllWeights(metadata)
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

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(weightDecay=2, epsilon=1.0, avgOfAvgUpdateFreq=2, whenCheckCanComputeWeights=1,
        weightsEpsilon=1.0, numbOfBatchMinStart=1, endSmoothingFreq=2, softMarginAdditionalLoops=1,
        lossContainer=3, lastKLossAverage=2, weightsArraySize=2)
        smoothing.__setDictionary__(model.getNNModelModule().named_parameters())
        helper.loss = torch.Tensor([1.0])

        smoothing(None, helper, model, None, None, metadata)
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), {})
        smoothing(None, helper, model, None, None, metadata) 
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), {})
        smoothing(None, helper, model, None, None, metadata) # aby zapisać wagi
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), init_weights)

        model.setConstWeights(17, 19)
        w = (17+5/2)/1.5
        b = (19+7/2)/1.5
        self.checkSmoothedWeights(smoothing, helper, model, metadata, w, b)

        model.setConstWeights(23, 27)
        w = (23+17/2)/1.5
        b = (27+19/2)/1.5
        self.checkSmoothedWeights(smoothing, helper, model, metadata, w, b)

        model.setConstWeights(31, 37)
        w = (31+23/2)/1.5
        b = (37+27/2)/1.5
        self.checkSmoothedWeights(smoothing, helper, model, metadata, w, b)

    def test_calcMean(self):
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(weightDecay=2)
        smoothing.__setDictionary__(model.getNNModelModule().named_parameters())

        weights = dict(model.named_parameters())
        self.compareDictTensorToNumpy(weights, init_weights)


        smoothing.calcMean(model)
        smoothedWg = smoothing.weightsArray.array
        weights = dict(model.named_parameters())
        i = smoothedWg[0]
        self.compareDictTensorToNumpy(weights, init_weights)
        self.compareDictTensorToNumpy(i, init_weights)

        #########

        second_weights = {
            'linear1.weight': [[11., 11., 11.]], 
            'linear1.bias': [13.], 
            'linear2.weight': [[11.], [11.], [11.]], 
            'linear2.bias': [13., 13., 13.]
        }

        model.setConstWeights(11, 13)  # change model weights
        smoothing.calcMean(model) 
        weights = dict(model.named_parameters())
        i = smoothedWg[1]
        self.compareDictTensorToNumpy(i, second_weights)
        self.compareDictTensorToNumpy(weights, second_weights)

    
        ########

        third_weights = {
            'linear1.weight': [[9., 9., 9.]], 
            'linear1.bias': [11.], 
            'linear2.weight': [[9.], [9.], [9.]], 
            'linear2.bias': [11., 11., 11.]
        }

        smoothing.countWeights = 2
        sm_weights = smoothing.__getSmoothedWeights__(None)
        weights = dict(model.named_parameters())
        self.compareDictTensorToNumpy(sm_weights, third_weights)
        self.compareDictTensorToNumpy(weights, second_weights)
 
class Test_DefaultSmoothingOscilationMovingMean(Test_DefaultSmoothing):
    def test_calcMean(self):
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationMovingMean(movingAvgParam=0.5)
        smoothing.__setDictionary__(model.getNNModelModule().named_parameters())

        smoothing.calcMean(model)

        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictTensorToNumpy(wg, init_weights)
        self.compareDictTensorToNumpy(smoothedWg, init_weights)

        #############

        smoothing.calcMean(model)
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

        model.setConstWeights(17, 19)
        smoothing.calcMean(model)
        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictTensorToNumpy(wg, second_base_weights)
        self.compareDictTensorToNumpy(smoothedWg, second_smth_weights)

    def test__getSmoothedWeights__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_4'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()

        helper.loss = torch.Tensor([1.0])

        smoothing = dc.DefaultSmoothingOscilationMovingMean(movingAvgParam=0.5, epsilon=1.0, avgOfAvgUpdateFreq=2, whenCheckCanComputeWeights=1,
        weightsEpsilon=1.0, numbOfBatchMinStart=1, endSmoothingFreq=2, softMarginAdditionalLoops=1,
        lossContainer=3, lastKLossAverage=2)
        smoothing.__setDictionary__(model.getNNModelModule().named_parameters())

        smoothing(None, helper, model, None, None, metadata)
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), {})
        smoothing(None, helper, model, None, None, metadata)
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), {})
        smoothing(None, helper, model, None, None, metadata) # zapisanie wag
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), init_weights)

        model.setConstWeights(17, 19)
        w = (17/2+5/2)
        b = (19/2+7/2)
        self.checkSmoothedWeights(smoothing, helper, model, metadata, w, b)

        model.setConstWeights(23, 27)
        w = (23/2+w/2)
        b = (27/2+b/2)
        self.checkSmoothedWeights(smoothing, helper, model, metadata, w, b)

        model.setConstWeights(31, 37)
        w = (31/2+w/2)
        b = (37/2+b/2)
        self.checkSmoothedWeights(smoothing, helper, model, metadata, w, b)

class Test_DefaultSmoothingBorderline(Test_DefaultSmoothing):
    def utils_checkSmoothedWeights(self, model, smoothing, helper, metadata, w, b, sumW, sumB, count):
        # utils
        model.setConstWeights(w, b)
        w = (w+sumW)/count
        b = (b+sumB)/count
        wg = self.setWeightDict(w, b)
        smoothing(None, helper, model, None, None, metadata)
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), wg)

    def test___isSmoothingGoodEnough__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_5'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()

        helper.loss = torch.Tensor([1.0])

        smoothing = dc.DefaultSmoothingBorderline(numbOfBatchAfterSwitchOn=2)
        smoothing.__setDictionary__(model.getNNModelModule().named_parameters())
        assert smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata) == False

    def test__getSmoothedWeights__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_6'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()

        helper.loss = torch.Tensor([1.0])

        smoothing = dc.DefaultSmoothingBorderline(numbOfBatchAfterSwitchOn=2)
        smoothing.__setDictionary__(model.getNNModelModule().named_parameters())

        smoothing(None, helper, model, None, None, metadata)
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), {})
        smoothing(None, helper, model, None, None, metadata)
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), {})
        smoothing(None, helper, model, None, None, metadata) # zapisanie wag
        self.compareDictTensorToNumpy(smoothing.__getSmoothedWeights__(metadata), init_weights)

        self.utils_checkSmoothedWeights(model, smoothing, helper, metadata, 17, 19, 5, 7, 2)
        self.utils_checkSmoothedWeights(model, smoothing, helper, metadata, 23, 29, 5+17, 7+19, 3)
        self.utils_checkSmoothedWeights(model, smoothing, helper, metadata, 45, 85, 5+17+23, 7+19+29, 4) 


if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
