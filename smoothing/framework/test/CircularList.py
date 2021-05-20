from framework import defaultClasses as dc
from framework import smoothingFramework as sf
import pandas as pd
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sf.StaticData.LOG_FOLDER = './smoothing/framework/test/dump/'

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
    def compareArrays(self, iterator, numpyArray: list, lab = lambda x : x):
        idx = 0
        for ar in iterator:
            if(idx >= len(numpyArray)):
                self.fail("Arrays size not equals.") 
            ar = lab(ar)
            #testCmpPandas(ar.shape, 'array_shape', numpyArray[idx].shape)
            testCmpPandas(ar, 'array', numpyArray[idx])
            idx += 1


class Test_DefaultSmoothingOscilationWeightedMean(Test_DefaultSmoothing):
    def test_calcAvgWeightedMean(self):
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(weightDecay=2)
        smoothing.enabled = True
        smoothing.calcAvgWeightedMean(model)
        weights = smoothing.weightsArray.array
        li1_wg = np.array([[5., 5., 5.]])
        li1_bias = np.array([7.])
        li2_wg = np.array([
            [5.],
            [5.],
            [5.]
        ])
        li2_bias = np.array([7., 7., 7.])

        i = iter(model.parameters())
        self.compareArrays(i, [li1_wg, li1_bias, li2_wg, li2_bias], lambda x : x.detach().numpy())

        i = iter(weights[0].values())
        self.compareArrays(i, [li1_wg, li1_bias, li2_wg, li2_bias], lambda x : x.detach().numpy())


        for m in model.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.constant_(m.weight, 11)
                nn.init.constant_(m.bias, 13)
        smoothing.calcAvgWeightedMean(model)

        li1_wg_2 = np.array([[11., 11., 11.]])
        li_bias_2 = np.array([13.])
        li2_wg_2 = np.array([
            [11.],
            [11.],
            [11.]
        ])
        li2_bias_2 = np.array([13., 13., 13.])

        i = iter(weights[1].values())
        self.compareArrays(i, [li1_wg_2, li_bias_2, li2_wg_2, li2_bias_2], lambda x : x.detach().numpy())

        smoothing.countWeights = 2
        sm_weights = smoothing.__getSmoothedWeights__(None)

        li1_wg_avg = np.array([[9., 9., 9.]])
        li_bias_avg = np.array([11.])
        li2_wg_avg = np.array([
            [9.],
            [9.],
            [9.]
        ])
        li2_bias_avg = np.array([11., 11., 11.])

        i = iter(sm_weights.values())
        self.compareArrays(i, [li1_wg_avg, li_bias_avg, li2_wg_avg, li2_bias_avg], lambda x : x.detach().numpy())

    def test__saveAvgSum(self): # TODO
        metadata = sf.Metadata()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean()
        smoothing.enabled = True

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
        
    def test__sumAllWeights(self):
        metadata = sf.Metadata()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean()
        smoothing.enabled = True
        smoothing.countWeights = 1

        smoothing.calcAvgWeightedMean(model)
        sumWg = smoothing._sumAllWeights(metadata)
        testCmpPandas(sumWg, 'weight_sum', 58.0)

    def test__smoothingGoodEnoughCheck(self):
        metadata = sf.Metadata()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(softMarginAdditionalLoops=1, weightsEpsilon = 1.0)
        smoothing.enabled = True
        smoothing.countWeights = 1
        a = 2.0
        b = 1.5

        testCmpPandas(smoothing._smoothingGoodEnoughCheck(a, b), 'bool', False)
        testCmpPandas(smoothing._smoothingGoodEnoughCheck(a, b), 'bool', True)
        testCmpPandas(smoothing._smoothingGoodEnoughCheck(b, a), 'bool', True)
        b = 3.1
        testCmpPandas(smoothing._smoothingGoodEnoughCheck(a, b), 'bool', False)

    def test___isSmoothingGoodEnough__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_1'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(epsilon=1.0, avgOfAvgUpdateFreq=2, whenCheckCanComputeWeights=1,
        weightsEpsilon=1.0, numbOfBatchMinStart=1, endSmoothingFreq=2, softMarginAdditionalLoops=1)
        smoothing.enabled = True
        helper.loss = torch.Tensor([1.0])

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 1.0)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 0)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)
        helper.loss = torch.Tensor([0.5])

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 1.5/2)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 0.5)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 2/3)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 0.5)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        helper.loss = torch.Tensor([1.5])
        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 3.5/4)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 2/2)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 5/5)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 2/2)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 6.5/6)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 3.5/3)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 8/7)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 3.5/3)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 9.5/8)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 5/4)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', True)

class Test_DefaultSmoothingOscilationMovingMean(Test_DefaultSmoothing):
    def test___isSmoothingGoodEnough__(self):
        metadata = sf.Metadata()
        metadata.debugInfo = True
        metadata.logFolderSuffix = 'test_2'
        metadata.debugOutput = 'debug'
        metadata.prepareOutput()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)
        helper = sf.TrainDataContainer()

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(epsilon=1.0, avgOfAvgUpdateFreq=2, whenCheckCanComputeWeights=1,
        weightsEpsilon=1.0, numbOfBatchMinStart=1, endSmoothingFreq=2, softMarginAdditionalLoops=1)
        smoothing.enabled = True
        helper.loss = torch.Tensor([1.0])

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 1.0)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 0)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)
        helper.loss = torch.Tensor([0.5])

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 1.5/2)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 0.5)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 2/3)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 0.5)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        helper.loss = torch.Tensor([1.5])
        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 3.5/4)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 2/2)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 5/5)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 2/2)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 6.5/6)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 3.5/3)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 8/7)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 3.5/3)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', False)

        smoothing(None, helper, model, None, None, metadata)
        testCmpPandas(smoothing.lossContainer.getAverage(), 'average', 9.5/8)
        testCmpPandas(smoothing.lastKLossAverage.getAverage(), 'average', 5/4)
        testCmpPandas(smoothing.__isSmoothingGoodEnough__(None, helper, model, None, None, metadata), 'isSmoothingGoodEnough', True)


def run():
    sf.useDeterministic()
    inst = Test_CircularList()
    inst.test_pushBack()
    inst.test_getAverage()
    inst.test_iteration()
    inst.test_len()

    inst = Test_DefaultSmoothingOscilationWeightedMean()
    inst.test_calcAvgWeightedMean()
    inst.test__saveAvgSum()
    inst.test__sumAllWeights()
    inst.test__smoothingGoodEnoughCheck()
    inst.test___isSmoothingGoodEnough__()

    init = Test_DefaultSmoothingOscilationMovingMean()
    init.test___isSmoothingGoodEnough__()


if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
