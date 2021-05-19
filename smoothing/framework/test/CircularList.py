from framework import defaultClasses as dc
from framework import smoothingFramework as sf
import pandas as pd
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sf.StaticData.LOG_FOLDER = './framework/test/dump/'

class Test_CircularList(unittest.TestCase):

    def test_pushBack(self):
        inst = dc.CircularList(2)
        inst.pushBack(1)
        assert inst.array[0] == 1
        inst.pushBack(2)
        assert inst.array[0] == 1
        assert inst.array[1] == 2
        inst.pushBack(3)
        assert inst.array[0] == 3
        assert inst.array[1] == 2
        inst.pushBack(4)
        assert inst.array[0] == 3
        assert inst.array[1] == 4

        inst.reset()
        assert len(inst.array) == 0
        inst.pushBack(10)
        assert inst.array[0] == 10
        assert len(inst.array) == 1

    def test_getAverage(self):
        inst = dc.CircularList(3)
        assert inst.getAverage() == 0
        inst.pushBack(1)
        pd.testing.assert_frame_equal(pd.DataFrame([{'average': inst.getAverage()}]), pd.DataFrame([{'average': 1.0}]))
        inst.pushBack(2)
        pd.testing.assert_frame_equal(pd.DataFrame([{'average': inst.getAverage()}]), pd.DataFrame([{'average': 1.5}]))
        inst.pushBack(3)
        pd.testing.assert_frame_equal(pd.DataFrame([{'average': inst.getAverage()}]), pd.DataFrame([{'average': 2.0}]))
        inst.pushBack(4)
        pd.testing.assert_frame_equal(pd.DataFrame([{'average': inst.getAverage()}]), pd.DataFrame([{'average': 3.0}]))
        inst.pushBack(5)
        pd.testing.assert_frame_equal(pd.DataFrame([{'average': inst.getAverage()}]), pd.DataFrame([{'average': 4.0}]))

        inst.reset()
        assert len(inst.array) == 0
        inst.pushBack(10)
        assert inst.array[0] == 10
        assert len(inst.array) == 1

    def test_iteration(self):
        inst = dc.CircularList(3)
        inst.pushBack(1)
        inst.pushBack(2)
        inst.pushBack(3)

        i = iter(inst)
        assert i.indexArray == [2, 1, 0]

        assert next(i) == 3
        assert next(i) == 2
        assert next(i) == 1
        self.assertRaises(StopIteration, lambda : next(i))

        inst.pushBack(4)
        i = iter(inst)
        assert next(i) == 4
        assert next(i) == 3
        assert next(i) == 2
        self.assertRaises(StopIteration, lambda : next(i))

        inst.pushBack(5)
        i = iter(inst)
        assert next(i) == 5
        assert next(i) == 4
        assert next(i) == 3
        self.assertRaises(StopIteration, lambda : next(i))
        
    def test_len(self):
        inst = dc.CircularList(3)
        assert len(inst) == 0
        inst.pushBack(1)
        assert len(inst) == 1
        inst.pushBack(2)
        assert len(inst) == 2
        inst.pushBack(3)
        assert len(inst) == 3
        inst.pushBack(4)
        assert len(inst) == 3
        inst.pushBack(5)
        assert len(inst) == 3

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

class Test_DefaultSmoothingOscilationWeightedMean(unittest.TestCase):
    def compareArrays(self, iterator, numpyArray: list, lab = lambda x : x):
        
        idx = 0
        for ar in iterator:
            if(idx >= len(numpyArray)):
                self.fail("Arrays size not equals.") 
            ar = lab(ar)
            print(ar, "------", numpyArray[idx], '\n')
            assert ar.shape == numpyArray[idx].shape 
            assert np.allclose(ar, numpyArray[idx])
            idx += 1

    def test_calcAvgWeightedMean(self):
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean()
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
        print(sm_weights)

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

    def test__isSmoothingIsGoodEnough__(self): # TODO
        metadata = sf.Metadata()
        modelMetadata = TestModel_Metadata()
        model = TestModel(modelMetadata)

        smoothing = dc.DefaultSmoothingOscilationWeightedMean()
        smoothing.enabled = True

        smoothing.calcAvgWeightedMean(model)
        assert smoothing.__isSmoothingIsGoodEnough__(None, None, model, None, None, metadata) == False


        for m in model.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.constant_(m.weight, 11)
                nn.init.constant_(m.bias, 13)
        smoothing.calcAvgWeightedMean(model)
        smoothing.calcAvgWeightedMean(model)
        smoothing.countWeights = 2
        smoothing.endSmoothingFreq = 1
        smoothing.weightsEpsilon = 1.0

        smoothing.__isSmoothingIsGoodEnough__(None, None, model, None, None, metadata)
        


def run():
    sf.useDeterministic()
    inst = Test_CircularList()
    inst.test_pushBack()
    inst.test_getAverage()
    inst.test_iteration()
    inst.test_len()

    inst = Test_DefaultSmoothingOscilationWeightedMean()
    inst.test_calcAvgWeightedMean()


if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
