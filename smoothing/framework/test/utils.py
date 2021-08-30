
import pandas as pd
from framework import smoothingFramework as sf
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

init_weights = {
    'linear1.weight': [[5., 5., 5.]], 
    'linear1.bias': [7.], 
    'linear2.weight': [[5.], [5.], [5.]], 
    'linear2.bias': [7., 7., 7.]
}

init_weights_tensor = {
    'linear1.weight': torch.tensor([[5., 5., 5.]]), 
    'linear1.bias': torch.tensor([7.]), 
    'linear2.weight': torch.tensor([[5.], [5.], [5.]]), 
    'linear2.bias': torch.tensor([7., 7., 7.])
}



class Utils(unittest.TestCase):
    """
    Utility class
    """

    def cmpPandas(self, obj_1, name_1, obj_2, name_2 = None):
        if(name_2 is None):
            pd.testing.assert_frame_equal(pd.DataFrame([{name_1: obj_1}]), pd.DataFrame([{name_1: obj_2}]))
        else:
            pd.testing.assert_frame_equal(pd.DataFrame([{name_1: obj_1}]), pd.DataFrame([{name_2: obj_2}]))

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
            self.cmpPandas(ar, 'array', numpyArray[idx])
            idx += 1

    def compareDictToNumpy(self, iterator, numpyDict: dict):
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
            self.cmpPandas(ar, 'array', numpyDict[key])
            idx += 1

    def compareDictTensorToTorch(self, iterator, torchDict: dict):
        if(len(torchDict) == 0):
            test = False
            for ar in iterator:
                test = True
            if(test):
                self.fail("Comparing empty to non-empty dicttionary. Iter:\n{}\n:Dict:\n{}".format(iterator, torchDict)) 

        idx = 0
        for key, ar in iterator.items():
            if(idx >= len(torchDict)):
                self.fail("Arrays size not equals.") 
            if(key not in torchDict):
                self.fail("Dictionary key not found.") 
            self.cmpPandas(ar.detach().numpy(), 'array', torchDict[key].detach().numpy())
            idx += 1
            
    def setWeightDict(self, w, b):
        return {
            'linear1.weight': [[w, w, w]], 
            'linear1.bias': [b], 
            'linear2.weight': [[w], [w], [w]], 
            'linear2.bias': [b, b, b]
        }

    def setWeightTensorDict(self, w, b, dtype=torch.float32, mul=1):
        return {
            'linear1.weight': torch.tensor([[w, w, w]]).to(dtype=dtype).mul_(mul), 
            'linear1.bias': torch.tensor([b]).to(dtype=dtype).mul_(mul), 
            'linear2.weight': torch.tensor([[w], [w], [w]]).to(dtype=dtype).mul_(mul), 
            'linear2.bias': torch.tensor([b, b, b]).to(dtype=dtype).mul_(mul)
        }


class Test_Data(Utils):

    def test_updateTotalNumbLoops_testMode(self):
        with sf.test_mode():
            dataMetadata = dc.DefaultData_Metadata(epoch=7)
            data = dc.DefaultDataMNIST(dataMetadata)
            data.epochHelper = sf.EpochDataContainer()

            self.cmpPandas(True, "test_mode", sf.test_mode.isActive())

            data._updateTotalNumbLoops(dataMetadata)

            self.cmpPandas(data.epochHelper.maxTrainTotalNumber, "max_loops_train", sf.StaticData.MAX_EPOCH_DEBUG_LOOPS * sf.StaticData.MAX_DEBUG_LOOPS * 1)
            self.cmpPandas(data.epochHelper.maxTestTotalNumber, "max_loops_test", sf.StaticData.MAX_EPOCH_DEBUG_LOOPS * sf.StaticData.MAX_DEBUG_LOOPS * 2)
        
    def test_updateTotalNumbLoops(self):
        dataMetadata = dc.DefaultData_Metadata(epoch=7)
        data = dc.DefaultDataMNIST(dataMetadata)
        data.epochHelper = sf.EpochDataContainer()

        data._updateTotalNumbLoops(dataMetadata)

        self.cmpPandas(False, "test_mode", sf.test_mode.isActive())

        self.cmpPandas(64, "batch_size", dataMetadata.batchTrainSize)
        self.cmpPandas(64, "batch_size", dataMetadata.batchTestSize)

        self.cmpPandas(data.epochHelper.maxTrainTotalNumber, "max_loops_train", int(7 * 1 * len(data.trainloader)))
        self.cmpPandas(data.epochHelper.maxTestTotalNumber, "max_loops_test", int(7 * 2 * len(data.testloader)))

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
