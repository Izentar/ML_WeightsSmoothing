
import pandas as pd
from framework import smoothingFramework as sf
import unittest
import torch

sf.StaticData.LOG_FOLDER = './framework/test/dump/'
sf.StaticData.MAX_DEBUG_LOOPS = 50
sf.CURRENT_STAT_PATH = sf.StaticData.LOG_FOLDER

def testCmpPandas(obj_1, name_1, obj_2, name_2 = None):
    if(name_2 is None):
        pd.testing.assert_frame_equal(pd.DataFrame([{name_1: obj_1}]), pd.DataFrame([{name_1: obj_2}]))
    else:
        pd.testing.assert_frame_equal(pd.DataFrame([{name_1: obj_1}]), pd.DataFrame([{name_2: obj_2}]))


class Utils(unittest.TestCase):
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
            testCmpPandas(ar, 'array', numpyDict[key])
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
            testCmpPandas(ar.detach().numpy(), 'array', torchDict[key].detach().numpy())
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