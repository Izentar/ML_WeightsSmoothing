from framework import defaultClasses as dc
from framework import smoothingFramework as sf
import pandas as pd
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle

from framework.test import utils as ut

init_weights = {
    'linear1.weight': [[5., 5., 5.]], 
    'linear1.bias': [7.], 
    'linear2.weight': [[5.], [5.], [5.]], 
    'linear2.bias': [7., 7., 7.]
}

class Test_BaseSampler(unittest.TestCase):
    def test_sequence(self):
        sampler = sf.BaseSampler(dataSize=10, batchSize=1, startIndex=2, seed=988)
        testList = [2, 3, 4, 5, 6, 7, 8, 9]
        random.Random(988).shuffle(testList)
        ut.testCmpPandas(sampler.sequence, "sampler_sequence", testList)

    def test_sequence_2(self):
        sampler = sf.BaseSampler(dataSize=10, batchSize=2, startIndex=2, seed=988)
        testList = [4, 5, 6, 7, 8, 9]
        random.Random(988).shuffle(testList)
        ut.testCmpPandas(sampler.sequence, "sampler_sequence_2", testList)

class Test_test_mode(unittest.TestCase):
    def test_onOff(self):
        ut.testCmpPandas(sf.test_mode.isActive(), "test_mode_plain", False)
        sf.StaticData.TEST_MODE = True
        ut.testCmpPandas(sf.test_mode.isActive(), "test_mode_on", True)
        sf.StaticData.TEST_MODE = False
        ut.testCmpPandas(sf.test_mode.isActive(), "test_mode_turn_off", False)

    def test_enterExit(self):
        ut.testCmpPandas(sf.test_mode.isActive(), "test_mode_plain", False)
        with sf.test_mode():
            ut.testCmpPandas(sf.test_mode.isActive(), "test_mode_on", True)
        ut.testCmpPandas(sf.test_mode.isActive(), "test_mode_turn_off", False)
        

class Test_Timer(unittest.TestCase):
    def setUp(self):
        self.timer = sf.Timer()

    def test_getDiff(self):
        self.timer.start()
        self.timer.end()
        self.timer.timeStart = 1.0
        self.timer.timeEnd = 2.5
        
        ut.testCmpPandas(self.timer.getDiff(), "timer_diff", 2.5 - 1.0)

        self.timer.timeStart = 5.6
        self.timer.timeEnd = 8.2
        ut.testCmpPandas(self.timer.getDiff(), "timer_diff", 8.2 - 5.6)

    def test_statistics(self):
        self.timer.start()
        self.timer.end()
        self.timer.timeStart = 1.0
        self.timer.timeEnd = 2.5
        self.timer.addToStatistics()

        self.timer.timeStart = 3.0
        self.timer.timeEnd = 5.5
        self.timer.addToStatistics()

        self.timer.timeStart = 10.0
        self.timer.timeEnd = 25.7
        self.timer.addToStatistics()
        
        ut.testCmpPandas(self.timer.getTimeSum(), "timer_sum", (2.5 - 1.0) + (5.5 - 3.0) + (25.7 - 10.0))
        ut.testCmpPandas(self.timer.getAverage(), "timer_sum", ((2.5 - 1.0) + (5.5 - 3.0) + (25.7 - 10.0)) / 3)


class Test_LoopsState(unittest.TestCase):
    def test_imprint(self):
        state = sf.LoopsState()
        ok = state.decide()
        ut.testCmpPandas(ok, "loopState_loop_here", 0)

        state.imprint(64, True)
        state.imprint(32, False)

        tmp = pickle.dumps(state)
        state = pickle.loads(tmp)

        ok = state.decide()
        ut.testCmpPandas(ok, "loopState_go_next", None)

        ok = state.decide()
        ut.testCmpPandas(ok, "loopState_loop_here", 32)

    def test_imprint_2(self):
        state = sf.LoopsState()
        ok = state.decide()
        ut.testCmpPandas(ok, "loopState_loop_here", 0)

        state.imprint(64, True)
        state.imprint(32, True)

        tmp = pickle.dumps(state)
        state = pickle.loads(tmp)

        ok = state.decide()
        ut.testCmpPandas(ok, "loopState_go_next", None)

        ok = state.decide()
        ut.testCmpPandas(ok, "loopState_go_next", None)

        ok = state.decide()
        ut.testCmpPandas(ok, "loopState_loop_here", 0)

class Test_Data_Metadata(unittest.TestCase):
    def test_pinMemory(self):
        ok = False
        if(torch.cuda.is_available()):
            ok = True

        metadata = sf.Metadata(debugInfo=False)
        model_metadata = sf.Model_Metadata()
        data_metadata = sf.Data_Metadata()
        ut.testCmpPandas(data_metadata.pin_memoryTrain, "pin_memory_train", False)
        ut.testCmpPandas(data_metadata.pin_memoryTest, "pin_memory_test", False)

        data_metadata.tryPinMemoryTrain(metadata, model_metadata)
        ut.testCmpPandas(data_metadata.pin_memoryTrain, "pin_memory_train", ok)

        data_metadata.tryPinMemoryTest(metadata, model_metadata)
        ut.testCmpPandas(data_metadata.pin_memoryTest, "pin_memory_test", ok)

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


if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
