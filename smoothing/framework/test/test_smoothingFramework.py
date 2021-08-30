from framework import smoothingFramework as sf
import pandas as pd
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
import time

from framework.test import utils as ut

init_weights = {
    'linear1.weight': [[5., 5., 5.]], 
    'linear1.bias': [7.], 
    'linear2.weight': [[5.], [5.], [5.]], 
    'linear2.bias': [7., 7., 7.]
}

class Test_BaseSampler(ut.Utils):
    def test_sequence(self):
        sampler = sf.BaseSampler(dataSize=10, batchSize=1, startIndex=2, seed=988)
        testList = [2, 3, 4, 5, 6, 7, 8, 9]
        random.Random(988).shuffle(testList)
        self.cmpPandas(sampler.sequence, "sampler_sequence", testList)

    def test_sequence_2(self):
        sampler = sf.BaseSampler(dataSize=10, batchSize=2, startIndex=2, seed=988)
        testList = [4, 5, 6, 7, 8, 9]
        random.Random(988).shuffle(testList)
        self.cmpPandas(sampler.sequence, "sampler_sequence_2", testList)

class Test_test_mode(ut.Utils):
    def test_onOff(self):
        self.cmpPandas(sf.test_mode.isActive(), "test_mode_plain", False)
        sf.StaticData.TEST_MODE = True
        self.cmpPandas(sf.test_mode.isActive(), "test_mode_on", True)
        sf.StaticData.TEST_MODE = False
        self.cmpPandas(sf.test_mode.isActive(), "test_mode_turn_off", False)

    def test_enterExit(self):
        self.cmpPandas(sf.test_mode.isActive(), "test_mode_plain", False)
        with sf.test_mode():
            self.cmpPandas(sf.test_mode.isActive(), "test_mode_on", True)
        self.cmpPandas(sf.test_mode.isActive(), "test_mode_turn_off", False)

class Test_CircularList(ut.Utils):

    def test_pushBack(self):
        inst = sf.CircularList(2)
        inst.pushBack(1)
        self.cmpPandas(inst.array[0], 'array_value', 1)
        inst.pushBack(2)
        self.cmpPandas(inst.array[0], 'array_value', 1)
        self.cmpPandas(inst.array[1], 'array_value', 2)
        inst.pushBack(3)
        self.cmpPandas(inst.array[0], 'array_value', 3)
        self.cmpPandas(inst.array[1], 'array_value', 2)
        inst.pushBack(4)
        self.cmpPandas(inst.array[0], 'array_value', 3)
        self.cmpPandas(inst.array[1], 'array_value', 4)

        inst.reset()
        self.cmpPandas(len(inst.array), 'array_length', 0)
        inst.pushBack(10)
        self.cmpPandas(inst.array[0], 'array_value', 10)
        self.cmpPandas(len(inst.array), 'array_length', 1)

    def test_getAverage(self):
        inst = sf.CircularList(3)
        self.cmpPandas(inst.getAverage(), 'average', 0)
        inst.pushBack(1)
        self.cmpPandas(inst.getAverage(), 'average', 1.0)
        inst.pushBack(2)
        self.cmpPandas(inst.getAverage(), 'average', 1.5)
        inst.pushBack(3)
        self.cmpPandas(inst.getAverage(), 'average', 2.0)
        self.cmpPandas(inst.getAverage(startAt=1), 'average', 1.5)
        inst.pushBack(4)
        self.cmpPandas(inst.getAverage(), 'average', 3.0)
        inst.pushBack(5)
        self.cmpPandas(inst.getAverage(), 'average', 4.0)

        self.cmpPandas(inst.getAverage(startAt=1), 'average', 3.5)
        self.cmpPandas(inst.getAverage(startAt=2), 'average', 3.0)

        inst.reset()
        self.cmpPandas(len(inst.array), 'array_length', 0)
        inst.pushBack(10)
        self.cmpPandas(inst.getAverage(), 'average', 10.0)
        self.cmpPandas(len(inst.array), 'array_length', 1)

        self.cmpPandas(inst.getAverage(startAt=1), 'average', 0)

    def test_iteration(self):
        inst = sf.CircularList(3)
        inst.pushBack(1)
        inst.pushBack(2)
        inst.pushBack(3)

        i = iter(inst)
        self.cmpPandas(i.indexArray, 'array', [2, 1, 0])

        self.cmpPandas(next(i), 'iter_next', 3)
        self.cmpPandas(next(i), 'iter_next', 2)
        self.cmpPandas(next(i), 'iter_next', 1)

        self.assertRaises(StopIteration, lambda : next(i))

        inst.pushBack(4)
        i = iter(inst)
        self.cmpPandas(next(i), 'iter_next', 4)
        self.cmpPandas(next(i), 'iter_next', 3)
        self.cmpPandas(next(i), 'iter_next', 2)
        self.assertRaises(StopIteration, lambda : next(i))

        inst.pushBack(5)
        i = iter(inst)
        self.cmpPandas(next(i), 'iter_next', 5)
        self.cmpPandas(next(i), 'iter_next', 4)
        self.cmpPandas(next(i), 'iter_next', 3)
        self.assertRaises(StopIteration, lambda : next(i))
        
    def test_len(self):
        inst = sf.CircularList(3)
        self.cmpPandas(len(inst), 'array_length', 0)
        inst.pushBack(1)
        self.cmpPandas(len(inst), 'array_length', 1)
        inst.pushBack(2)
        self.cmpPandas(len(inst), 'array_length', 2)
        inst.pushBack(3)
        self.cmpPandas(len(inst), 'array_length', 3)
        inst.pushBack(4)
        self.cmpPandas(len(inst), 'array_length', 3)
        inst.pushBack(5)
        self.cmpPandas(len(inst), 'array_length', 3)
 
    def test_std_mean(self):
        inst = sf.CircularList(2)
        inst.pushBack(1)
        self.cmpPandas(inst.array[0], 'array_value', 1)
        inst.pushBack(2)
        self.cmpPandas(inst.array[0], 'array_value', 1)
        self.cmpPandas(inst.array[1], 'array_value', 2)
        inst.pushBack(3)

        self.cmpPandas(inst.getMean(), 'mean', 2.5)
        self.cmpPandas(inst.getStd(), 'mean', 0.5)

        inst.pushBack(5)
        self.cmpPandas(inst.getMean(), 'mean', 4.0)
        self.cmpPandas(inst.getStd(), 'mean', 1.0)

    def test_min_max(self):
        inst = sf.CircularList(4)
        inst.pushBack(1)
        self.cmpPandas(inst.array[0], 'array_value', 1)
        inst.pushBack(2)
        self.cmpPandas(inst.array[0], 'array_value', 1)
        self.cmpPandas(inst.array[1], 'array_value', 2)
        inst.pushBack(3)
        inst.pushBack(1)
        inst.pushBack(9)
        inst.pushBack(8)


        self.cmpPandas(inst.getMin(), 'mean', 1)
        self.cmpPandas(inst.getMax(), 'mean', 9)

        inst.pushBack(4)
        inst.pushBack(5)

        self.cmpPandas(inst.getMin(), 'mean', 4)
        self.cmpPandas(inst.getMax(), 'mean', 9)

class Test_Timer(ut.Utils):
    def setUp(self):
        self.timer = sf.Timer()

    def test_getDiff(self):
        self.timer.start()
        self.timer.end()
        self.timer.start('cpu')
        self.timer.end('cuda:0')
        self.timer.timeStart = 1.0
        self.timer.timeEnd = 2.5
        
        self.cmpPandas(self.timer.getDiff(), "timer_diff", 2.5 - 1.0)

        self.timer.timeStart = 5.6
        self.timer.timeEnd = 8.2
        self.cmpPandas(self.timer.getDiff(), "timer_diff", 8.2 - 5.6)

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
        
        self.cmpPandas(self.timer.getTimeSum(), "timer_sum", (2.5 - 1.0) + (5.5 - 3.0) + (25.7 - 10.0))
        self.cmpPandas(self.timer.getAverage(), "timer_sum", ((2.5 - 1.0) + (5.5 - 3.0) + (25.7 - 10.0)) / 3)

class Test_LoopsState(ut.Utils):
    def test_imprint(self):
        state = sf.LoopsState()
        ok = state.decide()
        self.cmpPandas(ok, "loopState_loop_here", 0)

        state.imprint(64, True)
        state.imprint(32, False)

        tmp = pickle.dumps(state)
        state = pickle.loads(tmp)

        ok = state.decide()
        self.cmpPandas(ok, "loopState_go_next", None)

        ok = state.decide()
        self.cmpPandas(ok, "loopState_loop_here", 32)

    def test_imprint_2(self):
        state = sf.LoopsState()
        ok = state.decide()
        self.cmpPandas(ok, "loopState_loop_here", 0)

        state.imprint(64, True)
        state.imprint(32, True)

        tmp = pickle.dumps(state)
        state = pickle.loads(tmp)

        ok = state.decide()
        self.cmpPandas(ok, "loopState_go_next", None)

        ok = state.decide()
        self.cmpPandas(ok, "loopState_go_next", None)

        ok = state.decide()
        self.cmpPandas(ok, "loopState_loop_here", 0)

class Test_Data_Metadata(ut.Utils):
    def test_pinMemory(self):
        ok = False
        if(torch.cuda.is_available()):
            ok = True

        metadata = sf.Metadata(debugInfo=False)
        model_metadata = sf.Model_Metadata()
        data_metadata = sf.Data_Metadata()
        self.cmpPandas(data_metadata.pin_memoryTrain, "pin_memory_train", False)
        self.cmpPandas(data_metadata.pin_memoryTest, "pin_memory_test", False)

        data_metadata.tryPinMemoryTrain(metadata, model_metadata)
        self.cmpPandas(data_metadata.pin_memoryTrain, "pin_memory_train", ok)

        data_metadata.tryPinMemoryTest(metadata, model_metadata)
        self.cmpPandas(data_metadata.pin_memoryTest, "pin_memory_test", ok)

class Test_RunningArthmeticMeanWeights(ut.Utils):
    def test_calcMeanDullInit(self):
        weights = self.setWeightTensorDict(2, 5)
        arth = sf.RunningGeneralMeanWeights(initWeights=weights)

        weights = self.setWeightTensorDict(2, 5)
        arth.addWeights(weights)

        avgWeights = arth.getWeights()
        self.compareDictTensorToTorch(avgWeights, weights)

        weights = self.setWeightTensorDict(5, 8)
        arth.addWeights(weights)

        avgWeights = arth.getWeights()
        weights = self.setWeightTensorDict(3, 6)
        self.compareDictTensorToTorch(avgWeights, weights)

    def test_calcMeanInitZeros(self):
        weights = self.setWeightTensorDict(2, 5)
        arth = sf.RunningGeneralMeanWeights(initWeights=weights, setToZeros=True)

        weights = self.setWeightTensorDict(2, 5)
        arth.addWeights(weights)

        avgWeights = arth.getWeights()
        self.compareDictTensorToTorch(avgWeights, weights)

        weights = self.setWeightTensorDict(6, 7)
        arth.addWeights(weights)

        avgWeights = arth.getWeights()
        weights = self.setWeightTensorDict(4, 6)
        self.compareDictTensorToTorch(avgWeights, weights)

    def test_calcMeanPow2(self):
        weights = self.setWeightTensorDict(2, 5)
        arth = sf.RunningGeneralMeanWeights(initWeights=weights, setToZeros=True, power=2)

        weights = self.setWeightTensorDict(2, 5)
        arth.addWeights(weights)

        avgWeights = arth.getWeights()
        self.compareDictTensorToTorch(avgWeights, weights)

        weights = self.setWeightTensorDict(6, 7)
        arth.addWeights(weights)

        avgWeights = arth.getWeights()
        weights = self.setWeightTensorDict(4.47213, 6.08276)
        self.compareDictTensorToTorch(avgWeights, weights)

    def test_calcMeanAsWeightened(self):
        wg = 1
        weights = self.setWeightTensorDict(2, 5)
        arth = sf.RunningGeneralMeanWeights(initWeights=weights, setToZeros=True)

        weights = self.setWeightTensorDict(2, 5, mul=wg)
        wg = wg / 2
        arth.addWeights(weights)

        avgWeights = arth.getWeights()
        self.compareDictTensorToTorch(avgWeights, weights)

        weights = self.setWeightTensorDict(6, 7, mul=wg)
        wg = wg / 2
        arth.addWeights(weights)

        avgWeights = arth.getWeights()
        weights = self.setWeightTensorDict(2.5, 4.25)
        self.compareDictTensorToTorch(avgWeights, weights)

class Test_MultiplicativeLR(ut.Utils):
    def setUp(self):
        self.modelMetadata = ut.TestModel_Metadata()
        self.model = ut.TestModel(self.modelMetadata)

    def test_1(self):
        optimizer = optim.SGD(self.model.getNNModelModule().parameters(), lr=0.01, 
            weight_decay=0.0001, momentum=0.9, nesterov=False)
        inst = sf.MultiplicativeLR(optimizer, gamma=0.1, startAt=0)

        self.cmpPandas(inst.get_last_lr()[0], 'array_value', 0.01)
        
        inst.step()
        self.cmpPandas(inst.get_last_lr()[0], 'array_value', 0.001)

        inst.step()
        self.cmpPandas(inst.get_last_lr()[0], 'array_value', 0.0001)

    def test_2(self):
        optimizer = optim.SGD(self.model.getNNModelModule().parameters(), lr=0.01, 
            weight_decay=0.0001, momentum=0.9, nesterov=False)
        inst = sf.MultiplicativeLR(optimizer, gamma=0.1, startAt=2)

        self.cmpPandas(inst.get_last_lr()[0], 'array_value', 0.0001)

        inst.step()
        self.cmpPandas(inst.get_last_lr()[0], 'array_value', 0.00001)

        inst.step()
        self.cmpPandas(inst.get_last_lr()[0], 'array_value', 0.000001)

class Test_SchedulerContainer(ut.Utils):
    def setUp(self):
        self.modelMetadata = ut.TestModel_Metadata()
        self.model = ut.TestModel(self.modelMetadata)
        self.optimizer = optim.SGD(self.model.getNNModelModule().parameters(), lr=0.01, 
            weight_decay=0.0001, momentum=0.9, nesterov=False)
        self.metadata = sf.Metadata(debugInfo=True)
        self.metadata.debugOutput = 'debug'
        self.metadata.logFolderSuffix = str(time.time())
        self.metadata.prepareOutput()

    def test_1(self):
        inst = sf.SchedulerContainer(schedType='smoothing', importance=2)
        inst2 = sf.SchedulerContainer(schedType='normal', importance=1)

        self.cmpPandas(inst.getImportance(), 'importance', 2)
        self.cmpPandas(inst2.getImportance(), 'importance', 1)
        self.cmpPandas(inst.getType(), 'importance', 'smoothing')
        self.cmpPandas(inst2.getType(), 'importance', 'normal')

    def test_2(self):
        sched = sf.MultiplicativeLR(self.optimizer, gamma=0.1)
        sched2 = sf.MultiplicativeLR(self.optimizer, gamma=0.2)

        inst = sf.SchedulerContainer(schedType='smoothing', importance=2)

        inst.add(schedule=[0, 1, 2, 3, 4, 5, 6], scheduler=sched, metric=False)
        inst.add(schedule=[0, 1, 2, 3, 4, 5, 6], scheduler=sched2, metric=False)

        inst.step(shtypes='smoothing', epochNumb=5, metadata=self.metadata, metrics=None)
        self.cmpPandas(sched.get_last_lr()[0], 'lr', 0.0002)

        inst.step(shtypes='smoothing', epochNumb=7, metadata=self.metadata, metrics=None)
        self.cmpPandas(sched.get_last_lr()[0], 'lr', 0.0002)

class Test_SchedulerManager(ut.Utils):
    def setUp(self):
        self.modelMetadata = ut.TestModel_Metadata()
        self.model = ut.TestModel(self.modelMetadata)
        self.optimizer = optim.SGD(self.model.getNNModelModule().parameters(), lr=0.01, 
            weight_decay=0.0001, momentum=0.9, nesterov=False)
        self.metadata = sf.Metadata(debugInfo=True)
        self.metadata.debugOutput = 'debug'
        self.metadata.logFolderSuffix = str(time.time())
        self.metadata.prepareOutput()
        

    def test_1(self):
        sched = sf.MultiplicativeLR(self.optimizer, gamma=0.1)
        sched2 = sf.MultiplicativeLR(self.optimizer, gamma=0.2)

        inst = sf.SchedulerContainer(schedType='smoothing', importance=2)
        inst2 = sf.SchedulerContainer(schedType='normal', importance=1)

        inst.add(schedule=[0, 1, 2, 3, 4, 5, 6], scheduler=sched, metric=False)
        inst2.add(schedule=[0, 1, 2, 3, 4, 5, 6], scheduler=sched2, metric=False)
        manager = sf.SchedulerManager()
        manager.add(inst)

        inst.step(shtypes='smoothing', epochNumb=5, metadata=self.metadata, metrics=None)
        self.cmpPandas(sched.get_last_lr()[0], 'lr', 0.001)

        inst.step(shtypes='smoothing', epochNumb=7, metadata=self.metadata, metrics=None)
        self.cmpPandas(sched.get_last_lr()[0], 'lr', 0.001)

        manager.add(inst2)

        manager.schedulerStep(shtypes='normal', epochNumb=5, metadata=self.metadata, metrics=None)
        self.cmpPandas(sched.get_last_lr()[0], 'lr', 0.0002)

        manager.schedulerStep(shtypes='normal', epochNumb=7, metadata=self.metadata, metrics=None)
        self.cmpPandas(sched.get_last_lr()[0], 'lr', 0.0002)

if __name__ == '__main__':
    sf.useDeterministic()
    unittest.main()
