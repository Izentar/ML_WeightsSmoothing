from framework import defaultClasses as dc
from framework import smoothingFramework as sf
import pandas as pd
import unittest
import torch
import numpy as np
import time
from framework.test import utils as ut
import torchvision.models as models


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
        self.cmpPandas(smoothing.lossContainer.getAverage(), 'average', avgLoss)
        self.cmpPandas(smoothing.lossContainer.getAverage(smoothingMetadata.lossContainerDelayedStartAt), 'average', avgKLoss)
        self.cmpPandas(smoothing.__isSmoothingGoodEnough__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=None, modelMetadata=None, metadata=metadata, smoothingMetadata=smoothingMetadata), 'isSmoothingGoodEnough', booleanIsGood)

# Test__SmoothingOscilationBase
class Test__SmoothingOscilationBase(ut.Utils):
    def setUp(self):
        self.metadata = sf.Metadata()
        self.metadata.debugInfo = True
        self.metadata.debugOutput = 'debug'
        self.metadata.logFolderSuffix = str(time.time())
        self.metadata.prepareOutput()
        self.modelMetadata = ut.TestModel_Metadata()
        self.model = ut.TestModel(self.modelMetadata)
        self.helper = sf.TrainDataContainer()
        self.dataMetadata = dc.DefaultData_Metadata()

        self.helperEpoch = sf.EpochDataContainer()
        self.helperEpoch.trainTotalNumber = 3
        self.helperEpoch.maxTrainTotalNumber = 5
        self.helperEpoch.epochNumber = 1
    
class Test__cmpLoss_isLower(Test__SmoothingOscilationBase):
    def test_1(self):
        self.sdrel1 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            lossThreshold = 0, lossThresholdMode = 'rel')
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdrel1)

        smoothing.bestLoss = 9.99
        self.cmpPandas(
            obj_1 = smoothing.cmpLoss_isLower(metric=10, smoothingMetadata=self.sdrel1), 
            name_1 = 'metric_loss_rel_1', 
            obj_2 = False)

    def test_2(self):
        self.sdrel2 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            lossThreshold = 0.5, lossThresholdMode = 'rel')
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdrel2)

        smoothing.bestLoss = 19.99
        self.cmpPandas(
            obj_1 = smoothing.cmpLoss_isLower(metric=10, smoothingMetadata=self.sdrel2), 
            name_1 = 'metric_loss_rel_2', 
            obj_2 = False)

    def test_3(self):
        self.sdrel3 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            lossThreshold = 0.5, lossThresholdMode = 'rel')
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdrel3)

        smoothing.bestLoss = 21
        self.cmpPandas(
            obj_1 = smoothing.cmpLoss_isLower(metric=10, smoothingMetadata=self.sdrel3), 
            name_1 = 'metric_loss_rel_3', 
            obj_2 = True)
        
    def test_4(self):
        self.sdabs1 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            lossThreshold = 0, lossThresholdMode = 'abs')
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdabs1)

        smoothing.bestLoss = 9.99
        self.cmpPandas(
            obj_1 = smoothing.cmpLoss_isLower(metric=10, smoothingMetadata=self.sdabs1), 
            name_1 = 'metric_loss_abs_1', 
            obj_2 = False)

    def test_5(self):
        self.sdabs2 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            lossThreshold = 0.5, lossThresholdMode = 'abs')
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdabs2)
        smoothing.bestLoss = 9.6
        self.cmpPandas(
            obj_1 = smoothing.cmpLoss_isLower(metric=10, smoothingMetadata=self.sdabs2), 
            name_1 = 'metric_loss_abs_2', 
            obj_2 = False)

    def test_6(self):
        self.sdabs3 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            lossThreshold = 0.5, lossThresholdMode = 'abs')
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdabs3)
        smoothing.bestLoss = 10.4
        self.cmpPandas(
            obj_1 = smoothing.cmpLoss_isLower(metric=5, smoothingMetadata=self.sdabs3), 
            name_1 = 'metric_loss_abs_3', 
            obj_2 = True)

class Test__cmpWeightSum_isWider(Test__SmoothingOscilationBase):
    def test_1(self):
        self.sdrel1 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            weightThreshold = 0, weightThresholdMode = 'rel', weightSumContainerSize=2)
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdrel1)
        smoothing.weightContainerStd.pushBack(26)
        smoothing.weightContainerStd.pushBack(1)
        self.cmpPandas(
            obj_1 = smoothing.cmpWeightSum_isWider(metric=25.99, smoothingMetadata=self.sdrel1), 
            name_1 = 'metric_weight_sum_rel_1', 
            obj_2 = False)

    def test_2(self):
        self.sdrel2 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            weightThreshold = 0.5, weightThresholdMode = 'rel', weightSumContainerSize=2)
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdrel2)
        smoothing.weightContainerStd.pushBack(26)
        smoothing.weightContainerStd.pushBack(1)
        self.cmpPandas(
            obj_1 = smoothing.cmpWeightSum_isWider(metric=0.48, smoothingMetadata=self.sdrel2), 
            name_1 = 'metric_weight_sum_rel_2', 
            obj_2 = True)

    def test_3(self):
        self.sdrel3 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            weightThreshold = 0.5, weightThresholdMode = 'rel', weightSumContainerSize=2)
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdrel3)
        smoothing.weightContainerStd.pushBack(26)
        smoothing.weightContainerStd.pushBack(1)
        self.cmpPandas(
            obj_1 = smoothing.cmpWeightSum_isWider(metric=10, smoothingMetadata=self.sdrel3), 
            name_1 = 'metric_weight_sum_rel_3', 
            obj_2 = False)
        ###############################

    def test_4(self):
        self.sdabs1 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            weightThreshold = 0, weightThresholdMode = 'abs', weightSumContainerSize=2)
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdabs1)
        smoothing.weightContainerStd.pushBack(55)
        smoothing.weightContainerStd.pushBack(24)
        self.cmpPandas(
            obj_1 = smoothing.cmpWeightSum_isWider(metric=55, smoothingMetadata=self.sdabs1), 
            name_1 = 'metric_weight_sum_abs_1', 
            obj_2 = False)

    def test_5(self):
        self.sdabs2 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            weightThreshold = 0.5, weightThresholdMode = 'abs', weightSumContainerSize=2)
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdabs2)
        smoothing.weightContainerStd.pushBack(55)
        smoothing.weightContainerStd.pushBack(24)
        self.cmpPandas(
            obj_1 = smoothing.cmpWeightSum_isWider(metric=23.4, smoothingMetadata=self.sdabs2), 
            name_1 = 'metric_weight_sum_abs_2', 
            obj_2 = True)

    def test_6(self):
        self.sdabs3 = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(smoothingEndCheckType='wgsum',
            weightThreshold = 0.5, weightThresholdMode = 'abs', weightSumContainerSize=2)
        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=self.sdabs3)
        smoothing.weightContainerStd.pushBack(55)
        smoothing.weightContainerStd.pushBack(24)
        self.cmpPandas(
            obj_1 = smoothing.cmpWeightSum_isWider(metric=23.6, smoothingMetadata=self.sdabs3), 
            name_1 = 'metric_weight_sum_abs_3', 
            obj_2 = False)

class Test__canComputeWeights(Test__SmoothingOscilationBase):
    def test_1(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(
            lossWarmup = 1, lossPatience = 1, lossThreshold = 1e-4, lossThresholdMode = 'rel', startAt = 2
        )
        smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(dictionary=ut.init_weights_tensor, smoothingMetadata=smoothingMetadata)

        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        self.helperEpoch.epochNumber = 2
        self.helper.loss = torch.tensor(1.0)
        self.model.setConstWeights(weight=11, bias=13)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        self.helper.loss = torch.tensor(2.0)
        self.model.setConstWeights(weight=13, bias=15)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        self.helper.loss = torch.tensor(2.5)
        self.model.setConstWeights(weight=15, bias=17)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = True)

    def test_2(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(
            lossWarmup = 3, lossPatience = 2, lossThreshold = 0.5, lossThresholdMode = 'rel', lossContainerSize=2, startAt = 2
        )
        smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(dictionary=ut.init_weights_tensor, smoothingMetadata=smoothingMetadata)

        self.helperEpoch.epochNumber = 1
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        # lossWarmup
        #####################################

        self.helperEpoch.epochNumber = 2
        self.helper.loss = torch.tensor(1.0)
        self.model.setConstWeights(weight=11, bias=13)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        self.helper.loss = torch.tensor(2.0)
        self.model.setConstWeights(weight=13, bias=15)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)


        # lossPatience
        ###################################
        self.helper.loss = torch.tensor(2.0)
        self.model.setConstWeights(weight=13, bias=15)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        self.helper.loss = torch.tensor(2.0) # avg = 2.0
        self.model.setConstWeights(weight=13, bias=15)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        # lossThreshold - bad
        ##################################

        self.helper.loss = torch.tensor(1.5) # avg = 1.75 > 1.5 * 0.5
        self.model.setConstWeights(weight=15, bias=17)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        # lossThreshold - good
        ##################################

        self.helper.loss = torch.tensor(1) # avg = 1.25 < 1.5
        self.model.setConstWeights(weight=15, bias=17)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = True)

    def test_3(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationEWMA_Metadata(
            lossWarmup = 1, lossPatience = 1, lossThreshold = 1e-4, lossThresholdMode = 'rel',
        )
        smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(dictionary=ut.init_weights_tensor, smoothingMetadata=smoothingMetadata)

        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        self.helperEpoch.epochNumber = 2
        self.helper.loss = torch.tensor(1.0)
        self.model.setConstWeights(weight=11, bias=13)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        self.helper.loss = torch.tensor(2.0)
        self.model.setConstWeights(weight=13, bias=15)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)

        self.helper.loss = torch.tensor(2.5)
        self.model.setConstWeights(weight=15, bias=17)
        self.cmpPandas(
            obj_1 = smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
                modelMetadata=self.modelMetadata, metadata=self.metadata, smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = True)

class Test___isSmoothingGoodEnough__(Test__SmoothingOscilationBase):
    def test_1(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(weightSumContainerSize = 4,
            weightWarmup = 3, weightPatience = 2, weightThreshold = 0.1, weightThresholdMode = 'rel', startAt = 1
        )
        smoothing = dc.DefaultSmoothingOscilationGeneralizedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(dictionary=ut.init_weights_tensor, smoothingMetadata=smoothingMetadata)

        self.helperEpoch.epochNumber = 2
        smoothing.alwaysOn = True

        # weightWarmup
        ################################################
        self.model.setConstWeights(weight=10, bias=10)
        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        self.cmpPandas(
            obj_1 = smoothing.__isSmoothingGoodEnough__(helperEpoch=self.helperEpoch, helper=self.helper, 
                model=self.model, dataMetadata=self.dataMetadata, modelMetadata=self.modelMetadata, metadata=self.metadata, 
                smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)
        self.cmpPandas(obj_1 = smoothing._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=self.metadata), 
            name_1 = 'weight_sum', obj_2 = 100.0)
            
        self.model.setConstWeights(weight=10, bias=10)
        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        self.cmpPandas(
            obj_1 = smoothing.__isSmoothingGoodEnough__(helperEpoch=self.helperEpoch, helper=self.helper, 
                model=self.model, dataMetadata=self.dataMetadata, modelMetadata=self.modelMetadata, metadata=self.metadata, 
                smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)
        self.cmpPandas(obj_1 = smoothing._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=self.metadata), 
            name_1 = 'weight_sum', obj_2 = (100.0 + 100.0) / 2)


        self.model.setConstWeights(weight=20, bias=20)
        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        self.cmpPandas(
            obj_1 = smoothing.__isSmoothingGoodEnough__(helperEpoch=self.helperEpoch, helper=self.helper, 
                model=self.model, dataMetadata=self.dataMetadata, modelMetadata=self.modelMetadata, metadata=self.metadata, 
                smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)
        self.cmpPandas(obj_1 = smoothing._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=self.metadata), 
            name_1 = 'weight_sum', obj_2 = (100.0 + 100.0 + 200.0) / 3)

        # weightPatience
        ################################################
        self.model.setConstWeights(weight=10, bias=10)
        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        self.cmpPandas(
            obj_1 = smoothing.__isSmoothingGoodEnough__(helperEpoch=self.helperEpoch, helper=self.helper, 
                model=self.model, dataMetadata=self.dataMetadata, modelMetadata=self.modelMetadata, metadata=self.metadata, 
                smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)
        self.cmpPandas(obj_1 = smoothing._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=self.metadata), 
            name_1 = 'weight_sum', obj_2 = (100.0 + 100.0 + 200.0 + 100.0) / 4)

        self.model.setConstWeights(weight=10, bias=10)
        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        self.cmpPandas(
            obj_1 = smoothing.__isSmoothingGoodEnough__(helperEpoch=self.helperEpoch, helper=self.helper, 
                model=self.model, dataMetadata=self.dataMetadata, modelMetadata=self.modelMetadata, metadata=self.metadata, 
                smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)
        self.cmpPandas(obj_1 = smoothing._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=self.metadata), 
            name_1 = 'weight_sum', obj_2 = (100.0 + 100.0 + 200.0 + 100.0 + 100.0) / 5)

        # weightThreshold - bad
        # weightContainer: [120.0, 116.66667175292969, 133.3333282470703, 125.0] 
        # weightContainerStd: [12.268691795186637, 0.0, 15.713481628680341, 14.877974290304298]
        ################################################
        self.model.setConstWeights(weight=10, bias=10)
        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        self.cmpPandas(
            obj_1 = smoothing.__isSmoothingGoodEnough__(helperEpoch=self.helperEpoch, helper=self.helper, 
                model=self.model, dataMetadata=self.dataMetadata, modelMetadata=self.modelMetadata, metadata=self.metadata, 
                smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = False)
        self.cmpPandas(obj_1 = smoothing._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=self.metadata), 
            name_1 = 'weight_sum', obj_2 = (100.0 + 100.0 + 200.0 + 100.0 + 100.0 + 100.0) / 6)

        # weightThreshold - good
        # weightContainer: [120.0, 116.66667175292969, 142.85714721679688, 125.0] 
        # weightContainerStd: [12.268691795186637, 6.277712946345188, 15.713481628680341, 14.877974290304298]
        ################################################
        self.model.setConstWeights(weight=30, bias=30)
        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        self.cmpPandas(
            obj_1 = smoothing.__isSmoothingGoodEnough__(helperEpoch=self.helperEpoch, helper=self.helper, 
                model=self.model, dataMetadata=self.dataMetadata, modelMetadata=self.modelMetadata, metadata=self.metadata, 
                smoothingMetadata=smoothingMetadata), 
            name_1 = 'canComputeWeights', 
            obj_2 = True)
        self.cmpPandas(obj_1 = smoothing._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=self.metadata), 
            name_1 = 'weight_sum', obj_2 = (100.0 + 100.0 + 200.0 + 100.0 + 100.0 + 100.0 + 300.0) / 7)


class Test__sumAllWeights(Test__SmoothingOscilationBase):
    def test_1(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationEWMA_Metadata(lossWarmup=0, weightWarmup=1)

        smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())
        smoothing.countWeights = 1

        self.helperEpoch.epochNumber = 2
        smoothing.alwaysOn = True

        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        sumWg = smoothing._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=self.metadata)
        self.cmpPandas(sumWg, 'weight_sum', 58.0)

class Test__calcMean(Test__SmoothingOscilationBase):
    def test_DefaultSmoothingOscilationWeightedMean(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(2), smoothingEndCheckType='wgsum')

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())

        weights = dict(self.model.named_parameters())
        self.compareDictToNumpy(iterator=weights, numpyDict=ut.init_weights)


        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        smoothedWg = smoothing.weightsArray.array
        weights = dict(self.model.named_parameters())
        i = smoothedWg[0]
        self.compareDictToNumpy(iterator=weights, numpyDict=ut.init_weights)
        self.compareDictToNumpy(iterator=i, numpyDict=ut.init_weights)

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

    def test_DefaultSmoothingOscilationEWMA(self):
        modelMetadata = ut.TestModel_Metadata()
        model = ut.TestModel(modelMetadata)
        smoothingMetadata = dc.DefaultSmoothingOscilationEWMA_Metadata(movingAvgParam=0.5)

        smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())

        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)

        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictToNumpy(wg, ut.init_weights)
        self.compareDictToNumpy(smoothedWg, ut.init_weights)

        #############

        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)
        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictToNumpy(wg, ut.init_weights)
        self.compareDictToNumpy(smoothedWg, ut.init_weights)

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


class Test__DefaultSmoothingOscilationEWMA(Test__SmoothingOscilationBase, Test_DefaultSmoothing):
    def test__getSmoothedWeights__(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationEWMA_Metadata(movingAvgParam=0.5,
        lossContainerSize=3)

        self.helper.loss = torch.Tensor([1.0])

        smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())

        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, metadata=self.metadata, 
        smoothingMetadata=smoothingMetadata)
        self.compareDictToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata), {})
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, metadata=self.metadata, 
        smoothingMetadata=smoothingMetadata) # zapisanie wag
        self.compareDictToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata), ut.init_weights)

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

class Test_DefaultSmoothingSimpleMean(Test__SmoothingOscilationBase, Test_DefaultSmoothing):
    def setUp(self):
        super().setUp()
        self.smoothingMetadata = dc.Test_DefaultSmoothingSimpleMean_Metadata(batchPercentStart=0.01)
    
    def utils_checkSmoothedWeights(self, model, helperEpoch, dataMetadata, smoothing, smoothingMetadata, helper, metadata, w, b, sumW, sumB, count):
        # utils
        model.setConstWeights(weight=w, bias=b)
        w = (w+sumW)/count
        b = (b+sumB)/count
        wg = self.setWeightDict(w=w, b=b)
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, 
        metadata=self.metadata, smoothingMetadata=self.smoothingMetadata)
        self.compareDictToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=self.smoothingMetadata, metadata=self.metadata), numpyDict=wg)

    def test__isSmoothingGoodEnough__(self):
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
        self.compareDictToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=self.smoothingMetadata, metadata=self.metadata), numpyDict=ut.init_weights)

        self.utils_checkSmoothedWeights(model=self.model, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, smoothing=smoothing, 
        smoothingMetadata=self.smoothingMetadata, helper=self.helper, metadata=self.metadata, w=17, b=19, sumW=5, sumB=7, count=2)
        self.utils_checkSmoothedWeights(model=self.model, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, smoothing=smoothing, 
        smoothingMetadata=self.smoothingMetadata, helper=self.helper, metadata=self.metadata, w=23, b=29, sumW=5+17, sumB=7+19, count=3)
        self.utils_checkSmoothedWeights(model=self.model, helperEpoch=self.helperEpoch, dataMetadata=self.dataMetadata, smoothing=smoothing, 
        smoothingMetadata=self.smoothingMetadata, helper=self.helper, metadata=self.metadata, w=45, b=85, sumW=5+17+23, sumB=7+19+29, count=4) 


class Test_DefaultSmoothingOscilationWeightedMean(Test_DefaultSmoothing):
    def setUp(self):
        self.metadata = sf.Metadata()
        self.metadata.debugInfo = True
        self.metadata.logFolderSuffix = str(time.time())
        self.metadata.debugOutput = 'debug'
        self.metadata.prepareOutput()
        self.modelMetadata = ut.TestModel_Metadata()
        self.model = ut.TestModel(self.modelMetadata)
        self.helper = sf.TrainDataContainer()
        self.dataMetadata = dc.DefaultData_Metadata()
        self.helperEpoch = sf.EpochDataContainer()
        self.helperEpoch.trainTotalNumber = 3
        self.helperEpoch.maxTrainTotalNumber = 1

    def test__getSmoothedWeights__(self):
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationWeightedMean_Metadata(weightIter=dc.DefaultWeightDecay(2), smoothingEndCheckType='wgsum',
        lossContainerSize=3)

        self.helperEpoch.maxTrainTotalNumber = 1000

        smoothing = dc.DefaultSmoothingOscilationWeightedMean(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())
        self.helper.loss = torch.Tensor([1.0])

        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, 
        metadata=self.metadata, smoothingMetadata=smoothingMetadata)
        self.compareDictToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata), {})
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, 
        modelMetadata=None, metadata=self.metadata, smoothingMetadata=smoothingMetadata) # aby zapisać wagi
        self.compareDictToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata), ut.init_weights)

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
        self.compareDictToNumpy(iterator=weights, numpyDict=ut.init_weights)


        smoothing.calcMean(model=self.model, smoothingMetadata=smoothingMetadata)
        smoothedWg = smoothing.weightsArray.array
        weights = dict(self.model.named_parameters())
        i = smoothedWg[0]
        self.compareDictToNumpy(iterator=weights, numpyDict=ut.init_weights)
        self.compareDictToNumpy(iterator=i, numpyDict=ut.init_weights)

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
 
class Test_DefaultSmoothingOscilationEWMA(Test_DefaultSmoothing):
    def setUp(self):
        self.metadata = sf.Metadata()
        self.metadata.debugInfo = True
        self.metadata.logFolderSuffix = str(time.time())
        self.metadata.debugOutput = 'debug'
        self.metadata.prepareOutput()
        self.modelMetadata = ut.TestModel_Metadata()
        self.model = ut.TestModel(self.modelMetadata)
        self.helper = sf.TrainDataContainer()
        self.dataMetadata = dc.DefaultData_Metadata()
        self.helperEpoch = sf.EpochDataContainer()
        self.helperEpoch.trainTotalNumber = 3
        self.helperEpoch.maxTrainTotalNumber = 1000

    def test_calcMean(self):
        modelMetadata = ut.TestModel_Metadata()
        model = ut.TestModel(modelMetadata)
        smoothingMetadata = dc.DefaultSmoothingOscilationEWMA_Metadata(movingAvgParam=0.5)

        smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())

        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)

        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictToNumpy(wg, ut.init_weights)
        self.compareDictToNumpy(smoothedWg, ut.init_weights)

        #############

        smoothing.calcMean(model=model, smoothingMetadata=smoothingMetadata)
        wg = dict(model.named_parameters())
        smoothedWg = smoothing.weightsSum
        self.compareDictToNumpy(wg, ut.init_weights)
        self.compareDictToNumpy(smoothedWg, ut.init_weights)

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
        smoothingMetadata = dc.Test_DefaultSmoothingOscilationEWMA_Metadata(movingAvgParam=0.5,
        lossContainerSize=3)

        self.helper.loss = torch.Tensor([1.0])

        smoothing = dc.DefaultSmoothingOscilationEWMA(smoothingMetadata=smoothingMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=self.model.getNNModelModule().named_parameters())

        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, metadata=self.metadata, 
        smoothingMetadata=smoothingMetadata)
        self.compareDictToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata), {})
        smoothing(helperEpoch=self.helperEpoch, helper=self.helper, model=self.model, dataMetadata=self.dataMetadata, modelMetadata=None, metadata=self.metadata, 
        smoothingMetadata=smoothingMetadata) # zapisanie wag
        self.compareDictToNumpy(smoothing.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=self.metadata), ut.init_weights)

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
        self.modelMetadata = ut.TestModel_Metadata()
        self.model = ut.TestModel(self.modelMetadata)
        self.helper = sf.TrainDataContainer()
        self.smoothingMetadata = dc.Test_DefaultSmoothingSimpleMean_Metadata(batchPercentStart=0.01)
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
        self.compareDictToNumpy(iterator=smoothing.__getSmoothedWeights__(smoothingMetadata=self.smoothingMetadata, metadata=self.metadata), numpyDict=ut.init_weights)

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
    
