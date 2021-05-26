import torch
import torchvision
import torch.optim as optim
from framework import smoothingFramework as sf
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from framework import defaultClasses as dc

class GoogleNetData(dc.DefaultData):
    def __train__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata, metadata: 'Metadata', smoothing: 'Smoothing'):
        # forward + backward + optimize
        outputs = model.getNNModelModule()(helper.inputs)
        helper.loss = model.loss_fn(outputs.logits, helper.labels)
        del outputs
        helper.loss.backward()
        model.optimizer.step()

        # run smoothing
        smoothing(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata)

    def __test__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):       
        helper.pred = model.getNNModelModule()(helper.inputs)
        helper.test_loss = model.loss_fn(helper.pred.logits, helper.labels).item()



if(__name__ == '__main__'):
    torch.backends.cudnn.benchmark = True
    obj = models.GoogLeNet(init_weights=True)

    #sf.useDeterministic()
    #sf.modelDetermTest(sf.Metadata, DefaultData_Metadata, DefaultModel_Metadata, DefaultData, VGG16Model, DefaultSmoothing)
    stat = sf.modelRun(sf.Metadata, dc.DefaultData_Metadata, dc.DefaultModel_Metadata, dc.DefaultDataMNIST, dc.DefaultModelSimpleConv, dc.DefaultSmoothingOscilationGeneralizedMean, 
        obj, 
        load=False
        )

    #plt.plot(stat.trainLossArray)
    #plt.xlabel('Train index')
    #plt.ylabel('Loss')
    #plt.show()
