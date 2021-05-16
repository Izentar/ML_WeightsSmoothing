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

class VGG11Data_Metadata(dc.DefaultData_Metadata):
    def __init__(self):
        super().__init__()
        self.batchTrainSize = 4
        self.batchTestSize = 4


if(__name__ == '__main__'):
    torch.backends.cudnn.benchmark = True
    obj = models.vgg11()

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
