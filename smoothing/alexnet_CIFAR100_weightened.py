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

if(__name__ == '__main__'):
    sf.StaticData.TEST_MODE = True
    torch.backends.cudnn.benchmark = True
    obj = models.alexnet()

    #sf.useDeterministic()
    #sf.modelDetermTest(sf.Metadata, DefaultData_Metadata, DefaultModel_Metadata, DefaultData, VGG16Model, DefaultSmoothing)
    stat = sf.modelRun(sf.Metadata, dc.DefaultData_Metadata, dc.DefaultModel_Metadata, dc.DefaultDataMNIST, dc.DefaultModelPredef, dc.DefaultSmoothingOscilationWeightedMean, 
        obj, 
        load=False
        )

    #sf.plot([stat.logFolder + '/statLossTest.csv', stat.logFolder + '/statLossTestSmoothing.csv'])

    #plt.plot(stat.trainLossArray)
    #plt.xlabel('Train index')
    #plt.ylabel('Loss')
    #plt.show()
