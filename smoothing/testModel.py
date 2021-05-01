import torch
import torchvision
import torch.optim as optim
import smoothingFramework as sf
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class TestModel_Metadata(sf.Model_Metadata):
    def __init__(self):
        sf.Model_Metadata.__init__(self)
        sf.SaveClass.__init__(self)
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.oscilationMax = 0.001

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return sf.Model_Metadata.trySave(self, metadata, sf.StaticData.MODEL_METADATA_SUFFIX, TestModel_Metadata.__name__, onlyKeyIngredients, temporaryLocation)

    def tryLoad(metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return SaveClass.tryLoad(metadata.fileNameLoad, StaticData.MODEL_METADATA_SUFFIX, TestModel_Metadata.__name__, temporaryLocation)

class TestData_Metadata(sf.Data_Metadata):
    def __init__(self):
        sf.Data_Metadata.__init__(self)
        sf.SaveClass.__init__(self)
        self.worker_seed = 841874
        
        self.train = True
        self.download = True
        self.pin_memoryTrain = False
        self.pin_memoryTest = False

        self.epoch = 1
        self.batchTrainSize = 4
        self.batchTestSize = 4

        # batch size * howOftenPrintTrain
        self.howOftenPrintTrain = 2000

class TestModel(sf.Model):
    def __init__(self, modelMetadata):
        sf.Model.__init__(self, modelMetadata)
        sf.SaveClass.__init__(self)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)

        self.to(modelMetadata.device)

    def forward(self, x):
        x = self.pool(F.hardswish(self.conv1(x)))
        x = self.pool(F.hardswish(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.hardswish(self.linear1(x))
        x = F.hardswish(self.linear2(x))
        x = self.linear3(x)
        return x

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return sf.Model.trySave(metadata, StaticData.MODEL_SUFFIX, TestModel.__name__, onlyKeyIngredients, temporaryLocation)

    def tryLoad(metadata, modelMetadata, onlyKeyIngredients = False, temporaryLocation = False):
        obj = SaveClass.tryLoad(metadata.fileNameLoad, StaticData.MODEL_SUFFIX, TestModel.__name__, temporaryLocation)
        obj.update(modelMetadata)
        return obj
    
    def update(self, modelMetadata):
        super().update(modelMetadata)
        self.optimizer = optim.SGD(self.parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)

class TestSmoothing(sf.Smoothing):
    def __init__(self):
        self.lossSum = 0.0
        self.lossCounter = 0
        self.lossList = []
        self.lossAverage = []

        self.numbOfBatchAfterSwitchOn = 2000

        self.flushLossSum = 1000

        self.sumWeights = {}
        self.previousWeights = {}
        # [torch.tensor(0.0) for x in range(100)] # add more to array than needed
        self.countWeights = 0
        self.counter = 0

        self.mainWeights = None

    def call(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata):
        self.counter += 1
        if(self.counter > self.numbOfBatchAfterSwitchOn):
            self.countWeights += 1
            helper.substract = {}
            with torch.no_grad():
                for key, arg in model.named_parameters():
                    self.sumWeights[key].add_(arg)
            for key, arg in model.named_parameters():
                helper.substract[key] = arg.sub(self.previousWeights[key])
                self.previousWeights[key].data.copy_(arg.data)

    def getWeights(self):
        average = {}
        for key, arg in self.sumWeights.items():
            average[key] = self.sumWeights[key] / self.countWeights
        return average

    def setDictionary(self, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        for key, values in dictionary:
            self.sumWeights[key] = torch.zeros_like(values, requires_grad=False)
            self.previousWeights[key] = torch.zeros_like(values, requires_grad=False)


class TestData(sf.Data):

    def __customizeState__(self, state):
        super().__customizeState__(state)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def setTransform(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        return self.transform

    def __prepare__(self, dataMetadata):
        self.setTransform()

        self.trainset = torchvision.datasets.CIFAR10(root='~/.data', train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(root='~/.data', train=False, download=True, transform=self.transform)
        self.trainSampler = sf.BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = sf.BaseSampler(len(self.testset), dataMetadata.batchTrainSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

    def __update__(self, dataMetadata):
        self.setTransform()

        self.trainset = torchvision.datasets.CIFAR10(root='~/.data', train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(root='~/.data', train=False, download=True, transform=self.transform)
        self.trainSampler = BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = BaseSampler(len(self.testset), dataMetadata.batchTrainSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.DATA_SUFFIX, TestData.__name__, onlyKeyIngredients, temporaryLocation)

    def tryLoad(metadata, dataMetadata, temporaryLocation = False):
        return sf.SaveClass.tryLoad(metadata.fileNameLoad, sf.StaticData.DATA_SUFFIX, TestData.__name__, temporaryLocation)

if(__name__ == '__main__'):
    sf.useDeterministic()
    stat = sf.modelRun(sf.Metadata, TestData_Metadata, TestModel_Metadata, TestData, TestModel, TestSmoothing)

    plt.plot(stat.trainLossArray)
    plt.xlabel('Train index')
    plt.ylabel('Loss')
    plt.show()