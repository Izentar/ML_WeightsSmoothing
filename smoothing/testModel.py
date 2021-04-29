import torch
import torchvision
import torch.optim as optim
import smoothingFramework as sf

class TestModel_Metadata(sf.Model_Metadata, sf.SaveClass):
    def __init__(self):
        super().__init__()
        self.momentum = 0.9

    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.MODEL_METADATA_SUFFIX, TestModel_Metadata.__name__, temporaryLocation)

    def tryLoad(metadata, temporaryLocation = False):
        return SaveClass.tryLoad(metadata.fileNameLoad, StaticData.MODEL_METADATA_SUFFIX, TestModel_Metadata.__name__, temporaryLocation)

class TestData_Metadata(sf.Data_Metadata, sf.SaveClass):


class TestModel(sf.Model, sf.SaveClass):
    def __init__(self, modelMetadata):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)

    def forward(self, x):
        x = self.pool(F.hardswish(self.conv1(x)))
        x = self.pool(F.hardswish(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.hardswish(self.linear1(x))
        x = F.hardswish(self.linear2(x))
        x = self.linear3(x)
        return x

    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.MODEL_SUFFIX, TestModel.__name__, temporaryLocation)

    def tryLoad(metadata, modelMetadata, temporaryLocation = False):
        obj = SaveClass.tryLoad(metadata.fileNameLoad, StaticData.MODEL_SUFFIX, TestModel.__name__, temporaryLocation)
        obj.update(modelMetadata)
        return obj
    
    def update(self, modelMetadata):
        super().update(modelMetadata)
        self.optimizer = optim.SGD(self.parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)

class TestSmoothing(sf.Smoothing, sf.SaveClass):
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

    def saveMainWeight(self, model):
        self.mainWeights = model.getWeights()

    def addToAverageWeights(self, model):
        with torch.no_grad():
            for key, arg in model.named_parameters():
                self.sumWeights[key].add_(arg)
        
    def getStateDict(self):
        average = {}
        for key, arg in self.sumWeights.items():
            average[key] = self.sumWeights[key] / self.countWeights
        return average
        
    def lateStartAverageWeights(self, model):
        self.counter += 1
        if(self.countWeights > self.numbOfBatchAfterSwitchOn):
            self.countWeights += 1
            return self.addToAverageWeights(model)
        return dict(model)

    def setDictionary(self, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        for key, values in dictionary:
            self.sumWeights[key] = torch.zeros_like(values, requires_grad=False)
            self.previousWeights[key] = torch.zeros_like(values, requires_grad=False)


    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

class TestData(sf.Data, sf.SaveClass):

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

    def prepare(self, dataMetadata):
        self.setTransform()

        self.trainset = torchvision.datasets.CIFAR100(root='~/.data', train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR100(root='~/.data', train=False, download=True, transform=self.transform)
        self.trainSampler = BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = BaseSampler(len(self.testset), dataMetadata.batchTrainSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest)

    def update(self, dataMetadata):
        self.setTransform()
        self.trainset = torchvision.datasets.CIFAR100(root='~/.data', train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR100(root='~/.data', train=False, download=True, transform=self.transform)
        if(self.trainset is not None or self.trainSampler is not None):
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain)

        if(self.testset is not None or self.testSampler is not None):
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest)

    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.DATA_SUFFIX, TestData.__name__, temporaryLocation)

    def tryLoad(metadata, dataMetadata, temporaryLocation = False):
        return SaveClass.tryLoad(metadata.fileNameLoad, StaticData.DATA_SUFFIX, TestData.__name__, temporaryLocation)


