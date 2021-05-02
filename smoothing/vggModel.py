import torch
import torchvision
import torch.optim as optim
import smoothingFramework as sf
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class TestModel_Metadata(sf.Model_Metadata):
    def __init__(self):
        sf.Model_Metadata.__init__(self)
        self.device = 'cpu'
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.oscilationMax = 0.001

    def __str__(self):
        tmp_str = '\n/{} class\n-----------------------------------------------------------------------\n'.format(type(self).__name__)
        tmp_str += ('Learning rate:\t{}\n'.format(self.learning_rate))
        tmp_str += ('Momentum:\t{}\n'.format(self.momentum))
        tmp_str += ('-----------------------------------------------------------------------\nEnd Hyperparameters class\n')
        return tmp_str

class TestData_Metadata(sf.Data_Metadata):
    def __init__(self):
        sf.Data_Metadata.__init__(self)
        self.worker_seed = 8418748
        
        self.train = True
        self.download = True
        self.pin_memoryTrain = False
        self.pin_memoryTest = False

        self.epoch = 1
        self.batchTrainSize = 4
        self.batchTestSize = 4

        # batch size * howOftenPrintTrain
        self.howOftenPrintTrain = 2000

class TestModel(sf.PredefinedModel):
    def __init__(self, obj, modelMetadata):
        super().__init__(obj, modelMetadata)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.getNNModelModule().parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)
        #self.optimizer = optim.AdamW(self.parameters(), lr=modelMetadata.learning_rate)

        self.getNNModelModule().to(modelMetadata.device)

    def forward(self, x):
        x = self.pool(F.hardswish(self.conv1(x)))
        x = self.pool(F.hardswish(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.hardswish(self.linear1(x))
        x = F.hardswish(self.linear2(x))
        x = self.linear3(x)
        return x

    def __update__(self, modelMetadata):
        self.getNNModelModule().to(modelMetadata.device)
        self.optimizer = optim.SGD(self.parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)

class TestSmoothing(sf.Smoothing):
    def __init__(self):
        sf.Smoothing.__init__(self)
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
                for key, arg in model.getNNModelModule().named_parameters():
                    self.sumWeights[key].add_(arg)
            for key, arg in model.getNNModelModule().named_parameters():
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
    def __init__(self):
        super().__init__()

    def __customizeState__(self, state):
        super().__customizeState__(state)

    def __setstate__(self, state):
        super().__setstate__(state)

    def __setInputTransform__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        return self.transform

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__()

        self.trainset = torchvision.datasets.CIFAR10(root=sf.StaticData.DATA_PATH, train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(root=sf.StaticData.DATA_PATH, train=False, download=True, transform=self.transform)
        self.trainSampler = sf.BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = sf.BaseSampler(len(self.testset), dataMetadata.batchTrainSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

    def __update__(self, dataMetadata):
        self.__prepare__(dataMetadata)

    def __beforeTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helper.tmp_sumLoss = 0.0

    def __beforeTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helper.inputs, helper.labels = helper.inputs.to(modelMetadata.device), helper.labels.to(modelMetadata.device)
        model.optimizer.zero_grad()

    def __afterTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helper.tmp_sumLoss += helper.loss
        #if(helper.batchNumber % 32 == 0 and helper.batchNumber != 0):
        #    helperEpoch.statistics.addLoss(helper.tmp_sumLoss / 32)
        #    helper.tmp_sumLoss = 0.0

        helperEpoch.statistics.addLoss(helper.loss)

        if(bool(metadata.debugInfo) and dataMetadata.howOftenPrintTrain is not None and helper.batchNumber % dataMetadata.howOftenPrintTrain == 0):
            calcLoss, current = helper.loss.item(), helper.batchNumber * len(helper.inputs)
            metadata.stream.print(f"loss: {calcLoss:>7f}  [{current:>5d}/{helper.size:>5d}]")

            averageWeights = smoothing.getWeights()
            if(bool(averageWeights)):
                averKey = list(averageWeights.keys())[-1]
                metadata.stream.printTo(['debug', 'bash'], f"Average: {averageWeights[averKey]}")
            
            metadata.stream.print(f"loss: {calcLoss:>7f}  [{current:>5d}/{helper.size:>5d}]")
            if(helper.diff is None):
                metadata.stream.print(f"No weight difference")
            else:
                diffKey = list(helper.diff.keys())[-1]
                metadata.stream.printTo(['debug', 'bash'], f"Weight difference: {helper.diff[diffKey]}")
                metadata.stream.print(f"Weight difference of last layer average: {helper.diff[diffKey].sum() / helper.diff[diffKey].numel()} :: was divided by: {helper.diff[diffKey].numel()}")
                metadata.stream.print('################################################')

    def __afterTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        super().__afterTrainLoop__(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing)

        if(helper.diff is not None):
            diffKey = list(helper.diff.keys())[-1]
            metadata.stream.printFormated("trainLoop;\nAverage train time;Loop train time;Weight difference of last layer average;divided by;")
            metadata.stream.printFormated(f"{helper.timer.getAverage()};{helper.loopTimer.getDiff()};{helper.diff[diffKey].sum() / helper.diff[diffKey].numel()};{helper.diff[diffKey].numel()}")



    def __beforeTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        metadata.stream.printFormated("testLoop;\nAverage test time;Loop test time;Accuracy;Avg loss")

    def __beforeTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helper.inputs = helper.inputs.to(modelMetadata.device)
        helper.labels = helper.labels.to(modelMetadata.device)

    def __afterTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helper.test_loss += model.loss_fn(helper.pred, helper.labels).item()
        helper.test_correct += (helper.pred.argmax(1) == helper.labels).type(torch.float).sum().item()

    def __afterTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helper.test_loss /= helper.size
        helper.test_correct /= helper.size
        metadata.stream.print(f"Test summary: \n Accuracy: {(100*helper.test_correct):>0.1f}%, Avg loss: {helper.test_loss:>8f}")
        metadata.stream.print(f" Average test time ({helper.timer.getUnits()}): {helper.timer.getAverage()}")
        metadata.stream.print(f" Loop test time ({helper.timer.getUnits()}): {helper.loopTimer.getDiff()}")
        metadata.stream.print("")
        metadata.stream.printFormated(f"{helper.timer.getAverage()};{helper.loopTimer.getDiff()};{(helper.test_correct):>0.0001f};{helper.test_loss:>8f}")


    def __epoch__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', 
        modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        if(metadata.shouldTrain()):
            self.trainLoop(model, helperEpoch, dataMetadata, modelMetadata, metadata, smoothing)
        
        if(sf.enabledSaveAndExit()):
            return 

        with torch.no_grad():
            if(metadata.shouldTest()):
                metadata.stream.write("Plain weights, ")
                metadata.stream.writeFormated("Plain weights;")
                self.testLoop(helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing)
                smoothing.saveMainWeight(model)
                if(smoothing.getWeights()):
                    model.setWeights(smoothing.getWeights())
                    metadata.stream.write("Smoothing weights, ")
                    metadata.stream.writeFormated("Smoothing weights;")
                    self.testLoop(helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing)
                else:
                    print('Smoothing is not enabled. Test does not executed.')
            # model.linear1.weight = torch.nn.parameter.Parameter(model.average)
            # model.linear1.weight = model.average


if(__name__ == '__main__'):
    obj = models.vgg11()
    #sf.useDeterministic()
    #sf.modelDetermTest(sf.Metadata, TestData_Metadata, TestModel_Metadata, TestData, VGG16Model, TestSmoothing)
    stat = sf.modelRun(sf.Metadata, TestData_Metadata, TestModel_Metadata, TestData, TestModel, TestSmoothing, obj)

    plt.plot(stat.trainLossArray)
    plt.xlabel('Train index')
    plt.ylabel('Loss')
    plt.show()