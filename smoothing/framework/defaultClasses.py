import torch
import torchvision
import torch.optim as optim
from framework import smoothingFramework as sf
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DefaultModel_Metadata(sf.Model_Metadata):
    def __init__(self):
        sf.Model_Metadata.__init__(self)
        self.device = 'cuda:0'
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.oscilationMax = 0.001

    def __str__(self):
        tmp_str = '\n/{} class\n-----------------------------------------------------------------------\n'.format(type(self).__name__)
        tmp_str += ('Learning rate:\t{}\n'.format(self.learning_rate))
        tmp_str += ('Momentum:\t{}\n'.format(self.momentum))
        tmp_str += ('-----------------------------------------------------------------------\nEnd Hyperparameters class\n')
        return tmp_str

class DefaultData_Metadata(sf.Data_Metadata):
    def __init__(self):
        super().__init__()
        self.worker_seed = 8418748
        
        self.train = True
        self.download = True
        self.pin_memoryTrain = False
        self.pin_memoryTest = False

        self.epoch = 1
        self.batchTrainSize = 16
        self.batchTestSize = 16

        # batch size * howOftenPrintTrain
        self.howOftenPrintTrain = 2000

class DefaultModel(sf.PredefinedModel):
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
        self.optimizer = optim.SGD(self.getNNModelModule().parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)

class DefaultSmoothing(sf.Smoothing):
    def __init__(self):
        sf.Smoothing.__init__(self)
        self.lossSum = 0.0
        self.lossCounter = 0
        self.lossList = []
        self.lossAverage = []

        self.numbOfBatchAfterSwitchOn = None
        if(sf.test_mode.isActivated()):
            self.numbOfBatchAfterSwitchOn = 10
        else:
            self.numbOfBatchAfterSwitchOn = 1000 # dla 50000 / 32 ~= 1500

        self.sumWeights = {}
        self.previousWeights = {}
        # [torch.tensor(0.0) for x in range(100)] # add more to array than needed
        self.countWeights = 0
        self.counter = 0

        self.mainWeights = None

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata):
        super().__call__(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata)
        self.counter += 1
        if(self.counter > self.numbOfBatchAfterSwitchOn):
            self.countWeights += 1
            if(hasattr(helper, 'substract')):
                del helper.substract
            helper.substract = {}
            with torch.no_grad():
                for key, arg in model.getNNModelModule().named_parameters():
                    cpuArg = arg.to('cpu')
                    self.sumWeights[key].to('cpu').add_(cpuArg)
                    #helper.substract[key] = arg.sub(self.previousWeights[key])
                    helper.substract[key] = self.previousWeights[key].sub_(cpuArg).multiply_(-1)
                    self.previousWeights[key].detach().copy_(cpuArg.detach())

    def __getSmoothedWeights__(self):
        average = super().__getSmoothedWeights__()
        if(average is not None):
            return average
        average = {}
        if(self.countWeights == 0):
            return average
        for key, arg in self.sumWeights.items():
            average[key] = self.sumWeights[key].to('cpu') / self.countWeights
        torch.cuda.empty_cache()
        return average

    def __setDictionary__(self, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(dictionary)
        with torch.no_grad():
            for key, values in dictionary:
                self.sumWeights[key] = torch.zeros_like(values, requires_grad=False, device='cpu')
                self.previousWeights[key] = torch.zeros_like(values, requires_grad=False, device='cpu')

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['previousWeights']
            del state['countWeights']
            del state['counter']
            del state['mainWeights']
            del state['sumWeights']
            del state['enabled']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.previousWeights = {}
            self.countWeights = 0
            self.counter = 0
            self.mainWeights = None
            self.sumWeights = {}
            self.enabled = False

class DefaultData(sf.Data):
    def __init__(self):
        super().__init__()

    def __customizeState__(self, state):
        super().__customizeState__(state)

    def __setstate__(self, state):
        super().__setstate__(state)

    def __setInputTransform__(self):
        ''' self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        '''
        self.trainTransform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.testTransform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__()

        #self.trainset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform)
        #self.testset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform)
        self.trainset = torchvision.datasets.CIFAR10(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform)
        self.testset = torchvision.datasets.CIFAR10(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform)

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
        metadata.stream.print("Loss", ['statLossTrain'])

    def __beforeTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helper.inputs, helper.labels = helper.inputs.to(modelMetadata.device), helper.labels.to(modelMetadata.device)
        model.optimizer.zero_grad()

    def __afterTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        with torch.no_grad():
            helper.tmp_sumLoss += helper.loss.item()

            # helperEpoch.statistics.addLoss(helper.loss.item()) # nie, bo za dużo pamięci dla dużego modelu zje. Wizualizacja tylko z csv
            metadata.stream.print(helper.loss.item(), ['statLossTrain'])

            if(bool(metadata.debugInfo) and dataMetadata.howOftenPrintTrain is not None and (helper.batchNumber % dataMetadata.howOftenPrintTrain == 0 or sf.test_mode.isActivated())):
                sf.DefaultMethods.printLoss(metadata, helper)
        
                #averageWeights = smoothing.__getSmoothedWeights__()
                
                sf.DefaultMethods.printWeightDifference(metadata, helper)

    def __afterTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        super().__afterTrainLoop__(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing)
        with torch.no_grad():
            if(helper.diff is not None):
                diffKey = list(helper.diff.keys())[-1]
                metadata.stream.print("\n\ntrainLoop;\nAverage train time;Loop train time;Weight difference of last layer average;divided by;", ['stat'])
                metadata.stream.print(f"{helper.timer.getAverage()};{helper.loopTimer.getDiff()};{helper.diff[diffKey].sum() / helper.diff[diffKey].numel()};{helper.diff[diffKey].numel()}", ['stat'])
                del diffKey

    def __beforeTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        metadata.stream.print("\n\ntestLoop;\nAverage test time;Loop test time;Accuracy;Avg loss", ['stat'])
        metadata.stream.print('\n\nTest loss', ['statLossTest'])

    def __beforeTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helper.inputs = helper.inputs.to(modelMetadata.device)
        helper.labels = helper.labels.to(modelMetadata.device)

    def __afterTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        super().__afterTest__(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing)
        metadata.stream.print(helper.test_loss, ['statLossTest'])

    def __afterTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        lossRatio = helper.testLossSum / helper.predSizeSum
        correctRatio = helper.testCorrectSum / helper.predSizeSum
        metadata.stream.print(f"\nTest summary: \n Accuracy: {(100*correctRatio):>0.1f}%, Avg loss: {lossRatio:>8f}", ['model:0'])
        metadata.stream.print(f" Average test execution time in a loop ({helper.timer.getUnits()}): {helper.timer.getAverage():>3f}", ['model:0'])
        metadata.stream.print(f" Time to complete the entire loop ({helper.timer.getUnits()}): {helper.loopTimer.getDiff():>3f}\n", ['model:0'])
        metadata.stream.print(f"{helper.timer.getAverage()};{helper.loopTimer.getDiff()};{(correctRatio):>0.0001f};{lossRatio:>8f}", ['stat'])

    def __beforeEpochLoop__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        metadata.stream.open("formatedLog", 'statLossTrain', 'statLossTrain')
        metadata.stream.open("formatedLog", 'statLossTest', 'statLossTest')

    def __epoch__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', 
        modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        if(metadata.shouldTrain()):
            self.trainLoop(model, helperEpoch, dataMetadata, modelMetadata, metadata, smoothing)
        
        if(sf.enabledSaveAndExit()):
            return 

        with torch.no_grad():
            if(metadata.shouldTest()):
                #metadata.stream.write("Plain weights, ")
                #metadata.stream.write("Plain weights;", 'stat')
                self.testLoop(helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing)
                smoothing.saveWeights(model.getNNModelModule().named_parameters(), 'main')
                wg = smoothing.__getSmoothedWeights__()
                if(wg):
                    metadata.stream.print("Smoothing:", 'statLossTest')
                    model.setWeights(wg)
                    #metadata.stream.write("Smoothing weights, ")
                    #metadata.stream.write("Smoothing weights;", 'stat')
                    self.testLoop(helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing)
                    model.setWeights(smoothing.getWeights('main'))
                else:
                    sf.Output.printBash('Smoothing is not enabled. Test does not executed.', 'info')
            # model.linear1.weight = torch.nn.parameter.Parameter(model.average)
            # model.linear1.weight = model.average


if(__name__ == '__main__'):
    with sf.test_mode():
        torch.backends.cudnn.benchmark = True
        obj = models.alexnet(pretrained=True)

        #sf.useDeterministic()
        #sf.modelDetermTest(sf.Metadata, DefaultData_Metadata, DefaultModel_Metadata, DefaultData, VGG16Model, DefaultSmoothing)
        stat = sf.modelRun(sf.Metadata, DefaultData_Metadata, DefaultModel_Metadata, DefaultData, DefaultModel, DefaultSmoothing, obj)

        #plt.plot(stat.trainLossArray)
        #plt.xlabel('Train index')
        #plt.ylabel('Loss')
        #plt.show()




