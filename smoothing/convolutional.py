import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys, getopt
from os.path import expanduser
import statistics
import signal

from playsound import playsound

SAVE_AND_EXIT_FLAG = False

def saveWorkAndExit(signumb, frame):
    global SAVE_AND_EXIT_FLAG
    SAVE_AND_EXIT_FLAG = True
    print('Ending and saving model')
    return

def recSign(signumb, frame):
    exit(2)

signal.signal(signal.SIGTSTP, saveWorkAndExit)

signal.signal(signal.SIGINT, recSign)
    
class Hiperparameters:
    def __init__(self):
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.oscilationMax = 0.001

    def __str__(self):
        tmp_str = '\n/Hiperparameters class\n-----------------------------------------------------------------------\n'
        tmp_str += ('Learning rate:\t{}\n'.format(self.learning_rate))
        tmp_str += ('Momentum:\t{}\n'.format(self.momentum))
        tmp_str += ('-----------------------------------------------------------------------\nEnd Hiperparameters class\n')
        return tmp_str

class MetaData:
    def __init__(self):
        self.PATH = expanduser("~") + '/.data/models/'
        self.MODEL_SUFFIX = '.pt'
        self.METADATA_SUFFIX = '.mdat'
        self.DATA_SUFFIX = '.dat'
        self.epoch = 1
        self.batchTrainSize = 4
        self.batchTestSize = 4
        self.hiperparam = Hiperparameters()

        self.fileNameSave = None
        self.fileNameLoad = None
        self.device = 'cpu'
        self.pin_memoryTrain = False
        self.pin_memoryTest = False

        self.testFlag = False
        self.trainFlag = False

        self.debugInfo = False
        self.modelOutput = None
        self.debugOutput = None
        self.stream = None
        self.bashFlag = False
        self.name = None
        
        # batch size * howOftenPrintTrain
        self.howOftenPrintTrain = 2000

    def __str__(self):
        tmp_str = ('\n/MetaData class\n-----------------------------------------------------------------------\n')
        tmp_str += ('Save path:\t{}\n'.format(self.PATH + self.fileNameSave if self.fileNameSave is not None else 'Not set'))
        tmp_str += ('Load path:\t{}\n'.format(self.PATH + self.fileNameLoad if self.fileNameLoad is not None else 'Not set'))
        tmp_str += ('Number of epochs:\t{}\n'.format(self.epoch))
        tmp_str += ('Batch train size:\t{}\n'.format(self.batchTrainSize))
        tmp_str += ('Batch test size:\t{}\n'.format(self.batchTestSize))
        tmp_str += ('Used device:\t{}\n'.format(self.device))
        tmp_str += ('Pin memory train:\t{}\n'.format(self.pin_memoryTrain))
        tmp_str += ('Pin memory test:\t{}\n'.format(self.pin_memoryTest))
        tmp_str += str(self.hiperparam)
        tmp_str += ('Test flag:\t{}\n'.format(self.testFlag))
        tmp_str += ('Train flag:\t{}\n'.format(self.trainFlag))
        tmp_str += ('-----------------------------------------------------------------------\nEnd MetaData class\n')
        return tmp_str

    def checkCUDA(string):
        return string.startswith('cuda')

    def trySelectCUDA(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if(self.debugInfo):
            print('Using {} torch CUDA device version\nCUDA avaliable: {}\nCUDA selected: {}'.format(torch.version.cuda, torch.cuda.is_available(), self.device == 'cuda'))
        return self.device

    def selectCPU(self):
        self.device = 'cpu'
        if(self.debugInfo):
            print('Using {} torch CUDA device version\nCUDA avaliable: {}\nCUDA selected: False'.format(torch.version.cuda, torch.cuda.is_available()))
        return self.device

    def tryPinMemoryTrain(self):
        if self.device == 'cpu': self.trySelectCUDA()
        if(MetaData.checkCUDA(self.device)):
            self.pin_memoryTrain = True
        if(self.debugInfo):
            print('Train data pinned to GPU: {}'.format(self.pin_memoryTrain))
        return self.pin_memoryTrain

    def tryPinMemoryTest(self):
        if self.device == 'cpu': self.trySelectCUDA()
        if(MetaData.checkCUDA(self.device)):
            self.pin_memoryTest = True
        if(self.debugInfo):
            print('Test data pinned to GPU: {}'.format(self.pin_memoryTest))
        return self.pin_memoryTest

    def tryPinMemoryAll(self):
        return self.tryPinMemoryTrain(), self.tryPinMemoryTest()

    def onOff(arg):
        if arg == 'on' or arg == 'True' or arg == 'true':
            return True
        elif arg == 'off' or arg == 'False' or arg == 'false':
            return False
        else:
            return None

    def printStartNewModel(self):
        if(self.stream is None):
            self.stream = Output()
        if(self.name is not None):
            self.stream.print(f"Starting new model: {self.name}")
        else:
            self.stream.print(f"Starting new model without name")

    def exitError(help):
        print(help) 
        sys.exit(2)

    def prepareOutput(self):
        if(self.stream is None):
            self.stream = Output()

        if(self.debugInfo == True):
            if(self.debugOutput is not None):
                self.stream.open('debug', self.debugOutput)
        if(self.modelOutput is not None):
            self.stream.open('model', self.modelOutput)
        if(self.bashFlag == True):
            self.stream.open('bash')

    def commandLineArg(self, argv):
        help = 'Help:\n'
        help += os.path.basename(__file__) + ' -h <help> [-s,--save] <file name to save> [-l,--load] <file name to load>'

        shortOptions = 'hs:l:d'
        longOptions = [
            'save=', 'load=', 'test=', 'train=', 'pinTest=', 'pinTrain=', 'debug', 
            'debugOutput=',
            'modelOutput=',
            'bashOutput=',
            'name='
            ]

        try:
            opts, args = getopt.getopt(argv, shortOptions, longOptions)
        except getopt.GetoptError:
            MetaData.exitError(help)

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print(help)
                sys.exit()
            elif opt in ('-s', '--save'):
                metadata.fileNameSave = arg
            elif opt in ('-l', '--load'):
                metadata.fileNameLoad = arg
            elif opt in ('--test'):
                boolean = MetaData.onOff(arg)
                self.testFlag = boolean if boolean is not None else MetaData.exitError(help)
            elif opt in ('--train'):
                boolean = MetaData.onOff(arg)
                self.trainFlag = boolean if boolean is not None else MetaData.exitError(help)
            elif opt in ('--pinTest'):
                boolean = MetaData.onOff(arg)
                self.tryPinMemoryTest() if boolean is not None else MetaData.exitError(help)
            elif opt in ('--pinTrain'):
                boolean = MetaData.onOff(arg)
                self.tryPinMemoryTrain() if boolean is not None else MetaData.exitError(help)
            elif opt in ('-d', '--debug'):
                self.debugInfo = True
            elif opt in ('--debugOutput'):
                self.debugOutput = arg # debug output file path
            elif opt in ('--modelOutput'):
                self.modelOutput = arg # model output file path
            elif opt in ('--bashOutput'):
                boolean = MetaData.onOff(arg)
                self.bashFlag = boolean if boolean is not None else MetaData.exitError(help)
            elif opt in ('--name'):
                self.name = arg

        if(self.modelOutput is None):
            self.modelOutput = 'default.log'

        if(self.debugOutput is None):
            self.debugOutput = 'default.log'
        
        if(self.fileNameLoad is not None):
            return self.tryLoad()
        return True

    def createDump(self):
        return {
                'epoch': self.epoch,
                'device': self.device
            }

    def loadFromDump(self, dump):
        self.epoch = dump['epoch']
        self.device = dump['device']

    def tryLoad(self):
        path = self.PATH + self.fileNameLoad + self.METADATA_SUFFIX
        if self.fileNameLoad is not None and os.path.exists(path):
            dump = torch.load(path)
            self.loadFromDump(dump)
            print('Metadata loaded successfully')
            return True
        print('Metadata load failure')
        return False

    def trySave(self):
        if self.fileNameSave is not None and os.path.exists(self.PATH):
            path = self.PATH + self.fileNameSave + self.METADATA_SUFFIX
            torch.save(self.createDump(), path)
            print('Metadata saved successfully')
            return True
        print('Metadata save failure')
        return False

class Output:
    def __init__(self):
        self.debugF = None
        self.modelF = None
        self.debugPath = None
        self.modelPath = None
        self.bash = False

    def open(self, outputType, path = None):
            if(outputType != 'debug' and outputType != 'model' and outputType != 'bash'):
                raise Exception("unknown command")

            if(outputType == 'bash'):
                self.bash = True

            # if you want to open file with other path
            if(outputType == 'debug' and path != self.debugPath and self.debugPath is not None and path is not None):
                self.debugF.close()
                self.debugPath = None
                self.debugF = None
            elif(outputType == 'model' and path != self.modelPath and self.modelPath is not None and path is not None):
                self.modelF.close()
                self.modelPath = None
                self.modelF = None

            # if file is already open in different outputType
            if(outputType == 'debug' and path is not None and path == self.modelPath):
                self.debugPath = path
                self.debugF = self.modelF
            elif(outputType == 'model' and path is not None and path == self.debugPath):
                self.modelPath = path
                self.modelF = self.debugF

            # if file was not opened
            if(outputType == 'debug' and path is not None and self.debugPath is None):
                self.debugF = open(path + ".log", 'a')
                self.debugPath = path
            elif(outputType == 'model' and path is not None and self.modelPath is None):
                self.modelF = open(path + ".log", 'a')
                self.modelPath = path

    def write(self, arg):
        if(self.bash is True):
            print(arg, end='')

        if(self.debugF is self.modelF and self.debugF is not None):
            self.debugF.write(arg)
        elif(self.debugF is not None):
            self.debugF.write(arg)
        elif(self.modelF is not None):
            self.modelF.write(arg)

    def print(self, arg):
        if(self.bash is True):
            print(arg)

        if(self.debugF is self.modelF and self.debugF is not None):
            self.debugF.write(arg + '\n')
            return
        if(self.debugF is not None):
            self.debugF.write(arg + '\n')
        if(self.modelF is not None):
            self.modelF.write(arg + '\n')

    def writeTo(self, outputType, arg):
        if self.bash == True:
            print(arg, end='')

        for t in outputType:
            if t == 'debug' and self.debugF is not None:
                self.debugF.write(arg)
            if t == 'model' and self.modelF is not None:
                self.modelF.write(arg)

    def printTo(self, outputType, arg):
        if self.bash == True:
            print(arg)

        for t in outputType:
            if t == 'debug' and self.debugF is not None:
                self.debugF.write(arg + '\n')
            if t == 'model' and self.modelF is not None:
                self.modelF.write(arg + '\n')

    def __del__(self):
        if(self.debugF is not None):
            self.debugF.close()
        if(self.modelF is not None): 
            self.modelF.close()# if closed this do nothing

class Data:
    def __init__(self):
        self.trainset = None
        self.trainloader = None
        self.testset = None
        self.testloader = None
        self.transform = None
        self.bindedMetadata = None

        self.batchNumbTrain = 0
        self.batchNumbTest = 0
        self.epochNumb = 0

    def __str__(self):
        tmp_str = ('\n/Data class\n-----------------------------------------------------------------------\n')
        tmp_str += ('Is trainset set:\t{}\n'.format(self.trainset is not None))
        tmp_str += ('Is trainloader set:\t{}\n'.format(self.trainloader is not None))
        tmp_str += ('Is testset set:\t\t{}\n'.format(self.testset is not None))
        tmp_str += ('Is testloader set:\t{}\n'.format(self.testloader is not None))
        tmp_str += ('Is transform set:\t{}\n'.format(self.transform is not None))
        tmp_str += ('-----------------------------------------------------------------------\nEnd Data class\n')
        return tmp_str

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['bindedMetadata']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def setTrainData(self):
        self.trainset = torchvision.datasets.CIFAR10(root='~/.data', train=True, download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.bindedMetadata.batchTrainSize,
                                          shuffle=True, num_workers=2, pin_memory=self.bindedMetadata.pin_memoryTrain)
        return self.trainset, self.trainloader

    def setTestData(self):
        self.testset = torchvision.datasets.CIFAR10(root='~/.data', train=False, download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.bindedMetadata.batchTestSize,
                                         shuffle=False, num_workers=2, pin_memory=self.bindedMetadata.pin_memoryTest)
        return self.testset, self.testloader

    def setTransform(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        return self.transform

    def setAll(self):
        '''
        Set transform of the input data and set train and test data.
        Returns transform function, trainset, trainloader, testset, testloader
        '''
        return self.setTransform(), self.setTrainData(), self.setTestData()

    def trainLoop(self, model, smoothing):
        size = len(self.trainloader.dataset)
        model.train()
        model.bindedMetadata.prepareOutput()
        stream = model.bindedMetadata.stream

        for batch, (inputs, labels) in enumerate(self.trainloader, start=self.batchNumbTrain):
            self.batchNumbTrain = batch
            if(SAVE_AND_EXIT_FLAG):
                return
            inputs, labels = inputs.to(model.bindedMetadata.device), labels.to(model.bindedMetadata.device)
            model.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = model.loss_fn(outputs, labels)
            loss.backward()
            model.optimizer.step()

            # run smoothing
            #smoothing.forwardLossFun(loss.item())
            #average = smoothing.fullAverageWeights(model.named_parameters())
            diff = smoothing.lastWeightDifference(model)

            # print statistics
            if metadata.debugInfo and metadata.howOftenPrintTrain is not None and batch % metadata.howOftenPrintTrain == 0:
                average = smoothing.getAverage()
                loss, current = loss.item(), batch * len(inputs)
                averKey = list(average.keys())[-1]
                
                stream.print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                stream.printTo(['debug', 'bash'], f"Average: {average[averKey]}")
                if(diff is None):
                    stream.print(f"No weight difference")
                else:
                    diffKey = list(diff.keys())[-1]
                    stream.printTo(['debug', 'bash'], f"Weight difference: {diff[diffKey]}")
                    stream.print(f"Weight difference average: {diff[diffKey].sum() / diff[diffKey].numel()} :: divided by: {diff[diffKey].numel()}")
                stream.print('################################################')

        model.average = average

    def testLoop(self, model):
        size = len(self.testloader.dataset)
        test_loss, correct = 0, 0
        model.eval()
        model.bindedMetadata.prepareOutput()
        stream = model.bindedMetadata.stream

        with torch.no_grad():
            for batch, (X, y) in enumerate(self.testloader, self.batchNumbTest):
                self.batchNumbTest = batch
                if(SAVE_AND_EXIT_FLAG):
                    return

                X = X.to(model.bindedMetadata.device)
                y = y.to(model.bindedMetadata.device)
                pred = model(X)
                test_loss += model.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        stream.print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def epochLoop(self, model):
        smoothing = Smoothing()
        smoothing.setDictionary(model.named_parameters())
        model.bindedMetadata.prepareOutput()
        stream = model.bindedMetadata.stream

        for ep, (loopEpoch) in enumerate(range(model.bindedMetadata.epoch), self.epochNumb):  # loop over the dataset multiple times
            self.epochNumb = ep
            stream.print(f"\nEpoch {loopEpoch+1}\n-------------------------------")
            sys.stdout.flush()
            if(model.bindedMetadata.trainFlag):
                self.trainLoop(model, smoothing)

            if(SAVE_AND_EXIT_FLAG):
                return
                
            with torch.no_grad():
                if(model.bindedMetadata.testFlag):
                    self.testLoop(model)

                # model.linear1.weight = torch.nn.parameter.Parameter(model.average)
                # model.linear1.weight = model.average
                model.load_state_dict(model.average)
                self.testLoop(model)

            self.batchNumbTrain = 0
            self.batchNumbTest = 0
        sys.stdout.flush()


    def update(self, metadata = None):
        if(metadata is not None):
            self.bindedMetadata = metadata
        # TODO może coś więcej dodać jak aktualizacja ścieżek dla danych (ponowne wczytanie)

    def trySave(self, metadata):
        if metadata.fileNameSave is not None and os.path.exists(metadata.PATH):
            path = metadata.PATH + metadata.fileNameSave + metadata.DATA_SUFFIX
            torch.save(self, path)
            print('Data saved successfully')
            return True
        print('Data save failure')
        return False

    def tryLoad(metadata):
        path = metadata.PATH + metadata.fileNameLoad + metadata.DATA_SUFFIX
        if metadata.fileNameLoad is not None and os.path.exists(path):
            obj = torch.load(path)
            print('Data loaded successfully')
            return obj
        print('Data load failure')
        return None

class Smoothing:
    def __init__(self):
        self.batchNumber = None
        self.vectorOscilation = None
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

    def beforeParamUpdate(self, model):
        return

    def afteParamUpdate(self, model):
        return

    def shapeLike(self, model):
        return

    def getWeightsAverage(self):
        return

    def getString(self):
        return

    def getStringDebug(self):
        return

    def addToAverageWeights(self, model):
        for key, arg in model.named_parameters():
            self.sumWeights[key].add_(arg)
        
    def getAverage(self):
        average = {}
        for key, arg in self.sumWeights.items():
            average[key] = self.sumWeights[key] / self.countWeights
        return average

    def fullAverageWeights(self, model):
        self.countWeights += 1
        return self.addToAverageWeights(model)
        
    def lateStartAverageWeights(self, model):
        self.countWeights += 1
        if(self.countWeights > self.numbOfBatchAfterSwitchOn):
            return self.addToAverageWeights(model)
        return dict(model)

    def comparePrevWeights(self, model):
        substract = {}
        self.addToAverageWeights(model)
        for key, arg in model.named_parameters():
            substract[key] = arg.sub(self.previousWeights[key])
            self.previousWeights[key].data.copy_(arg.data)
        return substract

    def lastWeightDifference(self, model):
        self.countWeights += 1
        if(self.countWeights > self.numbOfBatchAfterSwitchOn):
            return self.comparePrevWeights(model)
        return None

    def setDictionary(self, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        for key, values in dictionary:
            self.sumWeights[key] = torch.zeros_like(values, requires_grad=False)
            self.previousWeights[key] = torch.zeros_like(values, requires_grad=False)

    def forwardLossFun(self, loss):
        self.lossSum += loss
        self.lossList.append(loss)
        self.lossCounter += 1
        if(self.lossCounter > self.flushLossSum):
            self.lossAverage.append(self.lossSum / self.lossCounter)
            self.lossSum = 0.0
            self.lossCounter = 0
            variance = statistics.variance(self.lossList, self.lossAverage[-1])
            print(self.lossAverage[-1])
            print(variance)

#    def isSimilarLossFun(self):


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=metadata.hiperparam.learning_rate)
        #self.optimizer = optim.SGD(self.parameters(), lr=metadata.hiperparam.learning_rate, momentum=metadata.hiperparam.momentum)

        self.saveObj = None
        self.bindedMetadata = None

    def forward(self, x):
        x = self.pool(F.hardswish(self.conv1(x)))
        x = self.pool(F.hardswish(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.hardswish(self.linear1(x))
        x = F.hardswish(self.linear2(x))
        x = self.linear3(x)
        return x

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['saveObj']
        del state['bindedMetadata']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def createDump(self, metadata):
        self.saveObj = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict()
            }
        return self.saveObj

    def loadFromDump(self, obj):
        self.saveObj = obj
        #self.to(obj['device'])
        self.load_state_dict(obj['model_state_dict'])
        self.optimizer.load_state_dict(obj['optimizer_state_dict'])

    def tryLoad(self, metadata):
        '''
        Tries to load the model from the path in the metadata.
        '''
        path = metadata.PATH + metadata.fileNameLoad + metadata.MODEL_SUFFIX
        if metadata.fileNameLoad is not None and os.path.exists(path):
            dump = torch.load(path, map_location=metadata.device)
            self.loadFromDump(dump)
            self.bindedMetadata = metadata
            print('Model loaded successfully')
            return True
        print('Model load failure')
        return False

    def trySave(self, metadata):
        if metadata.fileNameSave is not None:
            path = metadata.PATH + metadata.fileNameSave + metadata.MODEL_SUFFIX
            torch.save(self.createDump(metadata), path)
            print('Model saved successfully')
            return True
        print('Model save failure')
        return False

    def update(self, metadata):
        '''
        Updates the model against the metadata and binds the metadata to the model
        '''
        self.to(metadata.device)
        self.loss_fn.to(metadata.device)
        self.bindedMetadata = metadata

        # must create new optimizer because we changed the model device. It must be set after setting model.
        # Some optimizers like Adam will have problem with it, others like SGD wont.
        self.optimizer = optim.AdamW(self.parameters(), lr=metadata.hiperparam.learning_rate)


metadata = MetaData()

if(__name__ == '__main__'):
    if metadata.commandLineArg(sys.argv[1:]) == False:
        metadata.trySelectCUDA()

metadata.printStartNewModel()

model = Model()
model.tryLoad(metadata)
    
model.update(metadata)

data = Data.tryLoad(metadata)

upd = False
if(data is None):
    data = Data()
    upd = True

data.update(metadata)

if(upd):
    data.setAll()


#print(metadata)
#print(data)

data.epochLoop(model)

# TODO sprawdzić, czy metadata == model == data
metadata.trySave()
model.trySave(metadata)
data.trySave(metadata)

playsound('/home/mateusz/Muzyka/simple_bell.mp3')