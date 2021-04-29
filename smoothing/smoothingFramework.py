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
from datetime import datetime
import time
import copy as cp
import random
from torch.utils.data.dataloader import Sampler

import matplotlib.pyplot as plt

SAVE_AND_EXIT_FLAG = False
MODEL_RESUME = False

def saveWorkAndExit(signumb, frame):
    global SAVE_AND_EXIT_FLAG
    SAVE_AND_EXIT_FLAG = True
    print('Ending and saving model')
    return

def terminate(signumb, frame):
    exit(2)

signal.signal(signal.SIGTSTP, saveWorkAndExit)

signal.signal(signal.SIGINT, terminate)

#reproducibility https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
# set in environment CUBLAS_WORKSPACE_CONFIG=':4096:2' or CUBLAS_WORKSPACE_CONFIG=':16:8'
def setDeterministic():
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


class StaticData:
    PATH = expanduser("~") + '/.data/models/'
    TMP_PATH = expanduser("~") + '/.data/models/tmp/'
    MODEL_SUFFIX = '.model'
    METADATA_SUFFIX = '.metadata'
    DATA_SUFFIX = '.data'
    TIMER_SUFFIX = '.timer'
    SMOOTHING_SUFFIX = '.smoothing'
    OUTPUT_SUFFIX = '.output'
    DATA_METADATA_SUFFIX = '.dmd'
    MODEL_METADATA_SUFFIX = '.mmd'
    NAME_CLASS_METADATA = 'Metadata'

class SaveClass:
    '''
    Should implement __setstate__ and __getstate__
    and children should implement its own static class methods tryLoad and trySave that invoke those.
    '''

    def tryLoad(fileName: str, suffix: str, classNameStr: str, temporaryLocation = False):
        path = None
        if(temporaryLocation):
            path = StaticData.TMP_PATH + fileName + suffix
        else:
            path = StaticData.PATH + fileName + suffix
        if fileName is not None and os.path.exists(path):
            toLoad = torch.load(path)
            loadedClassNameStr = toLoad['classNameStr']
            obj = toLoad['obj']
            if(loadedClassNameStr == classNameStr):
                print(classNameStr + ' loaded successfully')
                return obj
        print(classNameStr + ' load failure')
        return None

    def trySave(self, metadata, suffix: str, classNameStr: str, temporaryLocation = False) -> bool:
        if(metadata.fileNameSave is None):
            print(classNameStr + ' save not enabled')
            return False
        if metadata.fileNameSave is not None and os.path.exists(StaticData.PATH) and os.path.exists(StaticData.TMP_PATH):
            path = None
            if(temporaryLocation):
                path = StaticData.TMP_PATH + metadata.fileNameSave + suffix
            else:
                path = StaticData.PATH + metadata.fileNameSave + suffix
            toSave = {
                'classNameStr': classNameStr,
                'obj': self
            }
            torch.save(toSave, path)
            print(classNameStr + ' saved successfully')
            return True
        print(classNameStr + ' save failure')
        return False

class BaseSampler:
    '''
    Returns a sequence of the next indices
    ''' 
    def __init__(self, dataSize, batchSize, startIndex = 0, seed = 984):
        random.seed(seed)
        self.sequence = list(range(dataSize))[startIndex * batchSize:]
        random.shuffle(self.sequence)

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

# nieużywane
class TensorsContainer(SaveClass):
    def __init__(self):
        self.tensors = {}

    def add(self, name: str, tensor):
        self.tensors[name] = tensor

    def remove(self, name: str):
        self.tensors[name] = None

    def __getstate__(self):
        dictRet = {}
        for key, val in self.tensors.items():
            dictRet[key] = val.state_dict()
        return dictRet

    def __setstate__(self, state):
        for key, val in state.items():
            self.tensors[key] = tensor.load_state_dict(val)

# nieużywane
class Hyperparameters(SaveClass):
    def __init__(self):
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.oscilationMax = 0.001

    def __str__(self):
        tmp_str = '\n/Hyperparameters class\n-----------------------------------------------------------------------\n'
        tmp_str += ('Learning rate:\t{}\n'.format(self.learning_rate))
        tmp_str += ('Momentum:\t{}\n'.format(self.momentum))
        tmp_str += ('-----------------------------------------------------------------------\nEnd Hyperparameters class\n')
        return tmp_str

class MetaData(SaveClass):
    def __init__(self):
        self.fileNameSave = None
        self.fileNameLoad = None

        self.testFlag = False
        self.trainFlag = False

        self.debugInfo = False
        self.modelOutput = None
        self.debugOutput = None
        self.stream = None
        self.bashFlag = False
        self.name = None
        self.formatedOutput = None

    def __str__(self):
        tmp_str = ('\n/MetaData class\n-----------------------------------------------------------------------\n')
        tmp_str += ('Save path:\t{}\n'.format(StaticData.PATH + self.fileNameSave if self.fileNameSave is not None else 'Not set'))
        tmp_str += ('Load path:\t{}\n'.format(StaticData.PATH + self.fileNameLoad if self.fileNameLoad is not None else 'Not set'))
        tmp_str += ('Test flag:\t{}\n'.format(self.testFlag))
        tmp_str += ('Train flag:\t{}\n'.format(self.trainFlag))
        tmp_str += ('-----------------------------------------------------------------------\nEnd MetaData class\n')
        return tmp_str

    def onOff(arg):
        if arg == 'on' or arg == 'True' or arg == 'true':
            return True
        elif arg == 'off' or arg == 'False' or arg == 'false':
            return False
        else:
            return None

    def printStartNewModel(self):
        if(self.stream is None):
            raise Exception("Stream not initialized")
        if(self.name is not None):
            self.stream.print(f"\n@@@@\nStarting new model: " + self.name + "\nTime: " + str(datetime.now()) + "\n@@@@\n")
        else:
            self.stream.print(f"\n@@@@\nStarting new model without name\nTime: " + str(datetime.now()) + "\n@@@@\n")

    def printContinueLoadedModel(self):
        if(self.stream is None):
            raise Exception("Stream not initialized")
        if(self.name is not None):
            self.stream.print(f"\n@@@@\nContinuing loaded model: " + self.name + "\nTime: " + str(datetime.now()) + "\n@@@@\n")
        else:
            self.stream.print(f"\n@@@@\nContinuing loaded model without name\nTime: " + str(datetime.now()) + "\n@@@@\n")

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
        if(self.formatedOutput is not None):
            self.stream.open('formatedLog', self.formatedOutput)

    def commandLineArg(self, dataMetadata, modelMetadata, argv):
        '''
        Returns False if metadata was not loaded.
        Otherwise True.
        '''
        help = 'Help:\n'
        help += os.path.basename(__file__) + ' -h <help> [-s,--save] <file name to save> [-l,--load] <file name to load>'

        shortOptions = 'hs:l:d'
        longOptions = [
            'save=', 'load=', 'test=', 'train=', 'pinTest=', 'pinTrain=', 'debug', 
            'debugOutput=',
            'modelOutput=',
            'bashOutput=',
            'name=',
            'formatedOutput='
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
                self.fileNameSave = arg
            elif opt in ('-l', '--load'):
                self.fileNameLoad = arg
            elif opt in ('--test'):
                boolean = MetaData.onOff(arg)
                self.testFlag = boolean if boolean is not None else MetaData.exitError(help)
            elif opt in ('--train'):
                boolean = MetaData.onOff(arg)
                self.trainFlag = boolean if boolean is not None else MetaData.exitError(help)
            elif opt in ('--pinTest'):
                boolean = MetaData.onOff(arg)
                dataMetadata.tryPinMemoryTest(self, modelMetadata) if boolean is not None else MetaData.exitError(help)
            elif opt in ('--pinTrain'):
                boolean = MetaData.onOff(arg)
                dataMetadata.tryPinMemoryTrain(self, modelMetadata) if boolean is not None else MetaData.exitError(help)
            elif opt in ('-d', '--debug'):
                self.debugInfo = True
            elif opt in ('--debugOutput'):
                self.debugOutput = arg # debug output file path
            elif opt in ('--modelOutput'):
                self.modelOutput = arg # model output file path
            elif opt in ('--bashOutput'):
                boolean = MetaData.onOff(arg)
                self.bashFlag = boolean if boolean is not None else MetaData.exitError(help)
            elif opt in ('--formatedOutput'):
                self.formatedOutput = arg # formated output file path
            elif opt in ('--name'):
                self.name = arg

        if(self.modelOutput is None):
            self.modelOutput = 'default.log'

        if(self.debugOutput is None):
            self.debugOutput = 'default.log'  
        
        if(self.fileNameLoad is not None):
            d = MetaData.tryLoad(self.fileNameLoad)
            if(d is None):
                return None
            print("Command line options mostly ignored.")
            return d
        return self

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def trySave(self, temporaryLocation = False):
        return super().trySave(self, StaticData.METADATA_SUFFIX, MetaData.__name__, temporaryLocation)

    def tryLoad(fileName: str, temporaryLocation = False):
        return SaveClass.tryLoad(fileName, StaticData.METADATA_SUFFIX, MetaData.__name__, temporaryLocation)



class Timer(SaveClass):
    def __init__(self):
        self.timeStart = None
        self.timeEnd = None
        self.modelTimeSum = 0.0
        self.modelTimeCount = 0

    def start(self):
        self.timeStart = time.perf_counter()

    def end(self):
        self.timeEnd = time.perf_counter()

    def getDiff(self):
        if(self.timeStart is not None and self.timeEnd is not None):
            return self.timeEnd - self.timeStart
        return None

    def addToStatistics(self):
        tmp = self.getDiff()
        if(tmp is not None):
            self.modelTimeSum += self.getDiff()
            self.modelTimeCount += 1

    def clearTime(self):
        self.timeStart = None
        self.timeEnd = None

    def clearStatistics(self):
        self.modelTimeSum = 0.0
        self.modelTimeCount = 0
        
    def getAverage(self):
        if(self.modelTimeCount != 0):
            return self.modelTimeSum / self.modelTimeCount
        return None

    def getUnits(self):
        return "s"

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.TIMER_SUFFIX, Timer.__name__, temporaryLocation)

    def tryLoad(metadata, temporaryLocation = False):
        return SaveClass.tryLoad(metadata.fileNameLoad, StaticData.TIMER_SUFFIX, Timer.__name__, temporaryLocation)

class Output(SaveClass):
    def __init__(self):
        self.debugF = None
        self.modelF = None
        self.debugPath = None
        self.modelPath = None
        self.bash = False
        self.formatedLogF = None
        self.formatedLogPath = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['debugF']
        del state['modelF']
        del state['formatedLogF']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.formatedLogF = None
        self.debugF = None
        self.modelF = None

        if(self.debugPath is not None):
            self.debugF = open(self.debugPath + ".log", 'a')
        if(self.modelPath is not None):
            if(self.modelPath == self.debugPath):
                self.modelF = self.debugF
            else:
                self.modelF = open(self.modelPath + ".log", 'a')
        if(self.formatedLogPath is not None):
            if(self.formatedLogPath == self.debugPath):
                self.formatedLogF = self.debugF
            elif(self.formatedLogPath == self.modelPath):
                self.formatedLogF = self.modelF
            else:
                self.formatedLogF = open(self.formatedLogPath + ".log", 'a')

    def open(self, outputType, path = None):
            if(outputType != 'debug' and outputType != 'model' and outputType != 'bash' and outputType != 'formatedLog'):
                raise Exception("unknown command")

            if(outputType == 'bash'):
                self.bash = True
                return

            # if you want to open file with other path
            if(outputType == 'debug' and path != self.debugPath and self.debugPath is not None and path is not None):
                self.debugF.close()
                self.debugPath = None
                self.debugF = None
            elif(outputType == 'model' and path != self.modelPath and self.modelPath is not None and path is not None):
                self.modelF.close()
                self.modelPath = None
                self.modelF = None
            elif(outputType == 'formatedLog' and path != self.formatedLogPath and self.formatedLogPath is not None and path is not None):
                self.formatedLogF.close()
                self.formatedLogF = None
                self.formatedLogPath = None

            # if file is already open in different outputType
            if(outputType == 'debug' and path is not None ):
                if(path == self.modelPath):
                    self.debugPath = path
                    self.debugF = self.modelF
                elif(path == self.formatedLogPath):
                    self.debugPath = path
                    self.debugF = self.formatedLogF
            elif(outputType == 'model' and path is not None):
                if(path == self.debugPath):
                    self.modelPath = path
                    self.modelF = self.debugF
                elif(path == self.formatedLogPath):
                    self.modelPath = path
                    self.modelF = self.formatedLogF
            elif(outputType == 'formatedLog' and path is not None and path == self.debugPath):
                if(path == self.debugPath):
                    self.formatedLogPath = path
                    self.formatedLogF = self.debugF
                elif(path == self.modelPath):
                    self.formatedLogPath = path
                    self.formatedLogF = self.modelF

            # if file was not opened
            if(outputType == 'debug' and path is not None and self.debugPath is None):
                self.debugF = open(path + ".log", 'a')
                self.debugPath = path
            elif(outputType == 'model' and path is not None and self.modelPath is None):
                self.modelF = open(path + ".log", 'a')
                self.modelPath = path
            elif(outputType == 'formatedLog' and path is not None and self.formatedLogPath is None):
                self.formatedLogF = open(path + ".log", 'a')
                self.formatedLogPath = path

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

    def writeFormated(self, arg):
        if(self.formatedLogPath is not None):
            self.formatedLogF.write(arg)

    def printFormated(self, arg):
        if(self.formatedLogPath is not None):
            self.formatedLogF.write(arg + '\n')

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
        if(self.formatedLogF is not None): 
            self.formatedLogF.close()# if closed this do nothing

    def flushAll(self):
        if(self.debugF is not None):
            self.debugF.flush()
        if(self.modelF is not None):
            self.modelF.flush()
        if(self.formatedLogF is not None):
            self.formatedLogF.flush()
        sys.stdout.flush()

    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.OUTPUT_SUFFIX, Output.__name__, temporaryLocation)

    def tryLoad(metadata, temporaryLocation = False):
        return SaveClass.tryLoad(metadata.fileNameLoad, StaticData.OUTPUT_SUFFIX, Output.__name__, temporaryLocation)



class Data_Metadata(SaveClass):
    DATA_PATH = '~/.data'

    def __init__(self):
        self.train = True
        self.download = True
        self.pin_memoryTrain = False
        self.pin_memoryTest = False

        self.epoch = 1
        self.batchTrainSize = 4
        self.batchTestSize = 4

        # batch size * howOftenPrintTrain
        self.howOftenPrintTrain = 2000

    def tryPinMemoryTrain(self, metadata, modelMetadata):
        if(torch.cuda.is_available()):
            self.pin_memoryTrain = True
        return bool(self.pin_memoryTrain)
        '''if modelMetadata.device == 'cpu': modelMetadata.trySelectCUDA(metadata)
        if(Model_Metadata.checkCUDA(modelMetadata.device)):
            self.pin_memoryTrain = True
        if(metadata.debugInfo):
            print('Train data pinned to GPU: {}'.format(self.pin_memoryTrain))
        return bool(self.pin_memoryTrain)'''

    def tryPinMemoryTest(self, metadata, modelMetadata):
        if(torch.cuda.is_available()):
            self.pin_memoryTest = True
        return bool(self.pin_memoryTest)
        '''
        if modelMetadata.device == 'cpu': modelMetadata.trySelectCUDA(metadata)
        if(Model_Metadata.checkCUDA(modelMetadata.device)):
            self.pin_memoryTest = True
        if(metadata.debugInfo):
            print('Test data pinned to GPU: {}'.format(self.pin_memoryTest))
        return bool(self.pin_memoryTest)'''

    # unused
    def tryPinMemoryAll(self, metadata, modelMetadata):
        return self.tryPinMemoryTrain(metadata, modelMetadata), self.tryPinMemoryTest(metadata, modelMetadata)

    def __str__(self):
        tmp_str = ('Should train data:\t{}\n'.format(self.train))
        tmp_str += ('Download data:\t{}\n'.format(self.download))
        tmp_str += ('Pin memory train:\t{}\n'.format(self.pin_memoryTrain))
        tmp_str += ('Pin memory test:\t{}\n'.format(self.pin_memoryTest))
        tmp_str += ('Batch train size:\t{}\n'.format(self.batchTrainSize))
        tmp_str += ('Batch test size:\t{}\n'.format(self.batchTestSize))
        tmp_str += ('Number of epochs:\t{}\n'.format(self.epoch))

    def _getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.DATA_METADATA_SUFFIX, Data_Metadata.__name__, temporaryLocation)

    def tryLoad(metadata, temporaryLocation = False):
        return SaveClass.tryLoad(metadata.fileNameLoad, StaticData.DATA_METADATA_SUFFIX, Data_Metadata.__name__, temporaryLocation)

class Model_Metadata(SaveClass):
    def __init__(self):
        self.learning_rate = 1e-3 # TODO usunąć, bo to klasa podstawowa
        self.device = 'cuda:0'
        
    def checkCUDA(string):
        return string.startswith('cuda')

    def trySelectCUDA(self, metadata):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if(metadata.debugInfo):
            print('Using {} torch CUDA device version\nCUDA avaliable: {}\nCUDA selected: {}'.format(torch.version.cuda, torch.cuda.is_available(), self.device == 'cuda'))
        return self.device

    def selectCPU(self, metadata):
        self.device = 'cpu'
        if(metadata.debugInfo):
            print('Using {} torch CUDA device version\nCUDA avaliable: {}\nCUDA selected: False'.format(torch.version.cuda, torch.cuda.is_available()))
        return self.device

    def _getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.MODEL_METADATA_SUFFIX, Model_Metadata.__name__, temporaryLocation)

    def tryLoad(metadata, temporaryLocation = False):
        return SaveClass.tryLoad(metadata.fileNameLoad, StaticData.MODEL_METADATA_SUFFIX, Model_Metadata.__name__, temporaryLocation)

# nieużywane
class Smoothing_Metadata(SaveClass):
    '''
    Here store weighs
    '''
    def __init__(self):
        self.weights = None

    def addWeights(self, weights):
        self.weights = weights

class Weight(SaveClass):
    def __init__(self, weightDict):
        self.weight = weightDict

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.WEIGHT_SUFFIX, Weight.__name__, temporaryLocation)

    def tryLoad(metadata, temporaryLocation = False):
        return SaveClass.tryLoad(metadata.fileNameSave, StaticData.WEIGHT_SUFFIX, Weight.__name__, temporaryLocation)

class TrainDataContainer():
    def __init__(self):
        self.size = None
        self.timer = None
        self.loopTimer = None

        # one loop train data
        self.loss = None
        self.diff = None
        self.inputs = None
        self.labels = None

        self.batchNumber = None

class TestDataContainer():
    def __init__(self):
        self.size = None
        self.timer = None
        self.loopTimer = None

        # one loop test data
        self.pred = None
        self.inputs = None
        self.labels = None

        self.batchNumber = None

        # one loop test data
        self.test_loss = 0
        self.test_correct = 0

class EpochDataContainer():
    def __init__(self):
        self.epochNumber = None
        self.returnObj = None

class Data(SaveClass):
    def __init__(self):
        self.trainset = None
        self.trainloader = None
        self.testset = None
        self.testloader = None
        self.transform = None

        self.batchNumbTrain = 0
        self.batchNumbTest = 0
        self.epochNumb = 0

        self.trainSampler = None
        self.testSampler = None

        """
        Aby wyciągnąć helper poza pętlę, należy go przypisać do dodatkowej zmiennej przy przeciążeniu dowolnej, wewnętrznej metody dla danej pętli. 
        Inaczej po zakończonych obliczeniach zmienna self.trainHelper zostanie ustawiona na None.
        Podane zmienne używane są po to, aby umożliwić zapisanie stanu programu.
        Aby jednak wymusić ponowne stworzenie danych obiektów, należy przypisać im wartość None.
        """
        self.trainHelper = None
        self.testHelper = None
        self.epochHelper = None

    def __str__(self):
        tmp_str = ('\n/Data class\n-----------------------------------------------------------------------\n')
        tmp_str += ('Is trainset set:\t{}\n'.format(self.trainset is not None))
        tmp_str += ('Is trainloader set:\t{}\n'.format(self.trainloader is not None))
        tmp_str += ('Is testset set:\t\t{}\n'.format(self.testset is not None))
        tmp_str += ('Is testloader set:\t{}\n'.format(self.testloader is not None))
        tmp_str += ('Is transform set:\t{}\n'.format(self.transform is not None))
        tmp_str += ('-----------------------------------------------------------------------\nEnd Data class\n')
        return tmp_str

    def __customizeState__(self, state):
        del state['trainset']
        del state['trainloader']
        del state['testset']
        del state['testloader']
        del state['transform']

    def __getstate__(self):
        """
        Nie nadpisujemy tej klasy. Do nadpisania służy \_\_customizeState__(self, state).
        """
        state = self.__dict__.copy()
        self.__customizeState__(state)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.trainset = None
        self.trainloader = None
        self.testset = None
        self.testloader = None
        self.transform = None
        self.setTransform()

    '''def setTrainData(self, trainset, trainSampler, dataMetadata):
        self.trainset = trainset
        self.trainSampler = trainSampler
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain)
        return self.trainset, self.trainloader

    def setTestData(self, testset, testSampler, dataMetadata):
        self.testset = testset
        self.testSampler = testSampler
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest)
        return self.testset, self.testloader'''

    def setTransform(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        return self.transform

    #def setAll(self, trainSampler, testSampler, trainset, testset, dataMetadata):
        '''
        Set transform of the input data and set train and test data.
        Returns transform function, trainset, trainloader, testset, testloader
        '''
        #return self.setTransform(), self.setTrainData(trainSampler, dataMetadata), self.setTestData(testSampler, dataMetadata)

    def prepare(self, dataMetadata):
        self.setTransform()

        self.trainset = torchvision.datasets.CIFAR10(root='~/.data', train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(root='~/.data', train=False, download=True, transform=self.transform)
        self.trainSampler = BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = BaseSampler(len(self.testset), dataMetadata.batchTrainSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest)

    def __train__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing):
        # forward + backward + optimize
        outputs = model(helper.inputs)
        helper.loss = model.loss_fn(outputs, helper.labels)
        helper.loss.backward()
        model.optimizer.step()

        # run smoothing
        helper.diff = smoothing.lastWeightDifference(model)
        smoothing.call(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata)

    def setTrainLoop(self, model, modelMetadata, metadata):
        helper = TrainDataContainer()
        helper.size = len(self.trainloader.dataset)
        model.train()
        metadata.prepareOutput()
        helper.timer = Timer()
        helper.loopTimer = Timer()
        helper.loss = None 
        helper.diff = None

        return helper

    def __beforeTrainLoop__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing):
        pass

    def __beforeTrain__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing):
        helper.inputs, helper.labels = helper.inputs.to(modelMetadata.device), helper.labels.to(modelMetadata.device)
        model.optimizer.zero_grad()

    def __afterTrain__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing):
        if(bool(metadata.debugInfo) and dataMetadata.howOftenPrintTrain is not None and helper.batchNumber % dataMetadata.howOftenPrintTrain == 0):
            averageWeights = smoothing.getStateDict()
            calcLoss, current = helper.loss.item(), helper.batchNumber * len(helper.inputs)
            averKey = list(averageWeights.keys())[-1]
            
            metadata.stream.print(f"loss: {calcLoss:>7f}  [{current:>5d}/{helper.size:>5d}]")
            metadata.stream.printTo(['debug', 'bash'], f"Average: {averageWeights[averKey]}")
            if(helper.diff is None):
                metadata.stream.print(f"No weight difference")
            else:
                diffKey = list(helper.diff.keys())[-1]
                metadata.stream.printTo(['debug', 'bash'], f"Weight difference: {helper.diff[diffKey]}")
                metadata.stream.print(f"Weight difference of last layer average: {helper.diff[diffKey].sum() / helper.diff[diffKey].numel()} :: was divided by: {helper.diff[diffKey].numel()}")
                metadata.stream.print('################################################')

    def __afterTrainLoop__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing):
        diffKey = list(helper.diff.keys())[-1]
        metadata.stream.print("Train summary:")
        metadata.stream.print(f" Average train time ({helper.timer.getUnits()}): {helper.timer.getAverage()}")
        metadata.stream.print(f" Loop train time ({helper.timer.getUnits()}): {helper.loopTimer.getDiff()}")
        metadata.stream.printFormated("trainLoop;\nAverage train time;Loop train time;Weight difference of last layer average;divided by;")
        metadata.stream.printFormated(f"{helper.timer.getAverage()};{helper.loopTimer.getDiff()};{helper.diff[diffKey].sum() / helper.diff[diffKey].numel()};{helper.diff[diffKey].numel()}")

    def trainLoop(self, model, helperEpoch, dataMetadata, modelMetadata, metadata, smoothing):
        if(self.trainHelper is None): # jeżeli nie było wznowione; nowe wywołanie
            self.trainHelper = self.setTrainLoop(model, modelMetadata, metadata)
        self.__beforeTrainLoop__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

        self.trainHelper.loopTimer.start()
        for batch, (inputs, labels) in enumerate(self.trainloader, start=self.batchNumbTrain):
            self.trainHelper.inputs = inputs
            self.trainHelper.labels = labels
            self.trainHelper.batchNumber = batch
            self.batchNumbTrain = batch
            if(SAVE_AND_EXIT_FLAG):
                return
            
            self.__beforeTrain__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

            self.trainHelper.timer.clearTime()
            self.trainHelper.timer.start()
            self.__train__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
            self.trainHelper.timer.end()
            self.trainHelper.timer.addToStatistics()

            self.__afterTrain__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

        self.trainHelper.loopTimer.end()
        self.__afterTrainLoop__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
        self.trainHelper = None

    def __test__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing):
        helper.pred = model(helper.inputs)

    def setTestLoop(self, model, modelMetadata, metadata):
        helper = TestDataContainer()
        helper.size = len(self.testloader.dataset)
        helper.test_loss, helper.test_correct = 0, 0
        model.eval()
        metadata.prepareOutput()
        helper.timer = Timer()
        helper.loopTimer = Timer()
        return helper

    def __beforeTestLoop__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing):
        metadata.stream.printFormated("testLoop;\nAverage test time;Loop test time;Accuracy;Avg loss")

    def __beforeTest__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing):
        helper.inputs = helper.inputs.to(modelMetadata.device)
        helper.labels = helper.labels.to(modelMetadata.device)

    def __afterTest__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing):
        helper.test_loss += model.loss_fn(helper.pred, helper.labels).item()
        helper.test_correct += (helper.pred.argmax(1) == helper.labels).type(torch.float).sum().item()

    def __afterTestLoop__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing):
        helper.test_loss /= helper.size
        helper.test_correct /= helper.size
        metadata.stream.print(f"Test summary: \n Accuracy: {(100*helper.test_correct):>0.1f}%, Avg loss: {helper.test_loss:>8f}")
        metadata.stream.print(f" Average test time ({helper.timer.getUnits()}): {helper.timer.getAverage()}")
        metadata.stream.print(f" Loop test time ({helper.timer.getUnits()}): {helper.loopTimer.getDiff()}")
        metadata.stream.print("")
        metadata.stream.printFormated(f"{helper.timer.getAverage()};{helper.loopTimer.getDiff()};{(helper.test_correct):>0.0001f};{helper.test_loss:>8f}")

    def testLoop(self, helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing):
        if(self.testHelper is None): # jeżeli nie było wznowione; nowe wywołanie
            self.testHelper = self.setTestLoop(model, modelMetadata, metadata)
        self.__beforeTestLoop__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

        with torch.no_grad():
            self.testHelper.loopTimer.start()
            for batch, (inputs, labels) in enumerate(self.testloader, self.batchNumbTest):
                self.testHelper.inputs = inputs
                self.testHelper.labels = labels
                self.testHelper.batchNumber = batch
                self.batchNumbTest = batch
                if(SAVE_AND_EXIT_FLAG):
                    return

                self.__beforeTest__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

                self.testHelper.timer.clearTime()
                self.testHelper.timer.start()
                self.__test__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
                self.testHelper.timer.end()
                self.testHelper.timer.addToStatistics()

                self.__afterTest__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

            self.testHelper.loopTimer.end()

        self.__afterTestLoop__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
        self.testHelper = None

    def __beforeEpochLoop__(self, helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing):
        pass

    def __afterEpochLoop__(self, helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing):
        pass

    def __epoch__(self, helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing):
        """
        Reprezentuje pojedynczy epoch.
        Znajduje się tu cała logika epocha. Aby wykorzystać możliwość wyjścia i zapisu w danym momencie stanu modelu, należy zastosować konstrukcję:

        if(SAVE_AND_EXIT_FLAG):\n
            return 
        """
        if(metadata.trainFlag):
            self.trainLoop(model, helperEpoch, dataMetadata, modelMetadata, metadata, smoothing)
        
        if(SAVE_AND_EXIT_FLAG):
            return 

        with torch.no_grad():
            if(metadata.testFlag):
                metadata.stream.write("Plain weights, ")
                metadata.stream.writeFormated("Plain weights;")
                self.testLoop(helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing)
                smoothing.saveMainWeight(model)
                model.setWeights(smoothing)
                metadata.stream.write("Smoothing weights, ")
                metadata.stream.writeFormated("Smoothing weights;")
                self.testLoop(helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing)
                # TODO dodać możliwość zmieniania wag

            # model.linear1.weight = torch.nn.parameter.Parameter(model.average)
            # model.linear1.weight = model.average

    def epochLoop(self, model, dataMetadata, modelMetadata, metadata, smoothing):
        '''smoothing = Smoothing()
        smoothing.setDictionary(model.named_parameters())'''
        metadata.prepareOutput()
        self.epochHelper = EpochDataContainer()

        self.__beforeEpochLoop__(self.epochHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

        for ep, (loopEpoch) in enumerate(range(dataMetadata.epoch), self.epochNumb):  # loop over the dataset multiple times
            self.epochHelper.epochNumber = ep
            metadata.stream.print(f"\nEpoch {loopEpoch+1}\n-------------------------------")
            metadata.stream.flushAll()
            
            self.__epoch__(self.epochHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
            if(SAVE_AND_EXIT_FLAG):
                return

            self.resetEpochState()

        self.__afterEpochLoop__(self.epochHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
        self.resetFullEpochState()
        metadata.stream.flushAll()
        self.epochHelper = None

    def resetEpochState(self):
        self.batchNumbTrain = 0
        self.batchNumbTest = 0

    def resetFullEpochState(self):
        self.resetEpochState()
        self.epochNumb = 0

    def __update__(self, dataMetadata):
        self.trainset = torchvision.datasets.CIFAR10(root='~/.data', train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(root='~/.data', train=False, download=True, transform=self.transform)
        self.setTransform()
        if(self.trainset is not None or self.trainSampler is not None):
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain)

        if(self.testset is not None or self.testSampler is not None):
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest)

    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.DATA_SUFFIX, Data.__name__, temporaryLocation)

    def tryLoad(metadata, dataMetadata, temporaryLocation = False):
        return SaveClass.tryLoad(metadata.fileNameLoad, StaticData.DATA_SUFFIX, Data.__name__, temporaryLocation)

class Smoothing(SaveClass):
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
                substract[key] = arg.sub(self.previousWeights[key])
                self.previousWeights[key].data.copy_(arg.data)


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

    def saveMainWeight(self, model):
        self.mainWeights = model.getWeights()

    def addToAverageWeights(self, model):
        with torch.no_grad():
            for key, arg in model.named_parameters():
                self.sumWeights[key].add_(arg)
        
    # ważne
    def getStateDict(self):
        average = {}
        for key, arg in self.sumWeights.items():
            average[key] = self.sumWeights[key] / self.countWeights
        return average
'''
    def fullAverageWeights(self, model):
        self.counter += 1
        self.countWeights += 1
        return self.addToAverageWeights(model)
   '''
   '''     
    def lateStartAverageWeights(self, model):
        self.counter += 1
        if(self.countWeights > self.numbOfBatchAfterSwitchOn):
            self.countWeights += 1
            return self.addToAverageWeights(model)
        return dict(model)
'''
    def comparePrevWeights(self, model):
        substract = {}
        self.addToAverageWeights(model)
        for key, arg in model.named_parameters():
            substract[key] = arg.sub(self.previousWeights[key])
            self.previousWeights[key].data.copy_(arg.data)
        return substract

    def lastWeightDifference(self, model):
        self.counter += 1
        if(self.counter > self.numbOfBatchAfterSwitchOn):
            self.countWeights += 1
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

    def __getstate__(self):
        state = self.__dict__.copy()
        '''del state['sumWeights']
        del state['previousWeights']
        del state['mainWeights']
        state['tensors']
        state['sumWeights'] = self.sumWeights.state_dict()
        state['previousWeights'] = self.previousWeights.state_dict()
        state['mainWeights'] = self.mainWeights.state_dict()'''
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.SMOOTHING_SUFFIX, Smoothing.__name__, temporaryLocation)

    def tryLoad(metadata, temporaryLocation = False):
        return SaveClass.tryLoad(metadata.fileNameSave, StaticData.SMOOTHING_SUFFIX, Smoothing.__name__, temporaryLocation)

    def __update__(self):
        if(bool(self.sumWeights)):
            for key, val in self.sumWeights.items():
                val.to('cuda:0')
        if(bool(self.previousWeights)):
            for key, val in self.previousWeights.items():
                val.to('cuda:0')
        if(self.mainWeights is not None):
            self.mainWeights.to('cuda:0')

class Model(nn.Module, SaveClass):
    def __init__(self, modelMetadata):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=modelMetadata.learning_rate)
        #self.optimizer = optim.SGD(self.parameters(), lr=metadata.hiperparam.learning_rate, momentum=metadata.hiperparam.momentum)

        #self.weightsStateDict: dict = None
        #self.weightsStateDictType = None

        self.to(modelMetadata.device)

    def forward(self, x):
        x = self.pool(F.hardswish(self.conv1(x)))
        x = self.pool(F.hardswish(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.hardswish(self.linear1(x))
        x = F.hardswish(self.linear2(x))
        x = self.linear3(x)
        return x

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.eval()

    '''def cloneStateDict(weights: dict):
        newDict = dict()
        for key, val in weights.items():
            newDict[key] = torch.clone(val)
        return newDict'''

    '''def pinAverageWeights(self, weights: dict, copy = False):
        if(copy):
            self.weightsStateDict = Model.cloneStateDict(weights)
        else:
            self.weightsStateDict = weights
        self.weightsStateDictType = 'plain'''

    def setWeights(self, smoothing):
        # inherent
        self.load_state_dict(smoothing.getStateDict())

    def getWeights(self):
        return self.state_dict()

    '''def swapWeights(self):
        if(self.weightsStateDictType == 'smoothing' and self.weightsStateDict is not None):
            tmp = self.state_dict()
            self.load_state_dict(self.weightsStateDict)
            self.weightsStateDict = tmp
            self.weightsStateDictType = 'plain'
            return 'plain'
        elif(self.weightsStateDictType == 'plain' and self.weightsStateDict is not None):
            tmp = self.state_dict()
            self.load_state_dict(self.weightsStateDict)
            self.weightsStateDict = tmp
            self.weightsStateDictType = 'smoothing'
            return 'smoothing'
        else:
            raise Exception("unknown command or unset variables")'''

    def trySave(self, metadata, temporaryLocation = False):
        return super().trySave(metadata, StaticData.MODEL_SUFFIX, Model.__name__, temporaryLocation)

    def tryLoad(metadata, modelMetadata, temporaryLocation = False):
        obj = SaveClass.tryLoad(metadata.fileNameLoad, StaticData.MODEL_SUFFIX, Model.__name__, temporaryLocation)
        obj.__update__(modelMetadata)
        return obj
    
    def __update__(self, modelMetadata):
        '''
        Updates the model against the metadata and binds the metadata to the model
        '''
        self.to(modelMetadata.device)
        #self.loss_fn.to(metadata.device)

        # must create new optimizer because we changed the model device. It must be set after setting model.
        # Some optimizers like Adam will have problem with it, others like SGD wont.
        self.optimizer = optim.AdamW(self.parameters(), lr=modelMetadata.learning_rate)


def tryLoad(tupleClasses: list, metadata, temporaryLocation = False):
    dictObjs = {}
    dictObjs[type(metadata).__name__] = metadata
    if(dictObjs['MetaData'] is None):
        return None

    for mdcl, objcl in tupleClasses:
        # load class metadata
        if(mdcl is not None):
            dictObjs[mdcl.__name__] = mdcl.tryLoad(metadata, temporaryLocation)
            if(dictObjs[mdcl.__name__] is None):
                return None
            # load class
            dictObjs[objcl.__name__] = objcl.tryLoad(metadata, dictObjs[mdcl.__name__], temporaryLocation)
            if(dictObjs[objcl.__name__] is None):
                return None
        else:
            dictObjs[objcl.__name__] = objcl.tryLoad(metadata, temporaryLocation)
            if(dictObjs[objcl.__name__] is None):
                return None
        #update
        if(mdcl is not None):
            dictObjs[objcl.__name__].__update__(dictObjs[mdcl.__name__])
    return dictObjs

def trySave(dictObjs: dict, temporaryLocation = False):
    dictObjs['MetaData'].trySave()
    md = dictObjs['MetaData']
    
    for key, obj in dictObjs.items():
        if(key != 'MetaData'):
            obj.trySave(md)

def model():
    if(__name__ == '__main__'):
        metadata = MetaData()
        metadata.prepareOutput()
        loadedSuccessful = False

        dictObjs = {'MetaData': metadata}
        dictObjs['Data_Metadata'] = Data_Metadata()
        dictObjs['Model_Metadata'] = Model_Metadata()

        metadataTmp = metadata.commandLineArg(dictObjs['Data_Metadata'], dictObjs['Model_Metadata'], sys.argv[1:])
        if(metadataTmp is not None): # if model should be loaded
            metadata = metadataTmp
            dictObjsTmp = tryLoad([(Data_Metadata, Data), (None, Smoothing), (Model_Metadata, Model)], metadata)
            if(dictObjsTmp is None):
                loadedSuccessful = False
            else:
                dictObjs = dictObjsTmp
                metadata = dictObjs['MetaData']
                dictObjs['Data'].__update__(dictObjs['Data_Metadata'])
                loadedSuccessful = True
                metadata.printContinueLoadedModel()

        if(loadedSuccessful == False):
            metadata.printStartNewModel()
            dictObjs['Data'] = Data()
            
            dictObjs['Data'].prepare(dictObjs['Data_Metadata'])
            
            dictObjs['Smoothing'] = Smoothing()
            dictObjs['Model'] = Model(dictObjs['Model_Metadata'])

            dictObjs['Smoothing'].setDictionary(dictObjs['Model'].named_parameters())

        dictObjs['Data'].epochLoop(dictObjs['Model'], dictObjs['Data_Metadata'], dictObjs['Model_Metadata'], dictObjs['MetaData'], dictObjs['Smoothing'])

        trySave(dictObjs)

setDeterministic()
model()


'''
dataDict = {
    0: 'a0',
    1: 'b1',
    2: 'c2',
    3: 'd3',
    4: 'e4',
    5: 'f5',
    6: 'g6',
    7: 'h7',
    8: 'i8',
    9: 'j9'
}

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
testset = torchvision.datasets.CIFAR10(root='~/.data', train=False, download=True, transform=transform)
sampler = BaseSampler(testset, 4, seed=6893647)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, sampler=sampler,
                                    shuffle=False, num_workers=2, pin_memory=True)




for i, (x, y) in enumerate(testloader):
    print(x, y)
    if(i == 1):
        break'''
'''
stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

train_data = torchvision.datasets.CIFAR100(download=True,root="~/.data",transform=train_transform)
train_dl = torch.utils.data.DataLoader(train_data,128,num_workers=4,pin_memory=True,shuffle=True)

for image,label in train_data:
    print("Image shape: ",image.shape)
    print("Image tensor: ", image)
    print("Label: ", label)
    break

def show_batch(dl):
    for batch in dl:
        images,labels = batch
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(torchvision.utils.make_grid(images[:20],nrow=5, normalize=True).permute(1,2,0))
        plt.show()
        break

show_batch(train_dl)'''