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
import numpy

SAVE_AND_EXIT_FLAG = False
DETERMINISTIC = False


def saveWorkAndExit(signumb, frame):
    global SAVE_AND_EXIT_FLAG
    SAVE_AND_EXIT_FLAG = True
    print('Ending and saving model')
    return

def terminate(signumb, frame):
    exit(2)

def enabledDeterminism():
    return bool(DETERMINISTIC)

def enabledSaveAndExit():
    return bool(SAVE_AND_EXIT_FLAG)

signal.signal(signal.SIGTSTP, saveWorkAndExit)

signal.signal(signal.SIGINT, terminate)

#reproducibility https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
# set in environment CUBLAS_WORKSPACE_CONFIG=':4096:2' or CUBLAS_WORKSPACE_CONFIG=':16:8'
def useDeterministic(torchSeed = 0, randomSeed = 0):
    DETERMINISTIC = True
    torch.cuda.manual_seed(torchSeed)
    torch.cuda.manual_seed_all(torchSeed)
    torch.manual_seed(torchSeed)
    numpy.random.seed(randomSeed)
    random.seed(randomSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(randomSeed)
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
    def __init__(self):
        self.only_Key_Ingredients = None
    """
    Child class should implement its own trySave, getFileSuffix(), canUpdate() methods.
    """

    def tryLoad(metadata, Class, classMetadataObj = None, temporaryLocation = False):
        suffix = Class.getFileSuffix()
        fileName = metadata.fileNameLoad
        path = None
        if(temporaryLocation):
            path = StaticData.TMP_PATH + fileName + suffix
        else:
            path = StaticData.PATH + fileName + suffix
        if fileName is not None and os.path.exists(path):
            toLoad = torch.load(path)
            loadedClassNameStr = toLoad['classNameStr']
            obj = toLoad['obj']
            obj.only_Key_Ingredients = None
            if(loadedClassNameStr == Class.__name__):
                print(Class.__name__ + ' loaded successfully')
                if(Class.canUpdate() == True):
                    obj.__update__(classMetadataObj)
                elif(classMetadataObj is not None):
                    print('There may be an error. Class: {} does not have corresponding metadata.'.format(Class.__name__))
                return obj
        print(Class.__name__ + ' load failure')
        return None

    def trySave(self, metadata, suffix: str, onlyKeyIngredients = False, temporaryLocation = False) -> bool:
        if(metadata.fileNameSave is None):
            print(type(self).__name__ + ' save not enabled')
            return False
        if metadata.fileNameSave is not None and os.path.exists(StaticData.PATH) and os.path.exists(StaticData.TMP_PATH):
            path = None
            if(temporaryLocation):
                path = StaticData.TMP_PATH + metadata.fileNameSave + suffix
            else:
                path = StaticData.PATH + metadata.fileNameSave + suffix
            self.only_Key_Ingredients = onlyKeyIngredients
            toSave = {
                'classNameStr': type(self).__name__,
                'obj': self
            }
            torch.save(toSave, path)
            self.only_Key_Ingredients = None
            print(type(self).__name__ + ' saved successfully')
            return True
        print(type(self).__name__ + ' save failure')
        return False

class BaseSampler:
    """
    Returns a sequence of the next indices.
    Supports saving and loading.
    """ 
    def __init__(self, dataSize, batchSize, startIndex = 0, seed = 984):
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


class Metadata(SaveClass):
    def __init__(self):
        super().__init__()
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
        tmp_str = ('\n/Metadata class\n-----------------------------------------------------------------------\n')
        tmp_str += ('Save path:\t{}\n'.format(StaticData.PATH + self.fileNameSave if self.fileNameSave is not None else 'Not set'))
        tmp_str += ('Load path:\t{}\n'.format(StaticData.PATH + self.fileNameLoad if self.fileNameLoad is not None else 'Not set'))
        tmp_str += ('Test flag:\t{}\n'.format(self.testFlag))
        tmp_str += ('Train flag:\t{}\n'.format(self.trainFlag))
        tmp_str += ('-----------------------------------------------------------------------\nEnd Metadata class\n')
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

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def trySave(self, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(self, StaticData.METADATA_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix():
        return StaticData.METADATA_SUFFIX

    def canUpdate():
        return False

    def shouldTrain(self):
        return bool(self.trainFlag)

    def shouldTest(self):
        return bool(self.testFlag)

class Timer(SaveClass):
    def __init__(self):
        super().__init__()
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
        obj = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del obj['timeStart']
            del obj['timeEnd']
            del obj['modelTimeSum']
            del obj['modelTimeCount']
        return obj

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.timeStart = None
            self.timeEnd = None
            self.modelTimeSum = 0
            self.modelTimeCount = 0

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.TIMER_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix():
        return StaticData.TIMER_SUFFIX

    def canUpdate():
        return False

class Output(SaveClass):
    def __init__(self):
        super().__init__()
        self.debugF = None
        self.modelF = None
        self.debugPath = None
        self.modelPath = None
        self.bash = False
        self.formatedLogF = None
        self.formatedLogPath = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['debugPath']
            del state['modelPath']
            del state['bash']
            del state['formatedLogPath']
        del state['debugF']
        del state['modelF']
        del state['formatedLogF']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.formatedLogF = None
        self.debugF = None
        self.modelF = None

        if(self.only_Key_Ingredients):
            self.debugPath = None
            self.modelPath = None
            self.bash = None
            self.formatedLogPath = None

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

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.OUTPUT_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix():
        return StaticData.OUTPUT_SUFFIX

    def canUpdate():
        return False


class Data_Metadata(SaveClass):
    DATA_PATH = '~/.data' # path to data

    def __init__(self):
        super().__init__()

        # default values:
        self.worker_seed = 841874
        
        self.train = True
        self.download = True
        self.pin_memoryTrain = False
        self.pin_memoryTest = False

        self.epoch = 1
        self.batchTrainSize = 4
        self.batchTestSize = 4

        # print = batch size * howOftenPrintTrain
        self.howOftenPrintTrain = 2000

    def tryPinMemoryTrain(self, metadata, modelMetadata):
        if(torch.cuda.is_available()):
            self.pin_memoryTrain = True
            if(metadata.debugInfo):
                print('Train data pinned to GPU: {}'.format(self.pin_memoryTrain))
        return bool(self.pin_memoryTrain)

    def tryPinMemoryTest(self, metadata, modelMetadata):
        if(torch.cuda.is_available()):
            self.pin_memoryTest = True
            if(metadata.debugInfo):
                print('Test data pinned to GPU: {}'.format(self.pin_memoryTest))
        return bool(self.pin_memoryTest)

    def __str__(self):
        tmp_str = ('Should train data:\t{}\n'.format(self.train))
        tmp_str += ('Download data:\t{}\n'.format(self.download))
        tmp_str += ('Pin memory train:\t{}\n'.format(self.pin_memoryTrain))
        tmp_str += ('Pin memory test:\t{}\n'.format(self.pin_memoryTest))
        tmp_str += ('Batch train size:\t{}\n'.format(self.batchTrainSize))
        tmp_str += ('Batch test size:\t{}\n'.format(self.batchTestSize))
        tmp_str += ('Number of epochs:\t{}\n'.format(self.epoch))
        tmp_str += ('How often print:\t{}\n'.format(self.howOftenPrintTrain))

    def _getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.DATA_METADATA_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix():
        return StaticData.DATA_METADATA_SUFFIX

    def canUpdate():
        return False

class Model_Metadata(SaveClass):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3 # TODO usunąć, bo to klasa podstawowa
        self.device = 'cuda:0'
        
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

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.MODEL_METADATA_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix():
        return StaticData.MODEL_METADATA_SUFFIX

    def canUpdate():
        return False


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
        self.statistics = Statistics()

class Statistics():
    """
    Klasa zwracana przez metodę epochLoop.
    """
    def __init__(self):
        super().__init__()
        self.trainLossArray = []

    def addLoss(self, loss):
        self.trainLossArray.append(loss)



class Data(SaveClass):
    """
        Metody konieczne do przeciążenia.
        __init__

        Metody konieczne do przeciążenia, dla których nie używa się super().
        __setInputTransform__
        __prepare__
        __update__
        __epoch__

        Metody możliwe do przeciążenia, które wymagają użycia super().
        __customizeState__
        __setstate__
        __str__

        Metody możliwe do przeciążenia, które nie wymagają użycia super(). Użycie go spowoduje zawarcie domyślnej treści danej metody (o ile taka istnieje), 
        która nie jest wymagana.
        __train__
        __beforeTrainLoop__
        __beforeTrain__
        __afterTrain__
        __afterTrainLoop__
        __test__
        __beforeTestLoop__
        __beforeTest__
        __afterTest__
        __afterTestLoop__
        __beforeEpochLoop__
        __afterEpochLoop__

        Metody, których nie powinno się przeciążać
        __getstate__
    """
    def __init__(self):
        super().__init__()
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
        if(self.only_Key_Ingredients):
            del state['batchNumbTrain']
            del state['batchNumbTest']
            del state['epochNumb']
            del state['trainHelper']
            del state['testHelper']
            del state['epochHelper']

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
        if(self.only_Key_Ingredients):
            self.batchNumbTrain = 0
            self.batchNumbTest = 0
            self.epochNumb = 0
            self.trainHelper = None
            self.testHelper = None
            self.epochHelper = None

        self.trainset = None
        self.trainloader = None
        self.testset = None
        self.testloader = None
        self.transform = None
        self.__setInputTransform__()

    def __setInputTransform__(self):
        """
        Wymaga przypisania wartości \n
            self.transform = ...
        """
        raise Exception("Not implemented")

    def __prepare__(self, dataMetadata: 'Data_Metadata'):
        """
        Używane przy pierwszym strojeniu.\n 
        Wymaga wywołania lub analogicznej akcji \n
            self.__setInputTransform__()\n\n
        Wymaga przypisania wartości \n
            self.trainset = ... \n
            self.testset = ...\n
            self.trainSampler = ...\n
            self.testSampler = ...\n
            self.trainloader = ...\n
            self.testloader = ...\n
        """
        raise Exception("Not implemented")

    def __train__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata, metadata: 'Metadata', smoothing: 'Smoothing'):
        """
        Główna logika treningu modelu. Następuje pomiar czasu dla wykonania danej metody.
        """
        
        # forward + backward + optimize
        outputs = model(helper.inputs)
        helper.loss = model.loss_fn(outputs, helper.labels)
        helper.loss.backward()
        model.optimizer.step()

        # run smoothing
        smoothing.call(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata)

    def setTrainLoop(self, model: 'Model', modelMetadata: 'Model_Metadata', metadata: 'Metadata'):
        helper = TrainDataContainer()
        helper.size = len(self.trainloader.dataset)
        model.train()
        metadata.prepareOutput()
        helper.timer = Timer()
        helper.loopTimer = Timer()
        helper.loss = None 
        helper.diff = None

        return helper

    def __beforeTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):      
        pass

    def __beforeTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def __afterTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def __afterTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        metadata.stream.print("Train summary:")
        metadata.stream.print(f" Average train time ({helper.timer.getUnits()}): {helper.timer.getAverage()}")
        metadata.stream.print(f" Loop train time ({helper.timer.getUnits()}): {helper.loopTimer.getDiff()}")

    def trainLoop(self, model: 'Model', helperEpoch: 'EpochDataContainer', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        """
        Główna logika pętli treningowej.
        """
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

    def __test__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        """
        Główna logika testu modelu. Następuje pomiar czasu dla wykonania danej metody.
        """
        
        helper.pred = model(helper.inputs)

    def setTestLoop(self, model: 'Model', modelMetadata: 'Model_Metadata', metadata: 'Metadata'):
        helper = TestDataContainer()
        helper.size = len(self.testloader.dataset)
        helper.test_loss, helper.test_correct = 0, 0
        model.eval()
        metadata.prepareOutput()
        helper.timer = Timer()
        helper.loopTimer = Timer()
        return helper

    def __beforeTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def __beforeTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def __afterTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def __afterTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def testLoop(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
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

    def __beforeEpochLoop__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def __afterEpochLoop__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def __epoch__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', 
        modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        """
        Reprezentuje pojedynczy epoch.
        Znajduje się tu cała logika epocha. Aby wykorzystać możliwość wyjścia i zapisu w danym momencie stanu modelu, należy zastosować konstrukcję:

        if(enabledSaveAndExit()):
            return 

        """
        raise Exception("Not implemented")

    def epochLoop(self, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
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
        stat = self.epochHelper.statistics
        self.epochHelper = None
        return stat

    def resetEpochState(self):
        self.batchNumbTrain = 0
        self.batchNumbTest = 0

    def resetFullEpochState(self):
        self.resetEpochState()
        self.epochNumb = 0

    def __update__(self, dataMetadata: 'Data_Metadata'):
        """
        Używane przy wczytaniu obiektu.\n 
        Wymaga wywołania lub analogicznej akcji \n
            self.__setInputTransform__()\n\n
        Wymaga przypisania wartości \n
            self.trainset = ... \n
            self.testset = ...\n
            self.trainSampler = ...\n
            self.testSampler = ...\n
            self.trainloader = ...\n
            self.testloader = ...\n
        """
        raise Exception("Not implemented")

    def trySave(self, metadata: 'Metadata', onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.DATA_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix():
        return StaticData.DATA_SUFFIX

    def canUpdate():
        return True

class Smoothing(SaveClass):
    def __init__(self):
        super().__init__()
        self.enabled = False # used only to prevent using smoothing when it is not set
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
        if(self.enabled == False):
            return
        self.counter += 1
        if(self.counter > self.numbOfBatchAfterSwitchOn):
            self.countWeights += 1
            helper.diff = {}
            with torch.no_grad():
                for key, arg in model.named_parameters():
                    self.sumWeights[key].add_(arg)
                    helper.diff[key] = arg.sub(self.previousWeights[key])
                    self.previousWeights[key].data.copy_(arg.data)

    def getWeights(self):
        """
        Zwraca słownik wag, który można użyć do załadowania ich do modelu. Wagi ładuje się standardową metodą torch.
        Może zwrócić pusty słownik, jeżeli obiekt nie jest gotowy do podania wag.
        """
        average = {}
        if(self.countWeights == 0 or self.enabled == False):
            return average
        for key, arg in self.sumWeights.items():
            average[key] = self.sumWeights[key] / self.countWeights
        return average

    def setDictionary(self, dictionary):
        """
        Used to map future weights into internal sums.
        """

        for key, values in dictionary:
            self.sumWeights[key] = torch.zeros_like(values, requires_grad=False)
            self.previousWeights[key] = torch.zeros_like(values, requires_grad=False)

        self.enabled = True

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

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.SMOOTHING_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix():
        return StaticData.SMOOTHING_SUFFIX

    def canUpdate():
        return False

    def saveMainWeight(self, model):
        self.mainWeights = model.getWeights()
        
    # ważne
    def getStateDict(self):
        #inheritance
        average = {}
        for key, arg in self.sumWeights.items():
            average[key] = self.sumWeights[key] / self.countWeights
        return average


class Model(nn.Module, SaveClass):
    """
        Metody, które wymagają przeciążenia bez wywołania super()
        __update__

        Metody, które wymagają przeciążenia z wywołaniem super()
        __init__

        Metody, które można przeciążyć i wymagają użycia super()
        __setstate__
    """
    def __init__(self, modelMetadata):
        nn.Module.__init__(self)
        SaveClass.__init__(self)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.eval()

    def setWeights(self, weights):
        self.load_state_dict(weights)

    def getWeights(self):
        return self.state_dict()
    
    def __update__(self, modelMetadata):
        raise Exception("Not implemented")

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.MODEL_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix():
        return StaticData.MODEL_SUFFIX

    def canUpdate():
        return True

def tryLoad(tupleClasses: list, metadata, temporaryLocation = False):
    dictObjs = {}
    dictObjs[type(metadata).__name__] = metadata
    if(dictObjs['Metadata'] is None):
        return None

    for mdcl, objcl in tupleClasses:
        # load class metadata
        if(mdcl is not None):
            dictObjs[mdcl.__name__] = SaveClass.tryLoad(metadata, mdcl, temporaryLocation=temporaryLocation)
            if(dictObjs[mdcl.__name__] is None):
                return None
            # load class
            dictObjs[objcl.__name__] = SaveClass.tryLoad(metadata, objcl, dictObjs[mdcl.__name__], temporaryLocation=temporaryLocation)
            if(dictObjs[objcl.__name__] is None):
                return None
        else:
            dictObjs[objcl.__name__] = SaveClass.tryLoad(metadata, objcl, temporaryLocation=temporaryLocation)
            if(dictObjs[objcl.__name__] is None):
                return None
    return dictObjs

def trySave(dictObjs: dict, onlyKeyIngredients = False, temporaryLocation = False):
    dictObjs['Metadata'].trySave(onlyKeyIngredients, temporaryLocation)
    md = dictObjs['Metadata']
    
    for key, obj in dictObjs.items():
        if(key != 'Metadata'):
            obj.trySave(md, onlyKeyIngredients, temporaryLocation)

def commandLineArg(metadata, dataMetadata, modelMetadata, argv, enableLoad = True):
    help = 'Help:\n'
    help += os.path.basename(__file__) + ' -h <help> [-s,--save] <file name to save> [-l,--load] <file name to load>'

    loadedMetadata = None

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
        Metadata.exitError(help)

    for opt, arg in opts:
        if opt in ('-l', '--load') and enableLoad:
            metadata.fileNameLoad = arg
            loadedMetadata = SaveClass.tryLoad(metadata, Metadata)
            if(loadedMetadata is None):
                break
            print("Command line options ignored because class Metadata was loaded.")
            return loadedMetadata, True

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(help)
            sys.exit()
        elif opt in ('-s', '--save'):
            metadata.fileNameSave = arg
        elif opt in ('-l', '--load'):
            continue
        elif opt in ('--test'):
            boolean = Metadata.onOff(arg)
            metadata.testFlag = boolean if boolean is not None else Metadata.exitError(help)
        elif opt in ('--train'):
            boolean = Metadata.onOff(arg)
            metadata.trainFlag = boolean if boolean is not None else Metadata.exitError(help)
        elif opt in ('--pinTest'):
            boolean = Metadata.onOff(arg)
            dataMetadata.tryPinMemoryTest(metadata, modelMetadata) if boolean is not None else Metadata.exitError(help)
        elif opt in ('--pinTrain'):
            boolean = Metadata.onOff(arg)
            dataMetadata.tryPinMemoryTrain(metadata, modelMetadata) if boolean is not None else Metadata.exitError(help)
        elif opt in ('-d', '--debug'):
            metadata.debugInfo = True
        elif opt in ('--debugOutput'):
            metadata.debugOutput = arg # debug output file path
        elif opt in ('--modelOutput'):
            metadata.modelOutput = arg # model output file path
        elif opt in ('--bashOutput'):
            boolean = Metadata.onOff(arg)
            metadata.bashFlag = boolean if boolean is not None else Metadata.exitError(help)
        elif opt in ('--formatedOutput'):
            metadata.formatedOutput = arg # formated output file path
        elif opt in ('--name'):
            metadata.name = arg
        else:
            print("Unknown flag provided to program: {}.".format(opt))

    if(metadata.modelOutput is None):
        metadata.modelOutput = 'default.log'

    if(metadata.debugOutput is None):
        metadata.debugOutput = 'default.log'  
    
    return metadata, False

def modelRun(Metadata_Class, Data_Metadata_Class, Model_Metadata_Class, Data_Class, Model_Class, Smoothing_Class):
    dictObjs = {}
    dictObjs[Metadata_Class.__name__] = Metadata_Class()
    dictObjs[Metadata_Class.__name__].prepareOutput()
    loadedSuccessful = False
    metadataLoaded = None

    dictObjs[Data_Metadata_Class.__name__] = Data_Metadata_Class()
    dictObjs[Model_Metadata_Class.__name__] = Model_Metadata_Class()

    dictObjs[Metadata_Class.__name__], metadataLoaded = commandLineArg(
        dictObjs[Metadata_Class.__name__], dictObjs[Data_Metadata_Class.__name__], dictObjs[Model_Metadata_Class.__name__], sys.argv[1:]
        )
    if(metadataLoaded): # if model should be loaded
        dictObjsTmp = tryLoad([(Data_Metadata_Class, Data_Class), (None, Smoothing_Class), (Model_Metadata_Class, Model_Class)], dictObjs[Metadata_Class.__name__])
        if(dictObjsTmp is None):
            loadedSuccessful = False
        else:
            dictObjs = dictObjsTmp
            dictObjs[Metadata_Class.__name__] = dictObjs[Metadata_Class.__name__]
            loadedSuccessful = True
            dictObjs[Metadata_Class.__name__].printContinueLoadedModel()

    if(loadedSuccessful == False):
        dictObjs[Metadata_Class.__name__].printStartNewModel()
        dictObjs[Data_Class.__name__] = Data_Class()
        
        dictObjs[Data_Class.__name__].__prepare__(dictObjs[Data_Metadata_Class.__name__])
        
        dictObjs[Smoothing_Class.__name__] = Smoothing_Class()
        dictObjs[Model_Class.__name__] = Model_Class(dictObjs[Model_Metadata_Class.__name__])

        dictObjs[Smoothing_Class.__name__].setDictionary(dictObjs[Model_Class.__name__].named_parameters())

    stat = dictObjs[Data_Class.__name__].epochLoop(
        dictObjs[Model_Class.__name__], dictObjs[Data_Metadata_Class.__name__], dictObjs[Model_Metadata_Class.__name__], 
        dictObjs[Metadata_Class.__name__], dictObjs[Smoothing_Class.__name__]
        )

    trySave(dictObjs)

    return stat

#########################################
# other functions
def cloneTorchDict(weights: dict):
    newDict = dict()
    for key, val in weights.items():
        newDict[key] = torch.clone(val)
    return newDict

def checkStrCUDA(string):
        return string.startswith('cuda')

#########################################
# test   
def modelDetermTest(Metadata_Class, Data_Metadata_Class, Model_Metadata_Class, Data_Class, Model_Class, Smoothing_Class):
    stat = []
    for i in range(2):
        dictObjs = {}
        dictObjs[Metadata_Class.__name__] = Metadata_Class()
        dictObjs[Metadata_Class.__name__].prepareOutput()
        loadedSuccessful = False

        dictObjs[Data_Metadata_Class.__name__] = Data_Metadata_Class()
        dictObjs[Model_Metadata_Class.__name__] = Model_Metadata_Class()

        dictObjs[Metadata_Class.__name__], _ = commandLineArg(dictObjs[Metadata_Class.__name__], dictObjs[Data_Metadata_Class.__name__], dictObjs[Model_Metadata_Class.__name__], sys.argv[1:], False)

        dictObjs[Metadata_Class.__name__].printStartNewModel()
        dictObjs[Data_Class.__name__] = Data_Class()
        
        dictObjs[Data_Class.__name__].__prepare__(dictObjs[Data_Metadata_Class.__name__])
        
        dictObjs[Smoothing_Class.__name__] = Smoothing_Class()
        dictObjs[Model_Class.__name__] = Model_Class(dictObjs[Model_Metadata_Class.__name__])

        dictObjs[Smoothing_Class.__name__].setDictionary(dictObjs[Model_Class.__name__].named_parameters())

        stat.append(
             dictObjs[Data_Class.__name__].epochLoop(dictObjs[Model_Class.__name__], dictObjs[Data_Metadata_Class.__name__], dictObjs[Model_Metadata_Class.__name__], dictObjs[Metadata_Class.__name__], dictObjs[Smoothing_Class.__name__])
        )
    equal = True
    for idx, (x) in enumerate(stat[0].trainLossArray):
        print(idx)
        if(torch.all(x.eq(stat[1].trainLossArray[idx])) == False):
            equal = False
            print(idx)
            print(x)
            print(stat[1].trainLossArray[idx])
            break
    print('Arrays are: ', equal)
    print(stat[0].trainLossArray)
    print(stat[1].trainLossArray)

if(__name__ == '__main__'):
    useDeterministic()
    stat = modelRun(Metadata, Data_Metadata, Model_Metadata, Data, Model, Smoothing)

    plt.plot(stat.trainLossArray)
    plt.xlabel('Train index')
    plt.ylabel('Loss')
    plt.show()
