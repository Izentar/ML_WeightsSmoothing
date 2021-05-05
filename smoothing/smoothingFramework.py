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
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy

SAVE_AND_EXIT_FLAG = False
DETERMINISTIC = False
PRINT_WARNINGS = True
FORCE_PRINT_WARNINGS = False


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

def warnings():
    return FORCE_PRINT_WARNINGS or PRINT_WARNINGS

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
    DATA_PATH = '~/.data'
    PREDEFINED_MODEL_SUFFIX = '.pdmodel'
    IGNORE_IO_WARNINGS = False
    TEST_MODE = False
    MAX_LOOPS = 31

class SaveClass:
    def __init__(self):
        self.only_Key_Ingredients = None
    """
    Child class should implement its own trySave, getFileSuffix(self = None), canUpdate() methods.
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
                Output.printBash(Class.__name__ + ' loaded successfully', 'info')
                if(Class.canUpdate() == True):
                    obj.__update__(classMetadataObj)
                elif(classMetadataObj is not None):
                    Output.printBash('There may be an error. Class: {} does not have corresponding metadata.'.format(Class.__name__), 'warn')
                return obj
        Output.printBash(Class.__name__ + ' load failure', 'info')
        return None

    def trySave(self, metadata, suffix: str, onlyKeyIngredients = False, temporaryLocation = False) -> bool:
        if(metadata.fileNameSave is None):
            Output.printBash(type(self).__name__ + ' save not enabled', 'info')
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
            Output.printBash(type(self).__name__ + ' saved successfully', 'info')
            return True
        Output.printBash(type(self).__name__ + ' save failure', 'info')
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

class test_mode():
    def __init__(self):
        self.prev = False

    def __enter__(self):
        self.prev = StaticData.TEST_MODE
        StaticData.TEST_MODE = True

    def __exit__(self, exc_type, exc_val, exc_traceback):
        StaticData.TEST_MODE = self.prev

    def isActivated(self = None):
        return bool(StaticData.TEST_MODE)

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

        self.noPrepareOutput = False

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
            string = f"\n\n@@@@\nStarting new model: " + self.name + "\nTime: " + str(datetime.now()) + "\n@@@@\n"
            self.stream.print(string)
            Output.printBash(string)
        else:
            string = f"\n\n@@@@\nStarting new model without name\nTime: " + str(datetime.now()) + "\n@@@@\n"
            self.stream.print(string)
            Output.printBash(string)

    def printContinueLoadedModel(self):
        if(self.stream is None):
            raise Exception("Stream not initialized")
        if(self.name is not None):
            string = f"\n\n@@@@\nContinuation of the loaded model: '" + self.name + "'\nTime: " + str(datetime.now()) + "\n@@@@\n"
            self.stream.print(string)
            Output.printBash(string)
        else:
            string = f"\n\n@@@@\nContinuation of loaded model without name\nTime: " + str(datetime.now()) + "\n@@@@\n"
            self.stream.print(string)
            Output.printBash(string)

    def printEndModel(self):
        if(self.stream is None):
            raise Exception("Stream not initialized")
        if(self.name is not None):
            string = f"\n\n@@@@\nEnding model: " + self.name + "\nTime: " + str(datetime.now()) + "\n@@@@\n"
            self.stream.print(string)
            Output.printBash(string)
        else:
            string = f"\n\n@@@@\nEnding model\nTime: " + str(datetime.now()) + "\n@@@@\n"
            self.stream.print(string)
            Output.printBash(string)

    def exitError(help):
        print(help) 
        sys.exit(2)

    def prepareOutput(self):
        """
        Spróbowano stworzyć aliasy:\n
        debug:0\n
        model:0\n
        stat\n
        oraz spróbowano otworzyć tryb 'bash'
        """
        if(self.noPrepareOutput):
            return
        Output.printBash('Preparing default output.', 'info')
        if(self.stream is None):
            self.stream = Output()

        if(self.debugInfo == True):
            if(self.debugOutput is not None):
                self.stream.open('debug', 'debug:0', self.debugOutput)
        if(self.modelOutput is not None):
            self.stream.open('model', 'model:0', self.modelOutput)
        if(self.bashFlag == True):
            self.stream.open('bash')
        if(self.formatedOutput is not None):
            self.stream.open('formatedLog', 'stat', self.formatedOutput)
        Output.printBash('Default outputs prepared.', 'info')

        self.noPrepareOutput = True

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.noPrepareOutput = False
        self.prepareOutput()

    def trySave(self, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(self, StaticData.METADATA_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.METADATA_SUFFIX

    def canUpdate(self = None):
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

    def getFileSuffix(self = None):
        return StaticData.TIMER_SUFFIX

    def canUpdate(self = None):
        return False

class Output(SaveClass):

    class FileHandler():
        def __init__(self, fullPathName, mode, OType):
            self.handler = open(fullPathName, mode)
            self.counter = 1
            self.pathName = fullPathName
            self.mode = mode
            self.OType = [OType]

        def __getstate__(self):
            state = self.__dict__.copy()
            del state['handler']
            return state

        def __setstate__(self, state):
            self.__dict__.update(state)
            self.handler = open(self.pathName, self.mode)

        def counterUp(self):
            self.counter +=1
            return self

        def get(self):
            if(self.handler is None):
                raise Exception("Tried to get to the file that is already closed.")
            return self.handler

        def addOType(self, OType):
            self.OType.append(OType)
            return self

        def exist(self):
            return bool(self.handler is not None)

        def close(self):
            if(self.counter <= 1):
                self.handler.close()
                self.handler = None
            self.counter -= 1

        def flush(self):
            if(self.handler is not None):
                self.handler.flush()
            return self

        def __del__(self):
            if(self.handler is not None):
                self.handler.close()
                self.counter = 0

    def __init__(self):
        super().__init__()
        self.filesDict = {}
        self.aliasToFH = {}

        self.bash = False

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __open(self, alias, pathName, outputType):
        if(alias in self.aliasToFH and self.aliasToFH[alias].exist()):
            if(warnings()):
                print("WARNING: Provided alias '{}' with opened file already exist: {}.".format(alias, outputType), 'This may be due to loaded Metadata object.')
            return
        suffix = '.log'
        if(outputType == 'formatedLog'):
            suffix = '.csv'
        fh = self.FileHandler(pathName + suffix, 'a', outputType)
        self.filesDict[pathName] = {outputType: fh}
        self.aliasToFH[alias] = fh

    def open(self, outputType: str, alias: str = None, pathName: str = None):
        if((outputType != 'debug' and outputType != 'model' and outputType != 'bash' and outputType != 'formatedLog') or outputType is None):
            if(warnings()):
                print("WARNING: Unknown command in open for Output class.")
            return

        if(alias == 'debug' or alias == 'model' or alias == 'bash' or alias == 'formatedLog'):
            if(warnings()):
                print("WARNING: Alias cannot have the same name as output configuration in open for Output class.")
            return

        if(outputType == 'bash'):
            self.bash = True
            return

        if(alias is None):
            if(warnings()):
                print("WARNING: Alias is None but the output is not 'bash'; it is: {}.".format(outputType))
            return

        if(pathName is not None):
            if(pathName in self.filesDict):
                if(outputType in self.filesDict[pathName] and self.filesDict[pathName][outputType].exist()):
                    pass # do nothing, already opened
                elif(self.filesDict[pathName][outputType] is None):
                    for _, val in self.filesDict[pathName]:
                        if(val.exist()): # copy reference
                            if(outputType == 'formatedLog'):
                                raise Exception("Output for type 'formatedLog' for provided pathName can have only one instance. For this pathName file in different mode is already opened.")
                            self.filesDict[pathName][outputType] = self.filesDict[pathName][self.filesDict[pathName].keys()[-1]].counterUp().addOType(outputType)
                            return
                        else:
                            del val
            self.__open(alias, pathName, outputType)
        else:
            if(warnings()):
                print("WARNING: For this '{}' Output type pathName should not be None.".format(outputType))
            return
    
    def __getPrefix(mode):
        prefix = ''
        if(mode is None):
            pass
        elif(mode == 'info'):
            prefix = 'INFO:'
        elif(mode == 'debug'):
            prefix = 'DEBUG:'
        elif(mode == 'warn'):
            prefix = 'WARNING:'
        elif(warnings()):
            print("WARNING: Unrecognized mode in Output.printBash method. Printing without prefix.")
        return prefix

    def write(self, arg, alias: list = None, ignore = False, end = '', mode: str = None):
        """
        Przekazuje argument do wszystkich możliwych, aktywnych strumieni wyjściowych.\n
        Na końcu argumentu nie daje znaku nowej linii.
        """
        prefix = Output.__getPrefix(mode)

        if(alias is None):
            for fh in self.aliasToFH.values():
                if(fh.exist() and 'formatedLog' not in fh.OType):
                    fh.get().write(str(arg) + end)
        else:
            for al in alias:
                if(al == 'bash'):
                    print(arg, end=end)
                if(al in self.aliasToFH.keys() and self.aliasToFH[al].exist()):
                    self.aliasToFH[al].get().write(str(arg) + end)
                    if('formatedLog' in self.aliasToFH[al].OType):
                        prBash = False
                elif(warnings() and not (ignore or StaticData.IGNORE_IO_WARNINGS)):
                    print("WARNING: Output alias for 'write / print' not found: '{}'".format(al), end=end)
                    print(al, self.aliasToFH.keys())
                
    def print(self, arg, alias: list = None, ignore = False, mode: str = None):
        """
        Przekazuje argument do wszystkich możliwych, aktywnych strumieni wyjściowych.\n
        Na końcu argumentu dodaje znak nowej linii.
        """
        self.write(arg, alias, ignore, '\n', mode)

    def __del__(self):
        for _, fh in self.aliasToFH.items():
            del fh

    def flushAll(self):
        for _, fh in self.aliasToFH.items():
            fh.flush()
        sys.stdout.flush()

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.OUTPUT_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.OUTPUT_SUFFIX

    def canUpdate(self = None):
        return False

    def printBash(arg, mode: str = None):
        """
        Tryby:\n
        'info'
        'debug'
        'warn'
        None
        """
        prefix = Output.__getPrefix(mode)
        print(prefix, arg)

class DefaultMethods():
    def printLoss(metadata, helper, alias: list = None):
        """
        Potrzebuje:\n
        helper.loss\n
        helper.batchNumber\n
        helper.inputs\n
        helper.size\n
        """
        calcLoss, current = helper.loss.item(), helper.batchNumber * len(helper.inputs)
        metadata.stream.print(f"loss: {calcLoss:>7f}  [{current:>5d}/{helper.size:>5d}]", alias)
        del calcLoss
        del current

    def printWeightDifference(metadata, helper, alias: list = None):
        """
        Potrzebuje\n
        helper.diff
        """
        if(helper.diff is None):
            metadata.stream.print(f"No weight difference")
        else:
            diffKey = list(helper.diff.keys())[-1]
            metadata.stream.print(f"Weight difference: {helper.diff[diffKey]}", 'debug:0', 'bash:0', alias)
            metadata.stream.print(f"Weight difference of last layer average: {helper.diff[diffKey].sum() / helper.diff[diffKey].numel()} :: was divided by: {helper.diff[diffKey].numel()}", alias)
            metadata.stream.print('################################################', alias)
            del diffKey

class LoopsState():
    def __init__(self):
        self.numbArray = []
        self.popNumbArray = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['popNumbArray']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.popNumbArray = None

    def imprint(self, numb, isEnd):
        if(len(self.popNumbArray) == 1 and self.popNumbArray[0][1] == False):
            self.numbArray[0][0] = numb
            self.numbArray[0][0] = True
            self.popNumbArray.clear()
        elif(len(self.popNumbArray) == 0):
            self.numbArray.append([numb, isEnd])
            self.popNumbArray.clear()

    def clear(self):
        self.popNumbArray = None
        self.numbArray = []

    def tryCreateNew(self):
        if(self.popNumbArray is None):
            self.popNumbArray = self.numbArray.copy()

    def canRun(self):
        if(self.popNumbArray is None):
            raise Exception("State not started")
        if(len(self.popNumbArray) != 0):
            numb, finished = self.popNumbArray[0]
            if(finished):
                self.popNumbArray.pop(0)
                return None # go to the next loop
            else:
                return numb # start loop with this number
        else:
            return 0 # start loop from 0

    def decide(self):
        self.tryCreateNew()
        return self.canRun()


class Data_Metadata(SaveClass):
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

    def getFileSuffix(self = None):
        return StaticData.DATA_METADATA_SUFFIX

    def canUpdate(self = None):
        return False

class Model_Metadata(SaveClass):
    def __init__(self):
        super().__init__()

    def _getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.MODEL_METADATA_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.MODEL_METADATA_SUFFIX

    def canUpdate(self = None):
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

        self.batchNumber = None # current batch number
        self.loopEnded = False # check if loop ened

class TestDataContainer():
    def __init__(self):
        self.size = None
        self.timer = None
        self.loopTimer = None

        # one loop test data
        self.pred = None
        self.inputs = None
        self.labels = None

        self.batchNumber = None # current batch number
        self.loopEnded = False # check if loop ened

        # one loop test data
        self.test_loss = 0
        self.test_correct = 0
        self.testLossSum = 0
        self.testCorrectSum = 0

class EpochDataContainer():
    def __init__(self):
        self.epochNumber = None
        self.returnObj = None
        self.loopsState = LoopsState()
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
        __trainLoopExit__
        __testLoopExit__


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
        __epochLoopExit__

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
            self.trainTransform = ...\n
            self.testTransform = ...

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
        outputs = model.getNNModelModule()(helper.inputs)
        helper.loss = model.loss_fn(outputs, helper.labels)
        del outputs
        #print(torch.cuda.memory_summary())
        helper.loss.backward()
        #print(torch.cuda.memory_summary())
        model.optimizer.step()

        # run smoothing
        smoothing(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata)

    def setTrainLoop(self, model: 'Model', modelMetadata: 'Model_Metadata', metadata: 'Metadata'):
        helper = TrainDataContainer()
        metadata.prepareOutput()
        helper.size = len(self.trainloader.dataset)
        model.getNNModelModule().train()
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

    def __trainLoopExit__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helperEpoch.loopsState.imprint(helper.batchNumber, helper.loopEnded)

    def trainLoop(self, model: 'Model', helperEpoch: 'EpochDataContainer', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        """
        Główna logika pętli treningowej.
        """
        startNumb = helperEpoch.loopsState.decide()
        if(startNumb is None):
            return # loop already ended. This state can occur when framework was loaded from file.

        if(self.trainHelper is None): # jeżeli nie było wznowione; nowe wywołanie
            self.trainHelper = self.setTrainLoop(model, modelMetadata, metadata)

        torch.cuda.empty_cache()
        self.__beforeTrainLoop__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

        self.trainHelper.loopTimer.start()
        for batch, (inputs, labels) in enumerate(self.trainloader, start=startNumb):
            del self.trainHelper.inputs
            del self.trainHelper.labels

            self.trainHelper.inputs = inputs
            self.trainHelper.labels = labels
            self.trainHelper.batchNumber = batch
            if(SAVE_AND_EXIT_FLAG):
                self.__trainLoopExit__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
                return

            if(StaticData.TEST_MODE and batch >= StaticData.MAX_LOOPS):
                break
            
            self.__beforeTrain__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
            
            del self.trainHelper.loss

            self.trainHelper.timer.clearTime()
            self.trainHelper.timer.start()
            

            self.__train__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

            self.trainHelper.timer.end()
            self.trainHelper.timer.addToStatistics()

            self.__afterTrain__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

        self.trainHelper.loopTimer.end()
        self.trainHelper.loopEnded = True

        self.__afterTrainLoop__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
        self.__trainLoopExit__(helperEpoch, self.trainHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
        self.trainHelper = None
        
    def __test__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        """
        Główna logika testu modelu. Następuje pomiar czasu dla wykonania danej metody.
        """
        
        helper.pred = model.getNNModelModule()(helper.inputs)
        helper.test_loss = model.loss_fn(helper.pred, helper.labels).item()

    def setTestLoop(self, model: 'Model', modelMetadata: 'Model_Metadata', metadata: 'Metadata'):
        helper = TestDataContainer()
        metadata.prepareOutput()
        helper.size = len(self.testloader.dataset)
        helper.test_loss, helper.test_correct = 0, 0
        model.getNNModelModule().eval()
        helper.timer = Timer()
        helper.loopTimer = Timer()
        return helper

    def __beforeTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def __beforeTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def __afterTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helper.testLossSum += helper.test_loss
        helper.test_correct = (helper.pred.argmax(1) == helper.labels).type(torch.float).sum().item()
        helper.testCorrectSum += helper.test_correct

    def __afterTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

    def __testLoopExit__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helperEpoch.loopsState.imprint(helper.batchNumber, helper.loopEnded)

    def testLoop(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        startNumb = helperEpoch.loopsState.decide()
        if(startNumb is None):
            return # loop already ended. This state can occur when framework was loaded from file.
        
        if(self.testHelper is None): # jeżeli nie było wznowione; nowe wywołanie
            self.testHelper = self.setTestLoop(model, modelMetadata, metadata)

        torch.cuda.empty_cache()
        self.__beforeTestLoop__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

        with torch.no_grad():
            self.testHelper.loopTimer.start()
            for batch, (inputs, labels) in enumerate(self.testloader, startNumb):
                self.testHelper.inputs = inputs
                self.testHelper.labels = labels
                self.testHelper.batchNumber = batch
                if(SAVE_AND_EXIT_FLAG):
                    self.__testLoopExit__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
                    return

                if(StaticData.TEST_MODE and batch >= StaticData.MAX_LOOPS):
                    break

                self.__beforeTest__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

                self.testHelper.timer.clearTime()
                self.testHelper.timer.start()
                self.__test__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
                self.testHelper.timer.end()
                self.testHelper.timer.addToStatistics()

                self.__afterTest__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)

            self.testHelper.loopTimer.end()
            self.testHelper.loopEnded = True

        self.__afterTestLoop__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
        self.__testLoopExit__(helperEpoch, self.testHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
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

    def __epochLoopExit__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        pass

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
                self.__epochLoopExit__(self.epochHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
                return

            if(StaticData.TEST_MODE and ep == 3):
                break

        self.__afterEpochLoop__(self.epochHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
        self.__epochLoopExit__(self.epochHelper, model, dataMetadata, modelMetadata, metadata, smoothing)
        self.resetEpochState()
        metadata.stream.flushAll()
        stat = self.epochHelper.statistics
        self.epochHelper = None
        return stat


    def resetEpochState(self):
        self.epochHelper.loopsState.clear()
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

    def getFileSuffix(self = None):
        return StaticData.DATA_SUFFIX

    def canUpdate(self = None):
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

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata):
        if(self.enabled == False):
            return

    def getWeights(self):
        """
        Zwraca słownik wag, który można użyć do załadowania ich do modelu. Wagi ładuje się standardową metodą torch.
        Może zwrócić pusty słownik, jeżeli obiekt nie jest gotowy do podania wag.
        """
        pass

    def setDictionary(self, dictionary):
        """
        Used to map future weights into internal sums.
        """
        pass

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

    def getFileSuffix(self = None):
        return StaticData.SMOOTHING_SUFFIX

    def canUpdate(self = None):
        return False

    def saveMainWeight(self, model):
        self.mainWeights = model.getWeights()
        

class Model(nn.Module, SaveClass):
    """
        Klasa służąca do tworzenia nowych modeli.
        Aby bezpiecznie skorzystać z metod nn.Module należy wywołać metodę getNNModelModule(), która zwróci obiekt typu nn.Module

        Metody, które wymagają przeciążenia bez wywołania super()
        __update__
        __initializeWeights__

        Metody, które wymagają przeciążenia z wywołaniem super()
        __init__

        Metody, które można przeciążyć i wymagają użycia super()
        __setstate__

        Klasa powinna posiadać zmienne
        self.loss_fn = ...
        self.optimizer = torch.optim...
    """
    def __init__(self, modelMetadata):
        """
        Metoda powinna posiadać zmienne\n
        self.loss_fn = ...\n
        self.optimizer = torch.optim...\n
        \n
        Na końcu _\_init__ powinno się zainicjalizować wagi metodą\n
        def __initializeWeights__(self)\n
        """
        super().__init__()

    def __initializeWeights__(self):
        raise Exception("Not implemented")

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

    def getFileSuffix(self = None):
        return StaticData.MODEL_SUFFIX

    def canUpdate(self = None):
        return True

    def getNNModelModule(self):
        """
        Używany, gdy chcemy skorzystać z funckji modułu nn.Module. Zwraca obiekt dla którego jest pewność, że implementuje klasę nn.Module. 
        """
        return self

class PredefinedModel(SaveClass):
    """
        Klasa używana do kapsułkowania istniejących już obiektów predefiniowanych modeli.
        Aby bezpiecznie skorzystać z metod nn.Module należy wywołać metodę getNNModelModule(), która zwróci obiekt typu nn.Module

        Metody, które wymagają przeciążenia bez wywołania super()
        __update__

        Metody, które wymagają przeciążenia z wywołaniem super()
        __init__

        Metody, które można przeciążyć i wymagają użycia super()
        __getstate__
        __setstate__

        Klasa powinna posiadać zmienne
        self.loss_fn = ...
        self.optimizer = torch.optim...
    """
    def __init__(self, obj: 'modelObject', modelMetadata):
        """
        Metoda powinna posiadać zmienne\n
        self.loss_fn = ...\n
        self.optimizer = torch.optim...\n
        """

        if(not isinstance(obj, nn.Module)):
            raise Exception("Object do not implement nn.Module class.")
        super().__init__()
        self.modelObj = obj

    def __initializeWeights__(self):
        self.modelObj._initialize_weights()

    def __getstate__(self):
        obj = self.__dict__.copy()
        return {
            'obj': obj,
            'classType': type(self)
        }

    def __setstate__(self, state):
        obj = state['obj']
        if(state['classType'] != type(self)):
            raise Exception("Loaded object '{}' is not the same class as PredefinedModel class '{}'.".format(str(state['classType']), str(type(self))))
        self.__dict__.update(obj)
        self.modelObj.eval()

    def setWeights(self, weights):
        self.modelObj.load_state_dict(weights)

    def getWeights(self):
        return self.modelObj.state_dict()

    def __update__(self, modelMetadata):
        raise Exception("Not implemented")

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata, StaticData.PREDEFINED_MODEL_SUFFIX, onlyKeyIngredients, temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.PREDEFINED_MODEL_SUFFIX

    def getNNModelModule(self):
        """
        Używany, gdy chcemy skorzystać z funckji modułu nn.Module. Zwraca obiekt dla którego jest pewność, że implementuje klasę nn.Module. 
        """
        return self.modelObj

    def canUpdate(self = None):
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
            Output.printBash("Command line options ignored because class Metadata was loaded.", 'info')
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
    
    metadata.noPrepareOutput = False

    return metadata, False

def modelRun(Metadata_Class, Data_Metadata_Class, Model_Metadata_Class, Data_Class, Model_Class, Smoothing_Class, modelObj = None):
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
        if(issubclass(Model_Class, PredefinedModel)):
            if(modelObj is not None):
                dictObjs[Model_Class.__name__] = Model_Class(modelObj, dictObjs[Model_Metadata_Class.__name__])
            else:
                raise Exception("Predefined model to be created needs object.")
        else:
            dictObjs[Model_Class.__name__] = Model_Class(dictObjs[Model_Metadata_Class.__name__])

        dictObjs[Smoothing_Class.__name__].setDictionary(dictObjs[Model_Class.__name__].getNNModelModule().named_parameters())

    stat = dictObjs[Data_Class.__name__].epochLoop(
        dictObjs[Model_Class.__name__], dictObjs[Data_Metadata_Class.__name__], dictObjs[Model_Metadata_Class.__name__], 
        dictObjs[Metadata_Class.__name__], dictObjs[Smoothing_Class.__name__]
        )
    dictObjs[Metadata_Class.__name__].printEndModel()

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

def trySelectCUDA(device, metadata):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if(metadata.debugInfo):
        print('Using {} torch CUDA device version\nCUDA avaliable: {}\nCUDA selected: {}'.format(torch.version.cuda, torch.cuda.is_available(), self.device == 'cuda'))
    return device

def selectCPU(device, metadata):
    device = 'cpu'
    if(metadata.debugInfo):
        print('Using {} torch CUDA device version\nCUDA avaliable: {}\nCUDA selected: False'.format(torch.version.cuda, torch.cuda.is_available()))
    return device

#########################################
# test   

def modelDetermTest(Metadata_Class, Data_Metadata_Class, Model_Metadata_Class, Data_Class, Model_Class, Smoothing_Class, modelObj = None):
    """
    Można użyć do przetestowania, czy dany model jest deterministyczny.
    Zmiana z CPU na GPU nadal zachowuje determinizm.
    """
    stat = []
    for i in range(2):
        useDeterministic()
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
        if(issubclass(Model_Class, PredefinedModel)):
            if(modelObj is not None):
                dictObjs[Model_Class.__name__] = Model_Class(modelObj, dictObjs[Model_Metadata_Class.__name__])
            else:
                raise Exception("Predefined model to be created needs object.")
        else:
            dictObjs[Model_Class.__name__] = Model_Class(dictObjs[Model_Metadata_Class.__name__])

        dictObjs[Smoothing_Class.__name__].setDictionary(dictObjs[Model_Class.__name__].getNNModelModule().named_parameters())

        stat.append(
             dictObjs[Data_Class.__name__].epochLoop(dictObjs[Model_Class.__name__], dictObjs[Data_Metadata_Class.__name__], dictObjs[Model_Metadata_Class.__name__], dictObjs[Metadata_Class.__name__], dictObjs[Smoothing_Class.__name__])
        )
        dictObjs[Metadata_Class.__name__].printEndModel()

    equal = True
    for idx, (x) in enumerate(stat[0].trainLossArray):
        if(torch.is_tensor(x) and torch.equal(x, stat[1].trainLossArray[idx]) == False):
            equal = False
            print(idx)
            print(x)
            print(stat[1].trainLossArray[idx])
            break
    print('Arrays are: ', equal)

if(__name__ == '__main__'):
    stat = modelRun(Metadata, Data_Metadata, Model_Metadata, Data, Model, Smoothing)

    plt.plot(stat.trainLossArray)
    plt.xlabel('Train index')
    plt.ylabel('Loss')
    plt.show()
