import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys, getopt
from os.path import expanduser
import signal
from datetime import datetime
import time
import random
from pathlib import Path
import pandas as pd
import errno

import matplotlib.pyplot as plt
import numpy

SAVE_AND_EXIT_FLAG = False
DETERMINISTIC = False
PRINT_WARNINGS = True
FORCE_PRINT_WARNINGS = False


def saveWorkAndExit(signumb, frame):
    global SAVE_AND_EXIT_FLAG
    SAVE_AND_EXIT_FLAG = True
    Output.printBash('Ending and saving model', 'info')
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
    SMOOTHING_METADATA_SUFFIX = '.smthmd'
    NAME_CLASS_METADATA = 'Metadata'
    DATA_PATH = '~/.data'
    PREDEFINED_MODEL_SUFFIX = '.pdmodel'
    LOG_FOLDER = './savedLogs/'
    IGNORE_IO_WARNINGS = False
    TEST_MODE = False
    MAX_DEBUG_LOOPS = 51

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

class BaseMainClass:
    def __strAppend__(self):
        return ""

    def __str__(self):
        tmp_str = ('\nStart {} class\n-----------------------------------------------------------------------\n'.format(type(self).__name__))
        tmp_str += self.__strAppend__()
        tmp_str += ('-----------------------------------------------------------------------\nEnd {} class\n'.format(type(self).__name__))
        return tmp_str

class BaseLogicClass:
    def createDefaultMetadataObj(self):
        raise Exception("Not implemented")

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

class Metadata(SaveClass, BaseMainClass):
    def __init__(self, fileNameSave=None, fileNameLoad=None, testFlag=False, trainFlag=False, debugInfo=False, modelOutput=None,
            debugOutput=None, stream=None, bashFlag=False, name=None, formatedOutput=None, logFolderSuffix=None):
        super().__init__()
        self.fileNameSave = fileNameSave
        self.fileNameLoad = fileNameLoad

        self.testFlag = testFlag
        self.trainFlag = trainFlag

        self.debugInfo = debugInfo
        self.modelOutput = modelOutput
        self.debugOutput = debugOutput
        self.stream = stream
        self.bashFlag = bashFlag
        self.name = name
        self.formatedOutput = formatedOutput

        self.logFolderSuffix = logFolderSuffix

        # zmienne wewnętrzne
        self.noPrepareOutput = False

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Save path:\t{}\n'.format(StaticData.PATH + self.fileNameSave if self.fileNameSave is not None else 'Not set'))
        tmp_str += ('Load path:\t{}\n'.format(StaticData.PATH + self.fileNameLoad if self.fileNameLoad is not None else 'Not set'))
        tmp_str += ('Test flag:\t{}\n'.format(self.testFlag))
        tmp_str += ('Train flag:\t{}\n'.format(self.trainFlag))
        tmp_str += ('Debug info flag:\t{}\n'.format(self.debugInfo))
        tmp_str += ('Model output path:\t{}\n'.format(self.modelOutput))
        tmp_str += ('Debug output path:\t{}\n'.format(self.debugOutput))
        tmp_str += ('Bash flag:\t{}\n'.format(self.bashFlag))
        tmp_str += ('Formated output name:\t{}\n'.format(self.formatedOutput))
        tmp_str += ('Folder sufix name:\t{}\n'.format(self.logFolderSuffix))
        tmp_str += ('Output is prepared flag:\t{}\n'.format(self.noPrepareOutput))
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
        self.prepareOutput()
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
        self.prepareOutput()
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
        self.prepareOutput()
        if(self.name is not None):
            string = f"\n\n@@@@\nEnding model: " + self.name + "\nTime: " + str(datetime.now()) + "\n@@@@\n"
            self.stream.print(string)
            Output.printBash(string)
        else:
            string = f"\n\n@@@@\nEnding model\nTime: " + str(datetime.now()) + "\n@@@@\n"
            self.stream.print(string)
            Output.printBash(string)

    def exitError(help):
        Output.printBash(help, 'info') 
        sys.exit(2)

    def prepareOutput(self):
        """
        Spróbowano stworzyć alias:
            debug:0
        Stworzono aliasy:
            model:0
            stat
            loopTrainTime
            loopTestTime_normal
            loopTestTime_smooothing
            statLossTrain
            statLossTest_normal
            statLossTest_smooothing
            weightsSumTrain
        oraz otwarto tryb 
            'bash'
        """
        if(self.noPrepareOutput):
            return
        Output.printBash('Preparing default output.', 'info')
        if(self.stream is None):
            self.stream = Output(self.logFolderSuffix)

        if(self.debugInfo == True):
            self.stream.open('debug', 'debug:0', 'debug')
        self.stream.open('model', 'model:0', 'model')
        self.stream.open('bash')
        self.stream.open('formatedLog', 'stat', 'statistics')

        self.stream.open('formatedLog', 'loopTrainTime', 'loopTrainTime')
        self.stream.open('formatedLog', 'loopTestTime_normal', 'loopTestTime_normal')
        self.stream.open('formatedLog', 'loopTestTime_smooothing', 'loopTestTime_smooothing')

        self.stream.open('formatedLog', 'statLossTrain', 'statLossTrain')
        self.stream.open('formatedLog', 'statLossTest_normal', 'statLossTest_normal')
        self.stream.open('formatedLog', 'statLossTest_smooothing', 'statLossTest_smooothing')

        self.stream.open('formatedLog', 'weightsSumTrain', 'weightsSumTrain')

        Output.printBash('Default outputs prepared.', 'info')

        self.noPrepareOutput = True

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.noPrepareOutput = False
        self.prepareOutput()

    def trySave(self, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata=self, suffix=StaticData.METADATA_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

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
        torch.cuda.synchronize()
        self.timeStart = time.perf_counter()

    def end(self):
        torch.cuda.synchronize()
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

    def getTimeSum(self):
        return self.modelTimeSum

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
        return super().trySave(metadata=metadata, suffix=StaticData.TIMER_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.TIMER_SUFFIX

    def canUpdate(self = None):
        return False

class Output(SaveClass):
    """
    Instancja tego obiektu odpowiada instancji jednego folderu, w którym będą się znajdowały wszystkie otwarte pliki.
    """
    class FileHandler():
        def __init__(self, root, pathName, mode, OType):
            if not os.path.exists(os.path.dirname(root + pathName)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            self.handler = open(root + pathName, mode)
            self.counter = 1
            self.pathName = pathName
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

    def __init__(self, folderSuffix):
        super().__init__()
        self.filesDict = {}
        self.aliasToFH = {}

        self.bash = False

        Path(StaticData.LOG_FOLDER).mkdir(parents=True, exist_ok=True)

        self.folderSuffix = folderSuffix
        self.root = None
        self.currentDefaultAlias = None

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def setDefaultAlias(self, name):
        if(name not in self.aliasToFH):
            self.printBash("Setting default alias in Output failed. Could not find '{}' in existing list: {}".format(name, list(self.aliasToFH.keys())), 'warn')
            return False
        self.currentDefaultAlias = name
        return True

    def __open(self, alias, root, pathName, outputType):
        if(alias in self.aliasToFH and self.aliasToFH[alias].exist()):
            if(warnings()):
                Output.printBash("Provided alias '{}' with opened file already exist: {}.".format(alias, outputType), 'This may be due to loaded Metadata object.',
                'warn')
            return
        suffix = '.log'
        if(outputType == 'formatedLog'):
            suffix = '.csv'
        fh = self.FileHandler(root, pathName + suffix, 'a', outputType)
        self.filesDict[pathName] = {outputType: fh}
        self.aliasToFH[alias] = fh

    def open(self, outputType: str, alias: str = None, pathName: str = None):
        if((outputType != 'debug' and outputType != 'model' and outputType != 'bash' and outputType != 'formatedLog') or outputType is None):
            if(warnings()):
                Output.printBash("Unknown command in open for Output class.", 'warn')
            return False

        if(alias == 'debug' or alias == 'model' or alias == 'bash' or alias == 'formatedLog'):
            if(warnings()):
                Output.printBash("Alias cannot have the same name as output configuration in open for Output class.", 'warn')
            return False

        if(outputType == 'bash'):
            self.bash = True
            return True

        if(alias is None):
            if(warnings()):
                Output.printBash("Alias is None but the output is not 'bash'; it is: {}.".format(outputType), 'warn')
            return False

        if(pathName is not None):
            root = self.createLogFolder()
            if(pathName in self.filesDict):
                if(outputType in self.filesDict[pathName] and self.filesDict[pathName][outputType].exist()):
                    pass # do nothing, already opened
                elif(outputType not in self.filesDict[pathName]):
                    for _, val in self.filesDict[pathName].items():
                        if(val.exist()): # copy reference
                            if(outputType == 'formatedLog'):
                                raise Exception("Output for type 'formatedLog' for provided pathName can have only one instance. For this pathName file in different mode is already opened.")
                            self.filesDict[pathName][outputType] = self.filesDict[pathName][list(self.filesDict[pathName].keys())[-1]].counterUp().addOType(outputType)
                            self.aliasToFH[alias] = self.filesDict[pathName][outputType]
                            return True
                        else:
                            del val
            self.__open(alias, root, pathName, outputType)
        else:
            if(warnings()):
                Output.printBash("For this '{}' Output type pathName should not be None.".format(outputType), 'warn')
            return True
    
    def createLogFolder(self):
        if(self.root is None):
            dt_string = datetime.now().strftime("%d.%m.%Y_%H-%M-%S_")
            prfx = self.folderSuffix if self.folderSuffix is not None else ""
            path = StaticData.LOG_FOLDER + "/" + str(dt_string) + prfx + "/"
            Path(path).mkdir(parents=True, exist_ok=False)
            self.root = path
            return path
        return self.root

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
        elif(mode == 'err'):
            prefix = 'ERROR:'
        elif(warnings()):
            Output.printBash("Unrecognized mode in Output.printBash method. Printing without prefix.", 'warn')
        return prefix

    def write(self, arg, alias: list = None, ignoreWarnings = False, end = '', mode: str = None):
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
            if(not isinstance(alias, list)):
                alias = [alias]
            for al in alias:
                if(al == 'bash'):
                    print(arg, end=end)
                if(al in self.aliasToFH.keys() and self.aliasToFH[al].exist()):
                    self.aliasToFH[al].get().write(str(arg) + end)
                    if('formatedLog' in self.aliasToFH[al].OType):
                        prBash = False
                elif(warnings() and not (ignoreWarnings or StaticData.IGNORE_IO_WARNINGS)):
                    print("WARNING: Output alias for 'write / print' not found: '{}'".format(al), end=end)
                    print(al, self.aliasToFH.keys())
                
    def print(self, arg, alias: list = None, ignoreWarnings = False, mode: str = None):
        """
        Przekazuje argument do wszystkich możliwych, aktywnych strumieni wyjściowych.\n
        Na końcu argumentu dodaje znak nowej linii.
        """
        self.write(arg, alias, ignoreWarnings, '\n', mode)

    def writeDefault(self, arg, ignoreWarnings = False, end = '', mode: str = None):
        if(self.currentDefaultAlias is None):
            self.printBash("Default output alias not set but called 'writeDefault'.", 'warn')
        self.write(arg, alias=self.currentDefaultAlias, ignoreWarnings=ignoreWarnings, end=end, mode=mode)
        
    def printDefault(self, arg, ignoreWarnings = False, mode: str = None):
        if(self.currentDefaultAlias is None):
            self.printBash("Default output alias not set but called 'printDefault'.", 'warn')
        self.print(arg, alias=self.currentDefaultAlias, ignoreWarnings=ignoreWarnings, mode=mode)

    def getFileName(self, alias):
        if(alias in self.aliasToFH):
            return os.path.basename(self.aliasToFH[alias].handler.name)
        self.printBash("Could not find alias '{}' in opened files.".format(alias), 'warn')
        return None

    def getRelativeFilePath(self, alias):
        if(alias in self.aliasToFH):
            return self.aliasToFH[alias].handler.name
        self.printBash("Could not find alias '{}' in opened files.".format(alias), 'warn')
        return None

    def __del__(self):
        for _, fh in self.aliasToFH.items():
            del fh

    def flushAll(self):
        for _, fh in self.aliasToFH.items():
            fh.flush()
        sys.stdout.flush()

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata=metadata, suffix=StaticData.OUTPUT_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

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
        'err'
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

    # niepotrzebna
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
    """
    Klasa służy do zapamiętania stanu pętli treningowych oraz testowych, niezależnie od kolejności ich wywołania.
    Kolejność wywoływania pętli treningowych oraz testowych powinna być niezmienna, 
    inaczej program pogubi się w tym, która pętla powinna zostać wznowiona.
    """
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
        """
        Dodaje do listy numer iteracji pętli oraz to, czy ona się skończyła.
        """
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
        """
        Sprawdza, czy w buforze występują jeszcze jakieś pętle.
        Jeżeli dana pętla się skończyła, to zwraca None.
        Jeżeli dana pętla wykonała się tylko w części, to zwraca wartość od której pętla ma zacząć iterować.
        Jeżeli bufor jest pusty, to zwraca 0 - wartość od której pętla ma się rozpocząć.
        """
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


class Data_Metadata(SaveClass, BaseMainClass):
    def __init__(self, worker_seed = 841874, train = True, download = True, pin_memoryTrain = False, pin_memoryTest = False,
            epoch = 1, batchTrainSize = 4, batchTestSize = 4, howOftenPrintTrain = 2000):
        super().__init__()

        # default values:
        self.worker_seed = worker_seed # ziarno dla torch.utils.data.DataLoader - worker_init_fn
        
        self.download = download
        self.pin_memoryTrain = pin_memoryTrain
        self.pin_memoryTest = pin_memoryTest

        self.epoch = epoch
        self.batchTrainSize = batchTrainSize
        self.batchTestSize = batchTestSize

        # print = batch size * howOftenPrintTrain
        self.howOftenPrintTrain = howOftenPrintTrain

    def tryPinMemoryTrain(self, metadata, modelMetadata):
        if(torch.cuda.is_available()):
            self.pin_memoryTrain = True
            if(metadata.debugInfo):
                Output.printBash('Train data pinned to GPU: {}'.format(self.pin_memoryTrain), 'info')
        return bool(self.pin_memoryTrain)

    def tryPinMemoryTest(self, metadata, modelMetadata):
        if(torch.cuda.is_available()):
            self.pin_memoryTest = True
            if(metadata.debugInfo):
                Output.printBash('Test data pinned to GPU: {}'.format(self.pin_memoryTest), 'info')
        return bool(self.pin_memoryTest)

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Download data:\t{}\n'.format(self.download))
        tmp_str += ('Pin memory train:\t{}\n'.format(self.pin_memoryTrain))
        tmp_str += ('Pin memory test:\t{}\n'.format(self.pin_memoryTest))
        tmp_str += ('Batch train size:\t{}\n'.format(self.batchTrainSize))
        tmp_str += ('Batch test size:\t{}\n'.format(self.batchTestSize))
        tmp_str += ('Number of epochs:\t{}\n'.format(self.epoch))
        tmp_str += ('How often print:\t{}\n'.format(self.howOftenPrintTrain))
        return tmp_str

    def _getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata=metadata, suffix=StaticData.DATA_METADATA_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.DATA_METADATA_SUFFIX

    def canUpdate(self = None):
        return False

class Model_Metadata(SaveClass, BaseMainClass):
    def __init__(self):
        super().__init__()

    def _getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata=metadata, suffix=StaticData.MODEL_METADATA_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.MODEL_METADATA_SUFFIX

    def canUpdate(self = None):
        return False

class Smoothing_Metadata(SaveClass, BaseMainClass):
    def __init__(self):
        super().__init__()

    def _getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata=metadata, suffix=StaticData.SMOOTHING_METADATA_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.SMOOTHING_METADATA_SUFFIX

    def canUpdate(self = None):
        return False


class TrainDataContainer():
    """
    trainHelper
    """
    def __init__(self):
        self.size = None
        self.timer = None
        self.loopTimer = None

        # one loop train data
        self.loss = None
        self.diff = None
        self.inputs = None
        self.labels = None
        self.smoothingSuccess = False # flaga mówiąca czy wygładzanie wzięło pod uwagę wagi modelu w danej iteracji

        self.batchNumber = None # current batch number
        self.loopEnded = False # check if loop ened

class TestDataContainer():
    """
    testHelper
    """
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
        self.test_loss = 0.0
        self.test_correct = 0
        self.testLossSum = 0.0
        self.testCorrectSum = 0
        self.predSizeSum = 0

class EpochDataContainer():
    """
    epochHelper
    """
    def __init__(self):
        self.epochNumber = None
        self.returnObj = None
        self.currentLoopTimeAlias = None
        self.loopsState = LoopsState()
        self.statistics = Statistics()

        self.firstSmoothingSuccess = False # flaga zostaje zapalona, gdy po raz pierwszy wygładzanie zostało włączone

class Statistics():
    """
    Klasa zwracana przez metodę epochLoop.
    """
    def __init__(self):
        super().__init__()
        self.logFolder = None # folder w którym zapisują się logi
        self.plotBatches = {} # słownik {nazwa_nowego_pliku: [lista_nazw_plików_do_przeczytania]}

    def printPlots(self, fileFormat = '.svg', dpi = 900, widthTickFreq = 0.08, aspectRatio = 0.3,
    startAt = None, resolutionInches = 11.5):
        for name, val in self.plotBatches.items():
            if(val is None):
                Output.printBash("Some of the files to plot were not properly created. Instance ignored. Method Statistics.printPlots", 'warn')
            plot(val, name=name, plotRoot=self.logFolder, fileFormat=fileFormat, dpi=dpi, widthTickFreq=widthTickFreq,
            aspectRatio=aspectRatio, startAt=startAt, resolutionInches=resolutionInches)

class Data(SaveClass, BaseMainClass, BaseLogicClass):
    """
        Metody konieczne do przeciążenia, dla których wymaga się użycia super().
        __init__
        __setInputTransform__

        Metody konieczne do przeciążenia, dla których nie używa się super().
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
    def __init__(self, dataMetadata):
        super().__init__()

        # dane wewnętrzne
        self.trainset = None
        self.trainloader = None
        self.testset = None
        self.testloader = None
        self.transform = None

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

        self.__prepare__(dataMetadata)

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Is trainset set:\t{}\n'.format(self.trainset is not None))
        tmp_str += ('Is trainloader set:\t{}\n'.format(self.trainloader is not None))
        tmp_str += ('Is testset set:\t\t{}\n'.format(self.testset is not None))
        tmp_str += ('Is testloader set:\t{}\n'.format(self.testloader is not None))
        tmp_str += ('Is transform set:\t{}\n'.format(self.transform is not None))
        return tmp_str

    def __customizeState__(self, state):
        if(self.only_Key_Ingredients):
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
            self.epochNumb = 0
            self.trainHelper = None
            self.testHelper = None
            self.epochHelper = None

        self.trainset = None
        self.trainloader = None
        self.testset = None
        self.testloader = None
        self.transform = None

    def __setInputTransform__(self, dataMetadata):
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

    def __train__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata, metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
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
        helper.smoothingSuccess = smoothing(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, smoothingMetadata=smoothingMetadata, metadata=metadata)

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

    def __beforeTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):      
        pass

    def __beforeTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        pass

    def __afterTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        pass

    def __afterTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        metadata.stream.print("Train summary:")
        metadata.stream.print(f" Average train time ({helper.timer.getUnits()}): {helper.timer.getAverage()}")
        metadata.stream.print(f" Loop train time ({helper.timer.getUnits()}): {helper.loopTimer.getDiff()}")

    def __trainLoopExit__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        helperEpoch.loopsState.imprint(numb=helper.batchNumber, isEnd=helper.loopEnded)

    def trainLoop(self, model: 'Model', helperEpoch: 'EpochDataContainer', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        """
        Główna logika pętli treningowej.
        """
        startNumb = helperEpoch.loopsState.decide()
        if(startNumb is None):
            return # loop already ended. This state can occur when framework was loaded from file.

        if(self.trainHelper is None): # jeżeli nie było wznowione; nowe wywołanie
            self.trainHelper = self.setTrainLoop(model=model, modelMetadata=modelMetadata, metadata=metadata)

        torch.cuda.empty_cache()
        self.__beforeTrainLoop__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

        self.trainHelper.loopTimer.start()
        for batch, (inputs, labels) in enumerate(self.trainloader):
            if(batch < startNumb): # already iterated
                continue

            del self.trainHelper.inputs
            del self.trainHelper.labels

            self.trainHelper.inputs = inputs
            self.trainHelper.labels = labels
            self.trainHelper.batchNumber = batch
            if(SAVE_AND_EXIT_FLAG):
                self.__trainLoopExit__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                return

            if(StaticData.TEST_MODE and batch >= StaticData.MAX_DEBUG_LOOPS):
                break
            
            self.__beforeTrain__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
            
            del self.trainHelper.loss

            self.trainHelper.timer.clearTime()
            self.trainHelper.timer.start()
            
            
            self.__train__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

            self.trainHelper.timer.end()
            if(helperEpoch.currentLoopTimeAlias is None and warnings()):
                Output.printBash("Alias for test loop file was not set. Variable helperEpoch.currentLoopTimeAlias may be set" +
                " as:\n\t'loopTestTime_normal'\n\t'loopTestTime_smooothing'\n\t'loopTrainTime'\n", 'warn')
            else:
                metadata.stream.print(self.trainHelper.timer.getDiff() , alias=helperEpoch.currentLoopTimeAlias)
            self.trainHelper.timer.addToStatistics()
            weightsSum = sumAllWeights(dict(model.getNNModelModule().named_parameters()))
            metadata.stream.print(str(weightsSum), 'weightsSumTrain')

            if(self.trainHelper.smoothingSuccess):
                if(helperEpoch.firstSmoothingSuccess == False):
                    metadata.stream.print("Successful first smoothing call while train at batch {}".format(batch), ['model:0', 'debug:0'])
                    helperEpoch.firstSmoothingSuccess = True
                else:
                    metadata.stream.print("Successful smoothing call while train at batch {}".format(batch), 'debug:0')

            self.__afterTrain__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

            if(self.trainHelper.smoothingSuccess and smoothing.__isSmoothingGoodEnough__(
                helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, 
                modelMetadata=modelMetadata, metadata=metadata, smoothingMetadata=smoothingMetadata)
            ):
                break

            self.trainHelper.smoothingSuccess = False

        self.trainHelper.loopTimer.end()
        self.trainHelper.loopEnded = True

        self.__afterTrainLoop__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        self.__trainLoopExit__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

        metadata.stream.print('Train time;', alias='stat')
        metadata.stream.print(str(self.trainHelper.loopTimer.getTimeSum()) + ';', alias='stat')

        self.trainHelper = None
        
    def __test__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
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

    def __beforeTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        pass

    def __beforeTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        pass

    def __afterTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        helper.testLossSum += helper.test_loss
        helper.test_correct = (helper.pred.argmax(1) == helper.labels).type(torch.float).sum().item()
        helper.testCorrectSum += helper.test_correct

    def __afterTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        pass

    def __testLoopExit__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        helperEpoch.loopsState.imprint(numb=helper.batchNumber, isEnd=helper.loopEnded)

    def testLoop(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        startNumb = helperEpoch.loopsState.decide()
        if(startNumb is None):
            return # loop already ended. This state can occur when framework was loaded from file.
        
        if(self.testHelper is None): # jeżeli nie było wznowione; nowe wywołanie
            self.testHelper = self.setTestLoop(model=model, modelMetadata=modelMetadata, metadata=metadata)

        torch.cuda.empty_cache()
        self.__beforeTestLoop__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

        with torch.no_grad():
            self.testHelper.loopTimer.start()
            for batch, (inputs, labels) in enumerate(self.testloader):
                if(batch < startNumb): # already iterated
                    continue
                self.testHelper.inputs = inputs
                self.testHelper.labels = labels
                self.testHelper.batchNumber = batch
                if(SAVE_AND_EXIT_FLAG):
                    self.__testLoopExit__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                    return

                if(StaticData.TEST_MODE and batch >= StaticData.MAX_DEBUG_LOOPS):
                    break

                self.__beforeTest__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

                self.testHelper.timer.clearTime()
                self.testHelper.timer.start()
                self.__test__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                self.testHelper.timer.end()
                if(helperEpoch.currentLoopTimeAlias is None and warnings()):
                    Output.printBash("Alias for test loop file was not set. Variable helperEpoch.currentLoopTimeAlias may be set" +
                    " as:\n\t'loopTestTime_normal'\n\t'loopTestTime_smooothing'\n\t'loopTrainTime'\n", 'warn')
                else:
                    metadata.stream.print(self.testHelper.timer.getDiff() , helperEpoch.currentLoopTimeAlias)
                self.testHelper.timer.addToStatistics()

                self.testHelper.predSizeSum += labels.size(0)
                self.__afterTest__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

            self.testHelper.loopTimer.end()
            self.testHelper.loopEnded = True

        self.__afterTestLoop__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        self.__testLoopExit__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        self.testHelper = None

    def __beforeEpochLoop__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        pass

    def __afterEpochLoop__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        pass

    def __epoch__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        """
        Reprezentuje pojedynczy epoch.
        Znajduje się tu cała logika epocha. Aby wykorzystać możliwość wyjścia i zapisu w danym momencie stanu modelu, należy zastosować konstrukcję:

        if(enabledSaveAndExit()):
            return 

        """
        raise Exception("Not implemented")

    def __epochLoopExit__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        pass

    def epochLoop(self, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        metadata.prepareOutput()
        self.epochHelper = EpochDataContainer()
        self.epochHelper.statistics.logFolder = metadata.stream.root

        self.__beforeEpochLoop__(helperEpoch=self.epochHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

        for ep, (loopEpoch) in enumerate(range(dataMetadata.epoch)):  # loop over the dataset multiple times
            if(ep < self.epochNumb): # already iterated
                continue
            self.epochHelper.epochNumber = ep
            metadata.stream.print(f"\nEpoch {loopEpoch+1}\n-------------------------------")
            metadata.stream.flushAll()
            
            self.__epoch__(helperEpoch=self.epochHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
            if(SAVE_AND_EXIT_FLAG):
                self.__epochLoopExit__(helperEpoch=self.epochHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                return

            if(StaticData.TEST_MODE and ep == 3):
                break

        self.__afterEpochLoop__(helperEpoch=self.epochHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        self.__epochLoopExit__(helperEpoch=self.epochHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

        a = metadata.stream.getRelativeFilePath('loopTrainTime')
        b = metadata.stream.getRelativeFilePath('loopTestTime_normal')
        c = metadata.stream.getRelativeFilePath('loopTestTime_smooothing')
        self.epochHelper.statistics.plotBatches['loopTimeTrain'] = [a]
        self.epochHelper.statistics.plotBatches['loopTimeTest'] = [b, c]

        a = metadata.stream.getRelativeFilePath('statLossTrain')
        b = metadata.stream.getRelativeFilePath('statLossTest_normal')
        c = metadata.stream.getRelativeFilePath('statLossTest_smooothing')
        self.epochHelper.statistics.plotBatches['lossTrain'] = [a]
        self.epochHelper.statistics.plotBatches['lossTest'] = [b, c]

        a = metadata.stream.getRelativeFilePath('weightsSumTrain')
        self.epochHelper.statistics.plotBatches['weightsSumTrain'] = [a]


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
        self.__setInputTransform__(dataMetadata)

    def trySave(self, metadata: 'Metadata', onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata=metadata, suffix=StaticData.DATA_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.DATA_SUFFIX

    def canUpdate(self = None):
        return True

    def lastEpoch(epochHelper, dataMetadata):
        return epochHelper.epochNumber == (dataMetadata.epoch - 1)

class Smoothing(SaveClass, BaseMainClass, BaseLogicClass):
    """
    Metody, które wymagają przeciążenia i wywołania super()
    __setDictionary__
    __getSmoothedWeights__ - zwraca pusty słownik lub None

    Metody, które wymagają przeciążenia bez wywołania super()
    __isSmoothingGoodEnough__
    """
    def __init__(self, smoothingMetadata):
        super().__init__()

        # dane wewnętrzne
        self.enabled = False # used only to prevent using smoothing when weights and dict are not set

        self.savedWeights = {}

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, smoothingMetadata, metadata):
        """
            Zwraca True jeżeli wygładzanie wzięło pod uwagę wagi modelu.
            Jeżeli wygładzone wagi nie zostały zmienione, zwraca False. 
        """
        return False

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        """
        Zwraca słownik wag, który można użyć do załadowania ich do modelu. Wagi ładuje się standardową metodą torch.
        Może zwrócić pusty słownik, jeżeli obiekt nie jest gotowy do podania wag.
        Jeżeli zwraca pusty słownik, to wygładzanie nie zostało poprawnie skonfigurowane.
        Gdy istnieje możliwość zwrócenia wag, zwraca None.
        """
        if(self.enabled == False):
            return {}
        else:
            None

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        """
            Zostaje wywołane tylko wtedy, gdy w danej iteracji pętli pomyślnie wywołano wygładzanie (__call__)
        """
        raise Exception("Not implemented.")

    def getWeights(self, key, toDevice=None, copy = False):
        if(key in self.savedWeights.keys()):
            if(copy):
                return cloneTorchDict(self.savedWeights[key], toDevice)
            else:
                return moveToDevice(self.savedWeights[key], toDevice)
        else:
            Output.printBash("Smoothing: could not find key '{}' while searching for weights.".format(key), 'warn')
            return None

    def __setDictionary__(self, smoothingMetadata, dictionary):
        """
        Used to map future weights into internal sums.
        """
        self.enabled = True

    def saveWeights(self, weights, key, canOverride = True, toDevice = None):
        with torch.no_grad():
            if(canOverride):
                self.savedWeights[key] = cloneTorchDict(weights, toDevice)
            elif(key in self.savedWeights.keys()):
                Output.printBash("The given key '{}' was found during the cloning of the scales. No override flag was specified.".format(key), 'warn')
                self.savedWeights[key] = cloneTorchDict(weights, toDevice)

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata=metadata, suffix=StaticData.SMOOTHING_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.SMOOTHING_SUFFIX

    def canUpdate(self = None):
        return False

class Model(nn.Module, SaveClass, BaseMainClass, BaseLogicClass):
    """
        Klasa służąca do tworzenia nowych modeli.
        Aby bezpiecznie skorzystać z metod nn.Module należy wywołać metodę getNNModelModule(), która zwróci obiekt typu nn.Module.
        Dana klasa nie posiada żadnych wag, dlatego nie trzeba używać rekursji, przykładowo w named_parameters(recurse=False) 

        Metody, które wymagają przeciążenia bez wywołania super()
        __update__
        __initializeWeights__ - należy go wywołać na samym końcu __init__, z powodu dodania w klasach pochodnych dodatkowych wag modelu.

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

        # dane wewnętrzne
        self.weightsInit = False # flaga mówiąca, czy wagi zostały zainicjalizowane

    def __initializeWeights__(self):
        self.weightsInit = True

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Weights initialized:\t{}\n'.format(self.weightsInit))
        return tmp_str

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
        return super().trySave(metadata=metadata, suffix=StaticData.MODEL_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.MODEL_SUFFIX

    def canUpdate(self = None):
        return True

    def getNNModelModule(self):
        """
        Używany, gdy chcemy skorzystać z funckji modułu nn.Module. Zwraca obiekt dla którego jest pewność, że implementuje klasę nn.Module. 
        """
        return self

class PredefinedModel(SaveClass, BaseMainClass, BaseLogicClass):
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
        return super().trySave(metadata=metadata, suffix=StaticData.PREDEFINED_MODEL_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

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
    dictObjs['Metadata'].trySave(onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)
    md = dictObjs['Metadata']
    
    for key, obj in dictObjs.items():
        if(key != 'Metadata'):
            obj.trySave(metadata=md, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

def commandLineArg(metadata, dataMetadata, modelMetadata, argv, enableLoad = True, enableSave = True):
    help = 'Help:\n'
    help += os.path.basename(__file__) + ' -h <help> [-s,--save] <file name to save> [-l,--load] <file name to load>'

    loadedMetadata = None

    shortOptions = 'hs:l:d'
    longOptions = [
        'save=', 'load=', 'test=', 'train=', 'pinTest=', 'pinTrain=', 'debug', 
        'debugOutput=',
        'modelOutput=',
        'bashOutput=',
        'mname=',
        'formatedOutput=',
        'log='
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
            Output.printBash(help, 'info')
            sys.exit()
        elif opt in ('-s', '--save'):
            if(enableSave):
                metadata.fileNameSave = arg
            else:
                continue
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
        elif opt in ('--mname'):
            metadata.name = arg
        elif opt in ('--log'):
            metadata.logFolderSuffix = arg
        else:
            Output.printBash("Unknown flag provided to program: {}.".format(opt), 'info')

    if(metadata.modelOutput is None):
        metadata.modelOutput = 'default_model'

    if(metadata.debugOutput is None):
        metadata.debugOutput = 'default_debug'  

    if(metadata.formatedOutput is None):
        metadata.formatedOutput = 'default_formatedOutput' 

    metadata.noPrepareOutput = False

    return metadata, False

def printClassToLog(metadata, *obj):
    where = ['debug:0', 'model:0']
    metadata.stream.print(str(metadata), where)
    for o in obj:
        metadata.stream.print(str(o), where)

def runObjs(metadataObj, dataMetadataObj, modelMetadataObj, smoothingMetadataObj, smoothingObj, dataObj, modelObj, folderLogNameSuffix = None):
    dictObjs = {}
    metadataObj.prepareOutput()
    dictObjs[type(metadataObj)] = metadataObj
    dictObjs[type(dataMetadataObj)] = dataMetadataObj
    dictObjs[type(modelMetadataObj)] = modelMetadataObj
    dictObjs[type(smoothingMetadataObj)] = smoothingMetadataObj
    dictObjs[type(smoothingObj)] = smoothingObj
    dictObjs[type(modelObj)] = modelObj
    dictObjs[type(dataObj)] = dataObj

    if(folderLogNameSuffix is not None):
        dictObjs[type(metadataObj)].logFolderSuffix = folderLogNameSuffix

    stats = dictObjs[type(dataObj)].epochLoop(
        model=dictObjs[type(modelObj)], dataMetadata=dictObjs[type(dataMetadataObj)], modelMetadata=dictObjs[type(modelMetadataObj)], 
        metadata=dictObjs[type(metadataObj)], smoothing=dictObjs[type(smoothingObj)], smoothingMetadata=dictObjs[type(smoothingMetadataObj)]
        )

    printClassToLog(dictObjs[type(metadataObj)], dictObjs[type(modelMetadataObj)], dictObjs[type(dataObj)],
        dictObjs[type(dataMetadataObj)],  dictObjs[type(modelObj)], dictObjs[type(smoothingObj)], dictObjs[type(smoothingMetadataObj)])

    return stats

def modelRun(Metadata_Class, Data_Metadata_Class, Model_Metadata_Class, Smoothing_Metadata_Class, Data_Class, Model_Class, Smoothing_Class, 
    modelObj = None, load = True, save = True, folderLogNameSuffix = None):
    dictObjs = {}
    dictObjs[Metadata_Class.__name__] = Metadata_Class()
    loadedSuccessful = False
    metadataLoaded = None

    dictObjs[Data_Metadata_Class.__name__] = Data_Metadata_Class()
    dictObjs[Model_Metadata_Class.__name__] = Model_Metadata_Class()
    dictObjs[Smoothing_Metadata_Class.__name__] = Smoothing_Metadata_Class()

    dictObjs[Metadata_Class.__name__], metadataLoaded = commandLineArg(
        metadata=dictObjs[Metadata_Class.__name__], dataMetadata=dictObjs[Data_Metadata_Class.__name__], modelMetadata=dictObjs[Model_Metadata_Class.__name__], argv=sys.argv[1:],
        enableSave=save, enableLoad=load)

    if(folderLogNameSuffix is not None):
        dictObjs[Metadata_Class.__name__].logFolderSuffix = folderLogNameSuffix

    dictObjs[Metadata_Class.__name__].prepareOutput()
    if(metadataLoaded): # if model should be loaded
        dictObjsTmp = tryLoad(tupleClasses=[(Data_Metadata_Class, Data_Class), (None, Smoothing_Class), (Model_Metadata_Class, Model_Class)], 
            metadata=dictObjs[Metadata_Class.__name__])
        if(dictObjsTmp is None):
            loadedSuccessful = False
        else:
            dictObjs = dictObjsTmp
            dictObjs[Metadata_Class.__name__] = dictObjs[Metadata_Class.__name__]
            loadedSuccessful = True
            dictObjs[Metadata_Class.__name__].printContinueLoadedModel()

    if(loadedSuccessful == False):
        dictObjs[Metadata_Class.__name__].printStartNewModel()
        dictObjs[Data_Class.__name__] = Data_Class(dataMetadata=dictObjs[Data_Metadata_Class.__name__])
        
        #dictObjs[Data_Class.__name__].__prepare__(dataMetadata=dictObjs[Data_Metadata_Class.__name__])
        
        dictObjs[Smoothing_Class.__name__] = Smoothing_Class(smoothingMetadata=dictObjs[Smoothing_Metadata_Class.__name__])
        if(issubclass(Model_Class, PredefinedModel)):
            if(modelObj is not None):
                dictObjs[Model_Class.__name__] = Model_Class(obj=modelObj, modelMetadata=dictObjs[Model_Metadata_Class.__name__])
            else:
                raise Exception("Predefined model to be created needs object.")
        else:
            dictObjs[Model_Class.__name__] = Model_Class(modelMetadata=dictObjs[Model_Metadata_Class.__name__])

        dictObjs[Smoothing_Class.__name__].__setDictionary__(smoothingMetadata=dictObjs[Smoothing_Metadata_Class.__name__], dictionary=dictObjs[Model_Class.__name__].getNNModelModule().named_parameters())

    stat = dictObjs[Data_Class.__name__].epochLoop(
        model=dictObjs[Model_Class.__name__], dataMetadata=dictObjs[Data_Metadata_Class.__name__], modelMetadata=dictObjs[Model_Metadata_Class.__name__], 
        metadata=dictObjs[Metadata_Class.__name__], smoothing=dictObjs[Smoothing_Class.__name__], smoothingMetadata=dictObjs[Smoothing_Metadata_Class.__name__]
        )
    
    printClassToLog(dictObjs[Metadata_Class.__name__], dictObjs[Model_Metadata_Class.__name__], dictObjs[Smoothing_Metadata_Class.__name__], dictObjs[Data_Metadata_Class.__name__],
        dictObjs[Data_Class.__name__], dictObjs[Model_Class.__name__], dictObjs[Smoothing_Class.__name__])
    
    dictObjs[Metadata_Class.__name__].printEndModel()

    trySave(dictObjs=dictObjs)

    return stat

#########################################
# other functions
def cloneTorchDict(weights: dict, toDevice = None):
    newDict = dict()
    if(isinstance(weights, dict)):
        for key, val in weights.items():
            newDict[key] = torch.clone(val).to(toDevice)
        return newDict
    else:
        for key, val in weights:
            newDict[key] = torch.clone(val).to(toDevice)
        return newDict

def moveToDevice(weights: dict, toDevice):
    if(isinstance(weights, dict)):
        for key, val in weights.items():
            weights[key] = val.to(toDevice)
        return weights
    else:
        for key, val in weights:
            weights[key] = val.to(toDevice)
        return weights

def sumAllWeights(weights):
    """
    Oblicza sumę wszyskich wartości bezwzględnych odstarczonych wag.
    """
    sumArray = []
    for val in weights.values():
        sumArray.append(torch.sum(torch.abs(val)))
    absSum = torch.sum(torch.stack(sumArray)).item()
    return absSum

def checkStrCUDA(string):
        return string.startswith('cuda')

def trySelectCUDA(device, metadata):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if(metadata.debugInfo):
        Output.printBash('Using {} torch CUDA device version\nCUDA avaliable: {}\nCUDA selected: {}'.format(torch.version.cuda, torch.cuda.is_available(), self.device == 'cuda'),
        'debug')
    return device

def selectCPU(device, metadata):
    device = 'cpu'
    if(metadata.debugInfo):
        Output.printBash('Using {} torch CUDA device version\nCUDA avaliable: {}\nCUDA selected: False'.format(torch.version.cuda, torch.cuda.is_available()),
        'debug')
    return device

def checkForEmptyFile(filePath):
    return os.path.isfile(filePath) and os.path.getsize(filePath) > 0

def plot(filePath: list, name = None, fetchedFilesRoot = None, plotRoot = None, fileFormat = '.svg', dpi = 900, widthTickFreq = 0.08, 
    aspectRatio = 0.3, startAt = None, resolutionInches = 11.5):
    """
    Rozmiar wyjściowej grafiki jest podana wzorem [resolutionInches; resolutionInches / aspectRatio]
    """
    if(isinstance(filePath, str)):
        filePath = [filePath]

    if(len(filePath) == 0):
        Output.printBash("Could not create plot. Input files names are empty.", 'warn')
        return

    fp = []
    xleft, xright = [], []
    ybottom, ytop = [], []

    sampleMaxSize = 0
    ax = plt.gca()
    fig = plt.gcf()
    if(fetchedFilesRoot is not None):
        for fn in filePath:
            fp.append(fetchedFilesRoot + '/' + fn)
    else:
        fp = filePath

    for fn in fp:
        if(not checkForEmptyFile(fn)):
            Output.printBash("Cannot plot file '{}'. File is empty or does not exist.".format(fn), 'warn')
            continue
        data = pd.read_csv(fn, header=None)
        if(len(data) > sampleMaxSize):
            sampleMaxSize = len(data)
        plt.plot(data, label=os.path.basename(fn))

        xleft2, xright2 = ax.get_xlim()
        xleft.append(xleft2)
        xright.append(xright2)
        ybottom2, ytop2 = ax.get_ylim()
        ybottom.append(ybottom2)
        ytop.append(ytop2)

    xleft = min(xleft)
    xright = max(xright)
    ybottom = min(ybottom)
    ytop = max(ytop)
    fig.set_size_inches(resolutionInches/aspectRatio, resolutionInches)

    if(startAt is None):
        startAt=xleft

    aspect = abs((xright-xleft)/(ybottom-ytop))*aspectRatio
    #ax.set_aspect(aspect)
    tmp = sampleMaxSize / widthTickFreq
    ax.xaxis.set_ticks(numpy.arange(startAt, xright, (sampleMaxSize*widthTickFreq)*aspectRatio))
    ax.set_xlim(xmin=startAt)
    plt.legend()
    plt.grid()


    if(name is not None):
        if(plotRoot is not None):
            plt.savefig(plotRoot + '/' + name + fileFormat, bbox_inches='tight', dpi=dpi)
        else:
            plt.savefig(name + fileFormat, bbox_inches='tight', dpi=dpi)
        plt.clf()
        return
    plt.show()


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
        loadedSuccessful = False

        dictObjs[Data_Metadata_Class.__name__] = Data_Metadata_Class()
        dictObjs[Model_Metadata_Class.__name__] = Model_Metadata_Class()

        dictObjs[Metadata_Class.__name__], _ = commandLineArg(dictObjs[Metadata_Class.__name__], dictObjs[Data_Metadata_Class.__name__], dictObjs[Model_Metadata_Class.__name__], sys.argv[1:], False)
        dictObjs[Metadata_Class.__name__].prepareOutput()

        dictObjs[Metadata_Class.__name__].printStartNewModel()
        dictObjs[Data_Class.__name__] = Data_Class()
        
        #dictObjs[Data_Class.__name__].__prepare__(dataMetadata=dictObjs[Data_Metadata_Class.__name__])
        
        dictObjs[Smoothing_Class.__name__] = Smoothing_Class()
        if(issubclass(Model_Class, PredefinedModel)):
            if(modelObj is not None):
                dictObjs[Model_Class.__name__] = Model_Class(modelObj, dictObjs[Model_Metadata_Class.__name__])
            else:
                raise Exception("Predefined model to be created needs object.")
        else:
            dictObjs[Model_Class.__name__] = Model_Class(dictObjs[Model_Metadata_Class.__name__])

        dictObjs[Smoothing_Class.__name__].__setDictionary__(dictObjs[Model_Class.__name__].getNNModelModule().named_parameters())

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
