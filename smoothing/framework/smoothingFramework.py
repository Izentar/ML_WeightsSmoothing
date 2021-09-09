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
import csv
import operator
import copy
import json

from experiments import setup

import matplotlib.pyplot as plt
import numpy

from typing import Union

# zmienne pomocnicze dla całego programu. Ich wartość na początku działania programu nie może się zmienić.
SAVE_AND_EXIT_FLAG = False
CURRENT_STAT = None # aktualna klasa statystyk, używana przy terminate.
CURRENT_STAT_PATH = None # dla testów ustawia się na folder dump
DETERMINISTIC = False

def registerStat(stat):
    """
        Rejestracja aktualnej klasy statystyk. Przydatne przy nagłym wyjściu programu.
    """
    global CURRENT_STAT
    CURRENT_STAT = stat

def clearStat():
    """
        Usuwa z tymczasowej zmiennej wskazanie na obiekt statystyk. Po tym momencie w przypadku nagłego zakończenia
        programu klasa statystyk nie zostanie zapisana.
    """
    global CURRENT_STAT
    CURRENT_STAT = None

def saveWorkAndExit(signumb, frame):
    global SAVE_AND_EXIT_FLAG
    SAVE_AND_EXIT_FLAG = True
    Output.printBash('Ending and saving model', 'info')
    return

def terminate(signumb, frame):
    global CURRENT_STAT
    global CURRENT_STAT_PATH
    Output.printBash('Catched signal: ' + str(signumb) + ". Terminating program.", 'info')
    if(CURRENT_STAT is not None):
        Output.printBash('Saving statistics', 'info')
        CURRENT_STAT.saveSelf(name="stat_exit", path=CURRENT_STAT_PATH)
    exit(2)

def enabledDeterminism():
    """
        Sprawdza, czy program został uruchomiony w trybie deterministycznym.
        Ten tryb uruchamia się za pomocą funkcji useDeterministic.
    """
    global DETERMINISTIC
    return bool(DETERMINISTIC)

def enabledSaveAndExit():
    """
        Sprawdza, czy program powinien zakończyć działanie oraz zapisać potrzebne obiekty.
        Ten tryb uruchamia się za pomocą sugnału SIGQUIT (Ctrl + \).
    """
    return bool(SAVE_AND_EXIT_FLAG)

if os.name != 'nt':
    signal.signal(signal.SIGQUIT, saveWorkAndExit) # Ctrl + \

signal.signal(signal.SIGINT, terminate)

# reproducibility https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
# set in environment CUBLAS_WORKSPACE_CONFIG=':4096:2' or CUBLAS_WORKSPACE_CONFIG=':16:8'
def useDeterministic(torchSeed = 0, randomSeed = 0):
    global DETERMINISTIC
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
    """
        Sprawdza, czy włączone jest wypisywanie ostrzeżeń do logów.
    """
    return StaticData.FORCE_PRINT_WARNINGS or StaticData.PRINT_WARNINGS

class StaticData:
    """
        Klasa przechowująca dane konfiguracyjne. 
        Pozwala się w trakcie działania programu na zmianę ich wartości, jednak
        nieumiejętna zmiana tychże wartości może spowodować późniejsze problemy.
    """
    PATH = os.path.join(expanduser("~"), 'dataSmoothing', 'models')
    TMP_PATH = os.path.join(expanduser("~"), 'dataSmoothing', 'models', 'tmp')
    MODEL_SUFFIX = '.model'
    METADATA_SUFFIX = '.metadata'
    DATA_SUFFIX = '.data'
    TIMER_SUFFIX = '.timer'
    SMOOTHING_SUFFIX = '.smoothing'
    OUTPUT_SUFFIX = '.output'
    DATA_METADATA_SUFFIX = '.dmd'
    MODEL_METADATA_SUFFIX = '.mmd'
    PYTORCH_AVERAGED_MODEL_SUFFIX = '.pyavgmmd'
    SMOOTHING_METADATA_SUFFIX = '.smthmd'
    NAME_CLASS_METADATA = 'Metadata'
    DATA_PATH = os.path.join(expanduser("~"), 'dataSmoothing')
    PREDEFINED_MODEL_SUFFIX = '.pdmodel'
    LOG_FOLDER = os.path.join(setup.PrimaryWorkingDir if setup.PrimaryWorkingDir is not None else '.', 'savedLogs') 
    IGNORE_IO_WARNINGS = False
    FORCE_DEBUG_PRINT = False
    TEST_MODE = False
    PRINT_WARNINGS = True
    FORCE_PRINT_WARNINGS = False
    MAX_DEBUG_LOOPS = 71
    MAX_EPOCH_DEBUG_LOOPS = 4

class SaveClass:
    """
        Klasa służąca do zapisywania oraz ładowania zapisanych obiektów.
        Aby z niej skorzystać, najlepiej po niej odziedziczyć, zmieniając 
    """
    def __init__(self):
        self.only_Key_Ingredients = None
        """
            Klasa dziedzicząca po niej powinna zaimplementować własną funkcję 
                def trySave(self, onlyKeyIngredients = False, temporaryLocation = False):
            Ponadto wymaga się zaimplementowania metod:
                * def getFileSuffix(self = None) -> str:
                * def canUpdate() -> bool
        """

    def tryLoad(metadata, Class, classMetadataObj = None, temporaryLocation = False):
        """
            Metoda służąca do wczytania obiektu określonej klasy. Wymagane jest podanie typu obiektu
            w zmiennej Class.
        """
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
        """
            Metoda służąca do zapisu klasy, która dziedziczy po tej klasie.
            Zaleca się zdefiniowanie jednej z dwóch funkcji:
                def trySave(self, onlyKeyIngredients = False, temporaryLocation = False),
                def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False)
            która wywołuje tę funkcję z odpowiednimi parametrami, jak 
                * suffix
        """
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
    Zwraca sekwencję kolejnych indeksów.
    Wspiera zapisywanie oraz wczytywanie.

    Przykład:
        trainSampler = sf.BaseSampler(len(data.trainset), dataMetadata.batchTrainSize)
        testSampler = sf.BaseSampler(len(data.testset), dataMetadata.batchTestSize)
    """ 
    def __init__(self, dataSize, batchSize, startIndex = 0, seed = 984):
        self.sequence = list(range(dataSize))[startIndex * batchSize:]
        random.Random(seed).shuffle(self.sequence)

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
    """
        Klasa służąca do wpisywania wartości zmiennych zawartych w klasach pochodnych.
        Wymaga przeciążenia metody 
            def __strAppend__(self) -> str:
        Klasę tą wykorzystuje się w klasach metadanych, które reprezentują dane klas pochodnych BaseMainClass.
    """
    def __strAppend__(self) -> str:
        return ""

    def __str__(self):
        tmp_str = ('\nStart {} class\n-----------------------------------------------------------------------\n'.format(type(self).__name__))
        tmp_str += self.__strAppend__()
        tmp_str += ('-----------------------------------------------------------------------\nEnd {} class\n'.format(type(self).__name__))
        return tmp_str

class BaseLogicClass:
    """
        Klasa, którą dziedziczą inne klasy implementujące jakąć logikę działania. 
        Jest to klasa wymagająca swojego odpowiednika metadanych, który dziedziczy po BaseMainClass.
    """
    def createDefaultMetadataObj(self):
        raise Exception("Not implemented")

class test_mode():
    """
        Klasa włączająca dla danego bloku tryb testowy.
        Użycie:
            with test_mode():
                ...
    """
    def __init__(self):
        self.prev = False

    def __enter__(self):
        self.prev = StaticData.TEST_MODE
        StaticData.TEST_MODE = True

    def __exit__(self, exc_type, exc_val, exc_traceback):
        StaticData.TEST_MODE = self.prev

    def isActive(self = None):
        return bool(StaticData.TEST_MODE)

class RunningGeneralMeanWeights():
    """
        Liczenie rekursywnej średniej arytmetycznej.
        
        initDummyWeights - słownik z wagami, który zostanie skopiowany do tej klasy. 
            Nowe wagi zostaną zainicjalizowane jako zera.
        device - urządzenie na którym mają być trzymane flagi. Ustawiając zmienną na None, wagi będą znajdowały się na 
            tym samym urządzeniu, co initWeights.
        setToZeros - jeżeli flaga ustawiona na True, to skopiowane wagi zostaną wyzerowane.
        dtype - typ danych jaki ma posiadać każda z wag. Domyślnie jest to torch.float32, jednak ustawiając zmienną na None,
            typ wag nie zostanie zmieniony względem initWeights.
        power - potęga dla której będzie obliczana średnia. Domyślnie ma wartość 1, co jest równoważne ze średnią arytmetyczną.
    """
    def __init__(self, initWeights: dict, device: str=None, setToZeros: bool=False, dtype: str=torch.float32, power: float=1.0):
        self.weightsDictAvg = {}
        self.N = None
        self.power = power
        self.device = device

        if(not isinstance(initWeights, dict)):
            initWeights = dict(initWeights)

        if(setToZeros):
            with torch.no_grad():
                for key, values in initWeights.items():
                    self.weightsDictAvg[key] = torch.zeros_like(values, requires_grad=False, device=device, dtype=dtype)
            self.N = 0
        else:
            with torch.no_grad():
                for key, values in initWeights.items():
                    self.weightsDictAvg[key] = torch.clone(values).to(device, dtype=dtype).requires_grad_(False)
            self.N = 1

        if(isinstance(power, int) and power == 1) or (isinstance(power, float) and power == 1.0):
            self.pow = self.__methodPow_1
            self.div = self.__methodDivGet_1
        elif(power > 1.0 or power < 1.0):
            self.pow = self.__methodPow_
            self.div = self.__methodDivGet_
        else:
            raise Exception("Power cannot be negative: {}".format(power))

    def __methodPow_(self, key, arg):
        self.weightsDictAvg[key].mul_(self.N).add_(arg.pow(self.power)).div_(self.N + 1)

    def _arithmRecurse(self, averaged_model_parameter, model_parameter, num_averaged):
        # ta sama implementacja co domyślnie w pytorch w AveragedModel
        return averaged_model_parameter + \
            (model_parameter - averaged_model_parameter) / (num_averaged + 1)

    def __methodPow_1(self, key, arg):
        self.weightsDictAvg[key].detach().copy_(self._arithmRecurse(averaged_model_parameter=self.weightsDictAvg[key].detach(),
            model_parameter=arg, num_averaged=self.N))

    def __methodDivGet_(self):
        tmpDict = {}
        for key, val in self.weightsDictAvg.items():
            tmpDict[key] = val.pow(1/self.power)
        return tmpDict

    def __methodDivGet_1(self):
        tmpDict = {}
        for key, val in self.weightsDictAvg.items():
            tmpDict[key] = torch.clone(self.weightsDictAvg[key].detach())
        return tmpDict

    def addWeights(self, weights: dict):
        """
            Dodaje do bufora wagi, z których następnie można wyciągnąć średnią.
        """
        with torch.no_grad():
            if(not isinstance(weights, dict)):
                weights = dict(weights)
            for key, values in weights.items():
                if(not key in self.weightsDictAvg):
                    raise Exception("Unknown weight name")
                #self.weightsDictAvg[key].mul_(self.N).add_(values).div_(self.N + 1)
                self.pow(key, values.to(self.device))

        self.N = self.N + 1

    def getWeights(self, device: str=None):
        """
            Zwraca uśrednione wagi. Można je modyfikować, gdyż są to kopie wag z bufora.
        """
        return self.div()

class CircularList():
    """
        Cykliczny bufor / lista. Może przechowywać ona wartości lub obiekty.
        W przypadku obiektów, niektóre metody mogą nie zadziałać. 
    """
    class CircularListIter():
        """
            Iterator dla cyklicznej listy. Dzięki takiej implementacji można iterować po 
            cyklicznej liście jednoczeście.
            Iteracja następuje od najstarszej wartości do najnowszej.
        """
        def __init__(self, circularList):
            self.circularList = circularList
            self.__iter__()

        def __iter__(self):
            lo = list(range(self.circularList.arrayIndex))
            hi = list(range(self.circularList.arrayIndex, len(self.circularList.array)))
            lo.reverse()
            hi.reverse()
            self.indexArray = lo + hi
            return self

        def __next__(self):
            if(self.indexArray):
                idx = self.indexArray.pop(0)
                return self.circularList.array[idx]
            else:
                raise StopIteration


    def __init__(self, maxCapacity):
        self.array = []
        self.arrayIndex = 0
        self.arrayMax = maxCapacity

    def pushBack(self, value):
        """
            Dodaje na koniec listy cyklicznej wartość lub obiekt. 
            Wstawiana wartość posiada największy indeks.
        """
        if(self.arrayIndex < len(self.array)):
            del self.array[self.arrayIndex] # trzeba usunąć, inaczej insert zachowa w liście obiekt
        self.array.insert(self.arrayIndex, value)
        self.arrayIndex = (self.arrayIndex + 1) % self.arrayMax

    def getAverage(self, startAt=0):
        """
            Zwraca średnią.
            Argument startAt mówi o tym, od którego momentu w kolejce należy liczyć średnią.
            Najstarsza wartość ma indeks 0.

            Można jej użyć tylko do typów, które wspierają dodawanie, które
            powinny implementować metody __copy__(self) oraz __deepcopy__(self)
        """
        l = len(self.array)
        if(startAt == 0):
            return sum(self.array) / l if l else 0
        if(l <= startAt):
            return 0
        l -= startAt
        tmpSum = None

        for i, (obj) in enumerate(iter(self)):
            if(i < startAt):
                continue
            tmpSum = copy.deepcopy(obj) # because of unknown type
            break

        for i, (obj) in enumerate(iter(self)):
            if(i < startAt + 1):
                continue
            tmpSum += obj

        return tmpSum / l

    def getStdMean(self):
        """
            Zwraca tuple(std, mean) z całego cyklicznego bufora.
        """
        return self.getStd(), self.getMean()

    def getMean(self):
        return numpy.mean(self.array)

    def getStd(self):
        return numpy.std(self.array)

    def __setstate__(self):
        self.__dict__.update(state)
        self.arrayIndex = self.arrayIndex % self.arrayMax

    def reset(self):
        """
            Usuwa cały cykliczny bufor, zastępując go nowym.
        """
        del self.array
        self.array = []
        self.arrayIndex = 0

    def __iter__(self):
        return CircularList.CircularListIter(self)

    def __len__(self):
        return len(self.array)

    def get(self, idx):
        """
            Najstarsza wartość ma indeks 0.
        """
        return self.array[(self.arrayIndex + idx) % self.arrayMax]

    def getMin(self):
        """
            Zwraca minimalną wartość / obiekt z całego bufora cyklicznego.
        """
        return min(self.array)
    
    def getMax(self):
        """
            Zwraca maksymalną wartość / obiekt z całego bufora cyklicznego.
        """
        return max(self.array)

class MultiplicativeLR():
    """ 
        Jest to scheduler, który zmienia learning rate w sposób liniowy.
    """
    def __init__(self, optimizer, gamma, startAt=0):
        """ 
            startAt - wykonuje step() podaną ilość razy. Wartość musi być większa od 0.
            gamma - wartość o jaką learning rate ma zostać pomnożony.
            optimizer - optymalizator, dla którego jest zmieniana learning rate.
        """
        self.optimizer = optimizer
        self.gamma = gamma
        self.stepCount = 0 # liczba wykonanych wywołań step()

        if startAt < 0:
            raise Exception("Bad argument startAt {}.".format(startAt))
        for i in range(startAt):
            self.step()

    def get_last_lr(self):
        """
            Zwraca learning rate optymalizatora.
        """
        ret = []
        for gr in self.optimizer.param_groups:
            ret.append(gr['lr'])
        return ret

    def step(self):
        """
            Zmienia learning rate o wartość 
                lr = lr * gamma
        """
        for gr in self.optimizer.param_groups:
            gr['lr'] *= self.gamma
        self.stepCount += 1


class Metadata(SaveClass, BaseMainClass):
    """
        Klasa ta może zmienić swój stan w kolejnych wywołaniach metod.
        Aby ją ponownie użyć należy wywołać resetOutput(), który zresetuje wszystkie dane dotyczące wyjścia modelu.
        Służy głównie do trzymania informacji o ścieżkach logowania, plikach, strumieniach.
    """
    def __init__(self, fileNameSave=None, fileNameLoad=None, testFlag=False, trainFlag=False, debugInfo=False, modelOutput=None,
            debugOutput=None, stream=None, bashFlag=False, name=None, formatedOutput=None, logFolderSuffix=None, relativeRoot=None):
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
        self.relativeRoot = relativeRoot

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
        tmp_str += ('Folder relative root name:\t{}\n'.format(self.relativeRoot))
        tmp_str += ('Output is prepared flag:\t{}\n'.format(self.noPrepareOutput))
        return tmp_str

    def onOff(arg):
        """
            Prosta funkcja, która tłumaczy zapisy on, True, off, False, true, false na wartości logiczne.
        """
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

    def resetOutput(self):
        self.stream = None
        self.logFolderSuffix = None
        self.relativeRoot = None
        self.noPrepareOutput = False

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
            self.stream = Output(self.logFolderSuffix, self.relativeRoot)

        if(self.debugInfo == True):
            self.stream.open(metadata=self, outputType='debug', alias='debug:0', pathName='debug')
        self.stream.open(metadata=self, outputType='model', alias='model:0', pathName='model')
        self.stream.open(metadata=self, outputType='bash')
        self.stream.open(metadata=self, outputType='formatedLog', alias='stat', pathName='statistics')

        self.stream.open(metadata=self, outputType='formatedLog', alias='loopTrainTime', pathName='loopTrainTime')
        self.stream.open(metadata=self, outputType='formatedLog', alias='loopTestTime_normal', pathName='loopTestTime_normal')
        self.stream.open(metadata=self, outputType='formatedLog', alias='loopTestTime_smooothing', pathName='loopTestTime_smooothing')

        self.stream.open(metadata=self, outputType='formatedLog', alias='statLossTrain', pathName='statLossTrain')
        self.stream.open(metadata=self, outputType='formatedLog', alias='statLossTest_normal', pathName='statLossTest_normal')
        self.stream.open(metadata=self, outputType='formatedLog', alias='statLossTest_smooothing', pathName='statLossTest_smooothing')

        self.stream.open(metadata=self, outputType='formatedLog', alias='weightsSumTrain', pathName='weightsSumTrain')

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
        """
            Sprawdza, czy faza treningowa powinna zostać wykonana.
        """
        return bool(self.trainFlag)

    def shouldTest(self):
        """
            Sprawdza, czy faza testowa powinna zostać wykonana.
        """
        return bool(self.testFlag)

class Timer(SaveClass):
    """
        Klasa służąca do odliczania czasu. Umożliwia synchroniację rdzeni CUDA.
    """
    def __init__(self):
        super().__init__()
        self.timeStart = None
        self.timeEnd = None
        self.modelTimeSum = 0.0
        self.modelTimeCount = 0

    def start(self, cudaDeviceSynch: str = None):
        """
            cudaSynchronize - urządzenie CUDA, które ma zostać zsynchronizowane.
        """
        if(cudaDeviceSynch is not None and cudaDeviceSynch != 'cpu'):
            torch.cuda.synchronize(device=cudaDeviceSynch)
        self.timeStart = time.perf_counter()

    def end(self, cudaDeviceSynch: str = None):
        """
            cudaSynchronize - urządzenie CUDA, które ma zostać zsynchronizowane.
        """
        if(cudaDeviceSynch is not None and cudaDeviceSynch != 'cpu'):
            torch.cuda.synchronize(device=cudaDeviceSynch)
        self.timeEnd = time.perf_counter()

    def getDiff(self):
        """
            Zwraca różnicę czasów końca oraz startu.
        """
        if(self.timeStart is not None and self.timeEnd is not None):
            return self.timeEnd - self.timeStart
        Output.printBash("Could not get time difference.", 'warn')
        return None

    def addToStatistics(self):
        """
            Metoda dodająca do wewnętrznych statystyk wartość z self.getDiff().
            Powinno się ją wywoływać po self.end(...).
        """
        tmp = self.getDiff()
        if(tmp is not None):
            self.modelTimeSum += self.getDiff()
            self.modelTimeCount += 1
        else:
            Output.printBash("Timer could not be added to statistics.", 'warn')

    def getTimeSum(self):
        """
            Metoda zwraca sumę czasów zapisanych w wewnętrznych statystykach.
        """
        return self.modelTimeSum

    def clearTime(self):
        """
            Metoda czyści dane zapisane przez start() oraz end().
            Nie czyści ona statystyk.
        """ 
        self.timeStart = None
        self.timeEnd = None

    def clearStatistics(self):
        """
            Metoda czyści tylko statystyki.
            Nie czyści danych zapisanych przez start() oraz end().
        """
        self.modelTimeSum = 0.0
        self.modelTimeCount = 0
        
    def getAverage(self):
        """
            Metoda zwraca średnią arytmetyczną z zapisanych wewnętrznych statystyk.
        """
        if(self.modelTimeCount != 0):
            return self.modelTimeSum / self.modelTimeCount
        return None

    def getCount(self):
        """
            Metoda zwraca liczbę różnic czasów zapisanych w wewnętrznych statystykach.
        """
        return self.modelTimeCount

    def getUnits(self):
        """
            Metoda zwraca jednostkę czasu w jakiej podaje się różnice czasów. 
        """
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
        Instancja tego obiektu odpowiada instancji jednego folderu, w którym będą się znajdowały wszystkie otwarte pliki,
        przy wykorzystaniu sposobu zapisu tego obiektu.
        Każdy folder zawiera w nazwie datę jego utworzenia oraz ustalony przez użytkownika sufiks.
    """
    class FileHandler():
        def __init__(self, root, pathName, mode, OType):
            if not os.path.exists(os.path.dirname(root + pathName)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            self.handler = open(os.path.join(root, pathName), mode)
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

    def __init__(self, folderSuffix, relativeRoot = None):
        super().__init__()
        self.filesDict = {}
        self.aliasToFH = {}

        self.bash = False

        Path(StaticData.LOG_FOLDER).mkdir(parents=True, exist_ok=True)

        self.folderSuffix = folderSuffix
        self.relativeRoot = relativeRoot
        self.root = None
        self.currentDefaultAlias = None
        self.debugDisabled = False

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

    def __open(self, alias, root, pathName, outputType, metadata):
        if(alias in self.aliasToFH and self.aliasToFH[alias].exist()):
            if(warnings()):
                Output.printBash("Provided alias '{}' with opened file already exist: {}. This may be due to loaded Metadata object.".format(alias, outputType),
                'warn')
            return
        if(alias == 'debug' and not (metadata.debugInfo or StaticData.FORCE_DEBUG_PRINT)):
            Output.printBash("Debug mode is not active. Debug output disabled.",
                'info')
            self.debugDisabled = True
            return

        suffix = '.log'
        if(outputType == 'formatedLog'):
            suffix = '.csv'
        fh = self.FileHandler(root, pathName + suffix, 'a', outputType)
        self.filesDict[pathName] = {outputType: fh}
        self.aliasToFH[alias] = fh

    def open(self, metadata, outputType: str, alias: str = None, pathName: str = None):
        """
            outputType - typ pliku, w jakim zostaną zapisane dane. Typ pliku determinuje jego sufiks. 
            Można mieć wiele plików o tym samym typie. Wyjątkiem jest typ 'bash', który zapisuje dane na standardowe wyjście.
            Jeden z typów:
                * debug
                * model
                * bash
                * formatedLog
            alias - alias dla danego pliku.
            pathName - nazwa pliku, który ma zostać stworzony. W tej zmiennej można zawrzeć również śceiżkę względną do folderu relativeRoot.
            Jeżeli foldery na ścieżce nie istnieją, zostaną one stworzone.
        """
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
            root = self.setLogFolder()
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
            self.__open(alias=alias, root=root, pathName=pathName, outputType=outputType, metadata=metadata)
        else:
            if(warnings()):
                Output.printBash("For this '{}' Output type pathName should not be None.".format(outputType), 'warn')
            return True
    
    def setLogFolder(self):
        """
            Tworzy folder logowania na podstawie folderSuffix oraz relativeRoot podanych przy inicjalizacji obiektu.
            Jednocześnie ustawia zmienną wewnętrzą root na ten folder.
        """
        if(self.root is None):
            self.root = Output.createLogFolder(folderSuffix=self.folderSuffix, relativeRoot=self.relativeRoot)[0]
        return self.root

    def createLogFolder(folderSuffix, relativeRoot = None) -> str:
        """
            Zwraca nazwę nowo stworzonego folderu. Wszsytkie foldery są tworzone
            względem folderu w zmiennej StaticData.LOG_FOLDER.
        """
        dt_string = datetime.now().strftime("%d.%m.%Y_%H-%M-%S_")
        prfx = folderSuffix if folderSuffix is not None else ""
        path = None
        pathRel = None
        if(relativeRoot is not None):
            path = os.path.join(StaticData.LOG_FOLDER, relativeRoot, str(dt_string) + prfx)
            pathRel = os.path.join(relativeRoot, str(dt_string) + prfx)
        else:
            path = os.path.join(StaticData.LOG_FOLDER, str(dt_string) + prfx)
            pathRel = os.path.join(str(dt_string) + prfx)
        Path(path).mkdir(parents=True, exist_ok=False)
        return path, pathRel

    def getTimeStr():
        return datetime.now().strftime("%d.%m.%Y_%H-%M-%S_")

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

    def write(self, arg, alias: list = None, ignoreWarnings = False, end = '', mode: str = None) -> None:
        """
            Przekazuje argument do wszystkich możliwych, aktywnych strumieni wyjściowych.\n
            Na końcu argumentu nie daje znaku nowej linii.
            mode - sufiks przed wiadomością:
                * info - INFO
                * debug - DEBUG
                * warn - WARNING
                * err - ERROR
            end - znak na końcu wiadomości.
            ignoreWarnings - nie wypisuje ostrzeżenia o braku danego aliasu.

        """
        if(alias == 'debug' and self.debugDisabled):
            return
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
                    self.printBash("Output alias for 'write / print' not found: '{}'".format(al), 'warn')
                    self.printBash(str(al) + " " + str(self.aliasToFH.keys()), 'warn')
                
    def print(self, arg, alias: list = None, ignoreWarnings = False, mode: str = None) -> None:
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
        """
            Wykonuje flush dla wszystkich otwartych plików.
        """
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
    """
        Klasa służąca do przechowywania tylko domyślnych metod o małym znaczeniu.
    """
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

class LoopsState():
    """
        Klasa służy do zapamiętania stanu pętli treningowych oraz testowych, niezależnie od kolejności ich wywołania.
        Kolejność wywoływania pętli treningowych oraz testowych powinna być niezmienna między wczytywaniami, 
        inaczej program nie zagwarantuje tego, która pętla powinna zostać wznowiona.
        Pomysł bazuje na zapisywaniu oraz wczytywaniu klasy, bez których metody błędnie zadziałają.
        Nie można użyć tej klasy do sprawdzania wykonania danej pętli w jednym wywołaniu programu. Wymaga to zapisu oraz wczytania klasy.
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
        Przed wywołaniem tej metody powinna zostać wywołana metoda decide(), aby się dowiedzieć, 
        czy aktualna pętla nie potrzebuje wznowienia
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
    """
        Klasa konfiguracyjna przechowująca dane odnośnie pewnego zbioru danych. Klasa zbioru danych 
        korzysta z instancji tej klasy.
    """
    def __init__(self, worker_seed = 841874, download = True, pin_memoryTrain = False, pin_memoryTest = False,
            epoch = 1, batchTrainSize = 64, batchTestSize = 64):
        """
            pin_memoryTrain - zarezerwowanie miejsca w pamięci karty graficznej na dane ze zbioru uczęcego. Przyśpiesza transfer danych.
            pin_memoryTest - zarezerwowanie miejsca w pamięci karty graficznej na dane ze zbioru treningowego. Przyśpiesza transfer danych.

            batchTrainSize - liczba próbek danych w jednym mini-batchu treningowym
            batchTrainSize - liczba próbek danych w jednym mini-batchu walidacyjnym

            epoch - liczba epok dla treningu
        """
        super().__init__()

        # default values:
        self.worker_seed = worker_seed # ziarno dla torch.utils.data.DataLoader - worker_init_fn
        
        self.download = download
        self.pin_memoryTrain = pin_memoryTrain
        self.pin_memoryTest = pin_memoryTest

        self.epoch = epoch
        self.batchTrainSize = batchTrainSize
        self.batchTestSize = batchTestSize

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
    """
        Klasa konfiguracyjna przechowująca dane odnośnie pewnego modelu.
    """
    def __init__(self, device = 'cuda:0'):
        super().__init__()
        self.device = device
        self.useCuda = False if device == 'cpu' else True

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

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Model device :\t{}\n'.format(self.device))
        return tmp_str

class Smoothing_Metadata(SaveClass, BaseMainClass):
    """
        Klasa konfiguracyjna przechowująca dane odnośnie pewnego obiektu wygładzania.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.useCuda = False if device == 'cpu' else True

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

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Device:\t{}\n'.format(self.device))
        return tmp_str

class TrainDataContainer():
    """
        Klasa pomocnicza, która zachowuje dane pomiędzy wywołaniami odnośnie treningu modelu.
        Pojawia się ona w zmiennej trainHelper.
    """
    def __init__(self):
        self.size = None # rozmiar dataset
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
        Klasa pomocnicza, która zachowuje dane pomiędzy wywołaniami odnośnie testu modelu.
        Pojawia się ona w zmiennej testHelper.
    """
    def __init__(self):
        self.size = None # rozmiar dataset
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
        Klasa pomocnicza, która zachowuje dane pomiędzy wywołaniami odnośnie epok.
        Pojawia się ona w zmiennej epochHelper.
        Ponadto posiada ona obiekt 'statistics', do której można zapisywać dane zwracane na koniec
        przejścia wszystkich epok.
    """
    def __init__(self):
        self.epochNumber = 0
        self.trainTotalNumber = None
        self.testTotalNumber = None
        self.maxTrainTotalNumber = None
        self.maxTestTotalNumber = None

        self.returnObj = None
        self.currentLoopTimeAlias = None
        self.loopsState = LoopsState()
        self.statistics = Statistics()
        registerStat(self.statistics)

        self.firstSmoothingSuccess = False # flaga zostaje zapalona, gdy po raz pierwszy wygładzanie zostało włączone
        self.averaged = False # flaga powinna zostać zapalona, gdy model posiada wygładzone wagi i wyłączona w przeciwnym wypadku
        
        self.modes = [] # różne rodzaje trybów, w tym 'normal' oraz 'smoothing'
        self.endEpoches = False # flaga zostanie zapalona, gdy faza treningowa zostanie całkowicie zakończona i modelowi pozostało
                                # wykonać tylko fazę walidacyjną 

    def addSmoothingMode(self):
        self.modes.append("smoothing")

    def addNormalMode(self):
        self.modes.append("normal")

class Statistics():
    """
        Statystyki wywołania. Posiada ona szereg pól, w których zawarte są informacje o logach oraz poszczególnych 
        zmiennych po przejściu treningu oraz testów walidacyjnych.
        Posiada metody potrzebne do jej zapisu jak i odczytu.
    """
    def __init__(self, logFolder = None, plotBatches = None, avgPlotBatches = None, rootInputFolder = None,
            trainLoopTimerSum = None, testLoopTimerSum = None, 
            lossRatio = None, correctRatio = None, testLossSum = None, testCorrectSum = None, predSizeSum = None,
            trainTimeLoop = None, avgTrainTimeLoop = None, trainTotalNumb = None, trainTimeUnits = None,
            testTimeLoop = None, avgTestTimeLoop = None, testTimeUnits = None,
            smthTestTimeLoop = None, smthAvgTestTimeLoop = None, smthTestTimeUnits = None,
            smthLossRatio = None, smthCorrectRatio = None, smthTestLossSum = None, smthTestCorrectSum = None, smthPredSizeSum = None):
        """
            logFolder - folder wyjściowy dla zapisywanych logów. Może być None.
            plotBatches - słownik {(nazwa_nowego_pliku, nazwa_osi_X, nazwa_osi_Y): [lista_nazw_plików_do_przeczytania_dla_jednego_wykresu]}. Domyślnie {} dla None.
                Pliki tutaj zawarte już istnieją.
            avgPlotBatches - słownik uśrednionych plików csv {(nazwa_nowego_pliku, nazwa_osi_X, nazwa_osi_Y): [lista_nazw_plików_do_przeczytania]}. 
                Domyślnie {} dla None. Pliki tutaj zawarte już istnieją.
            rootInputFolder - folder wejściowy dla plików. Może być None.

            trainLoopTimerSum - Domyślnie [] dla None.
            testLoopTimerSum - Domyślnie [] dla None.

            lossRatio - zapisywane po wykonanym teście, średnia strata modelu. Dla obliczenia jej przy tworzeniu
                należy przypisać mu wyraz "count". Domyślnie [] dla None.
            correctRatio - zapisywane po wykonanym teście, stosunek udanych do wszystkich predykcji. Dla obliczenia jej przy tworzeniu
                należy przypisać mu wyraz "count". Domyślnie [] dla None.
            testLossSum - zapisywane po wykonanym teście, suma strat testowych. Domyślnie [] dla None.
            self.testCorrectSum - zapisywane po wykonanym teście, suma poprawnych predykcji testowych. Domyślnie [] dla None.
            predSizeSum - zapisywane po wykonanym teście, ilość wszystkich predykcji. Domyślnie [] dla None.

            trainTimeLoop - czas wykonania całej pętli treningowej. Domyślnie [] dla None.
            avgTrainTimeLoop - średni czas wykonania jednej pętli treningowej. Domyślnie [] dla None.
            trainTotalNumb - całkowita liczba przebytych pętli treningowych. Domyślnie [] dla None.
            self.trainTimeUnits - jednostki w jakich liczony był czas. Domyślnie [] dla None.

            testTimeLoop - czas wykonania całej pętli testowej. Domyślnie [] dla None.
            avgTestTimeLoop - średni czas wykonania jednej pętli testowej. Domyślnie [] dla None.
            testTimeUnits - jednostki w jakich liczony był czas. Domyślnie [] dla None.

            smthTestTimeLoop - czas wykonania całej pętli testowej z wygładzonymi wagami. Domyślnie [] dla None.
            smthAvgTestTimeLoop - średni czas wykonania jednej pętli testowej z wygładzonymi wagami. Domyślnie [] dla None.
            smthTestTimeUnits - jednostki w jakich liczony był czas. Domyślnie [] dla None.

            smthLossRatio - zapisywane po wykonanym teście, gdy model posiada wygładzone wagi, średnia strata modelu. Dla obliczenia jej przy tworzeniu
                należy przypisać mu wyraz "count". Domyślnie [] dla None.
            smthCorrectRatio - zapisywane po wykonanym teście, gdy model posiada wygładzone wagi, stosunek udanych do wszystkich 
                predykcji. Dla obliczenia jej przy tworzeniu należy przypisać mu wyraz "count". Domyślnie [] dla None.
            smthTestLossSum - zapisywane po wykonanym teści, gdy model posiada wygładzone wagie, suma strat testowych. Domyślnie [] dla None.
            smthTestCorrectSum - zapisywane po wykonanym teście, gdy model posiada wygładzone wagi, suma poprawnych predykcji testowych. Domyślnie [] dla None.
            smthPredSizeSum - zapisywane po wykonanym teście, gdy model posiada wygładzone wagi, ilość wszystkich predykcji. Domyślnie [] dla None.
        """
        ##################################
        self.logFolder = logFolder
        if(isinstance(plotBatches, dict) or plotBatches is None):
            self.plotBatches = plotBatches if plotBatches is not None else {}
        else:
            raise Exception("Plot batches must be dictionary")
        if(isinstance(avgPlotBatches, dict) or avgPlotBatches is None):
            self.avgPlotBatches = avgPlotBatches if avgPlotBatches is not None else {}
        else:
            raise Exception("Average plot batches must be dictionary")
        self.rootInputFolder = rootInputFolder

        ###################################
        def fromTensorToItem(fromObj):
            if(isinstance(fromObj, torch.Tensor)):
                return fromObj.item()
            elif(isinstance(fromObj, list)):
                return [x.item() for x in fromObj]
            elif(isinstance(fromObj, tuple)):
                return tuple(x.item() for x in fromObj)
            else:
                return fromObj

        def setAndCheckList(fromObj):
            fromObj = fromTensorToItem(fromObj)
            return (fromObj if isinstance(fromObj, list) else [fromObj]) if fromObj is not None else []

        def specialCase(checkWhat, total, size):
            if(checkWhat == "count"):
                return [], [float(total[0] / size[0])]
            else:
                checkWhat = fromTensorToItem(checkWhat)
                if(isinstance(checkWhat, tuple) and len(checkWhat) == 2):
                    return setAndCheckList(checkWhat[0]), setAndCheckList(checkWhat[1])
                return [], setAndCheckList(checkWhat)
        ###################################

        self.testLossSum = setAndCheckList(testLossSum)
        self.testCorrectSum = setAndCheckList(testCorrectSum)
        self.predSizeSum = setAndCheckList(predSizeSum)

        self.correctRatioStd, self.correctRatio = specialCase(checkWhat=correctRatio, 
            total=self.testCorrectSum, size=self.predSizeSum)
        '''
        if(correctRatio == "count"):
            self.correctRatio = [float(self.testCorrectSum[0] / self.predSizeSum[0])]
        else:
            self.correctRatio = setAndCheckList(correctRatio)
        '''

        self.trainLoopTimerSum = setAndCheckList(trainLoopTimerSum)
        self.testLoopTimerSum = setAndCheckList(testLoopTimerSum)

        self.lossRatioStd, self.lossRatio = specialCase(checkWhat=lossRatio,
            total=self.testLossSum, size=self.predSizeSum)
        '''
        if(lossRatio == "count"):
            self.lossRatio = [float(self.testLossSum[0] / self.predSizeSum[0])]
        else:
            self.lossRatio = setAndCheckList(lossRatio)
        '''
        self.trainTimeLoop = setAndCheckList(trainTimeLoop)
        self.avgTrainTimeLoop = setAndCheckList(avgTrainTimeLoop)
        self.trainTotalNumb = setAndCheckList(trainTotalNumb)
        self.trainTimeUnits = setAndCheckList(trainTimeUnits)

        self.testTimeLoop = setAndCheckList(testTimeLoop)
        self.avgTestTimeLoop = setAndCheckList(avgTestTimeLoop)
        self.testTimeUnits = setAndCheckList(testTimeUnits)

        self.smthTestTimeLoop = setAndCheckList(smthTestTimeLoop)
        self.smthAvgTestTimeLoop = setAndCheckList(smthAvgTestTimeLoop)
        self.smthTestTimeUnits = setAndCheckList(smthTestTimeUnits)

        #####################################
        self.smthTestLossSum = setAndCheckList(smthTestLossSum)
        self.smthTestCorrectSum = setAndCheckList(smthTestCorrectSum)
        self.smthPredSizeSum = setAndCheckList(smthPredSizeSum)

        self.smthLossRatioStd, self.smthLossRatio = specialCase(checkWhat=smthLossRatio, 
            total=self.smthTestLossSum, size=self.smthPredSizeSum)

        self.smthCorrectRatioStd, self.smthCorrectRatio = specialCase(checkWhat=smthCorrectRatio,
            total=self.smthTestCorrectSum, size=self.smthPredSizeSum)
        '''
        if(smthLossRatio == "count"):
            self.smthLossRatio = [float(self.smthTestLossSum[0] / self.smthPredSizeSum[0])]
        else:
            self.smthLossRatio = setAndCheckList(smthLossRatio)
        if(smthCorrectRatio == "count"):
            self.smthCorrectRatio = [float(self.smthTestCorrectSum[0] / self.smthPredSizeSum[0])]
        else:
            self.smthCorrectRatio = setAndCheckList(smthCorrectRatio)
        '''

    def averageOneFile(fileNameWithPath: str, runningAvgSize: int, outputFolder: str = None, inputFolder: str = None, 
        getBaseNameFile = False) -> str:
        """
            Metoda uśrednia wartości dla jednego pliku, przechodząc po nim ruchomym oknem.
            Zwraca nazwę nowego pliku połączoną ze zmienną 'outputFolder'.

            getBaseNameFile - wybiera nazwę pliku ze ścieżki 'fileNameWithPath'
        """
        # add '.avg' to the name of the file
        whereDot = fileNameWithPath.rfind(".")
        avg = ".avg"
        avgFileName = fileNameWithPath[:whereDot] + avg + fileNameWithPath[whereDot:]

        avgFileFolderName = avgFileName
        if(getBaseNameFile):
            avgFileFolderName = os.path.basename(avgFileFolderName)
        if(outputFolder is not None):
            avgFileFolderName = os.path.join(outputFolder, avgFileFolderName)

        folder_fileName = fileNameWithPath
        if(inputFolder is not None):
            folder_fileName = os.path.join(inputFolder, fileNameWithPath)

        if(checkForEmptyFile(avgFileFolderName)):
            Output.printBash("averageOneFile - file '{}' exist. The file will be overwritten".format(avgFileFolderName), 'warn')

        with open(avgFileFolderName, 'w') as fileAvgH, \
                open(folder_fileName, 'r') as fileH:
            counter = 0
            circularList = CircularList(runningAvgSize)
            for line in fileH.readlines():
                circularList.pushBack(float(line))
                fileAvgH.write(str(circularList.getAverage()) + '\n')
        return avgFileName

    def slidingWindow(fileNamesWithPaths: list, runningAvgSize, **kwargs) -> list:     
        """
            Metoda iteruje po wszsytkich podanych plikach 'fileNamesWithPaths', aby je pojedynczo uśrednić.
            Na końcu zwraca listę nowo stworzonych, uśrednionych plików wraz z ich względnymi nazwami 
            folderów, o ile takie istnieją.
        """
        if(runningAvgSize > 1):
            avgFileNames = []
            for fileNameWithPath in fileNamesWithPaths:
                avgFileNames.append(Statistics.averageOneFile(fileNameWithPath=fileNameWithPath, runningAvgSize=runningAvgSize, **kwargs))
            return avgFileNames
        return None

    def printPlots(self, runningAvgSize=1, **kwargs):
        """
            Metoda na podstawie danych zawartych w tym obiekcie rysuje wykresy.
            Foldery dla wykresów oraz nazwy osi brane są z instancji tego obiektu.

            kwargs - argumenty dla funckji plot()
        """
        for (name, xlabel, ylabel), val in self.plotBatches.items():
            if(val is None):
                Output.printBash("Some of the files to plot were not properly created. Instance ignored. Method Statistics.printPlots", 'warn')

            if(runningAvgSize > 1):
                avgName = name + ".avg"
                newylabel = ylabel
                out = Statistics.slidingWindow(fileNamesWithPaths=val, runningAvgSize=runningAvgSize, outputFolder=self.logFolder, inputFolder=self.rootInputFolder)
                if(out is not None):
                    newylabel = newylabel + ' (okno=' + str(runningAvgSize) + ')'
                idx = (avgName, xlabel, newylabel)
                self.avgPlotBatches[idx] = out

                plot(filePath=self.avgPlotBatches[idx], xlabel=xlabel, ylabel=newylabel, name=avgName, 
                    plotInputRoot=self.rootInputFolder, plotOutputRoot=self.logFolder, **kwargs)
            
            elif(runningAvgSize > 0):
                plot(filePath=val, xlabel=xlabel, ylabel=ylabel, name=name, plotInputRoot=self.rootInputFolder, 
                    plotOutputRoot=self.logFolder, **kwargs)
            else:
                raise Exception("Wrong parametr. Running average size must be greater than 0. Get: {}".format(runningAvgSize))

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def saveSelf(self, name, path=None):
        if(path is None):
            if(self.logFolder is not None):
                torch.save({"stat" : self}, os.path.join(self.logFolder, name))
            else:
                torch.save({"stat" : self}, os.path.join(Output.getTimeStr() + name))
        else:
            torch.save({"stat" : self}, os.path.join(path, name))

    def load(pathName):
        obj = torch.load(pathName)
        return obj['stat']

    def __str__(self):
        return '\n'.join("%s: %s" % item for item in vars(self).items())

class Data(SaveClass, BaseMainClass, BaseLogicClass):
    """
        Metody konieczne do przeciążenia, dla których wymaga się użycia super().
        __init__
        __setInputTransform__

        Metody konieczne do przeciążenia, dla których nie używa się super().
        __prepare__
        __update__
        __epoch__
        __howManyTestInvInOneEpoch__
        __howManyTrainInvInOneEpoch__

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

    def __howManyTestInvInOneEpoch__(self):
        """
            Zwraca liczbę wykonanych wywołań 'testLoop' w jednym epochu.
            Wartość ta jest pobierana przy każdej iteracji epocha.

            Zwracana wartość: integer >= 0
        """
        raise Exception("Not implemented")

    def __howManyTrainInvInOneEpoch__(self):
        """
            Zwraca liczbę wykonanych wywołań 'trainLoop' w jednym epochu.
            Wartość ta jest pobierana przy każdej iteracji epocha.

            Zwracana wartość: integer >= 0
        """
        raise Exception("Not implemented")

    def __train__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata, metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        """
        Główna logika treningu modelu. Następuje pomiar czasu dla wykonania danej metody.
        """
        
        # forward + backward + optimize
        #print(torch.cuda.memory_summary(device='cuda:0'))
        helper.outputs = model.getNNModelModule()(helper.inputs)
        helper.loss = model.__getLossFun__()(helper.outputs, helper.labels)
        #print(torch.cuda.memory_summary())
        helper.loss.backward()
        #print(torch.cuda.memory_summary())
        model.__getOptimizer__().step()

        # run smoothing
        helper.smoothingSuccess = smoothing(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, smoothingMetadata=smoothingMetadata, metadata=metadata)

    def setTrainLoop(self, model: 'Model', modelMetadata: 'Model_Metadata', dataMetadata: 'dataMetadata', metadata: 'Metadata'):
        helper = TrainDataContainer()
        metadata.prepareOutput()
        helper.size = (StaticData.MAX_DEBUG_LOOPS * dataMetadata.batchTrainSize if test_mode.isActive() else len(self.trainloader.dataset))
        helper.timer = Timer()
        helper.loopTimer = Timer()
        helper.loss = None 
        helper.diff = None

        return helper

    def __beforeTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):      
        model.getNNModelModule().train()

    def __beforeTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        helper.inputs, helper.labels = helper.inputs.to(modelMetadata.device), helper.labels.to(modelMetadata.device)
        model.__getOptimizer__().zero_grad()

    def __afterTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        metadata.stream.print(helper.loss.item(), ['statLossTrain'])

    def __afterTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        metadata.stream.print("Train summary:")
        metadata.stream.print(f" Average train time ({helper.timer.getUnits()}): {helper.timer.getAverage()}")
        metadata.stream.print(f" Loop train time ({helper.timer.getUnits()}): {helper.loopTimer.getTimeSum()}")
        metadata.stream.print(f" Number of batches done in total: {helperEpoch.trainTotalNumber}")
        helperEpoch.statistics.trainTimeLoop.append(helper.loopTimer.getTimeSum())
        helperEpoch.statistics.trainTimeUnits.append(helper.timer.getUnits())
        helperEpoch.statistics.avgTrainTimeLoop.append(helper.timer.getAverage())
        helperEpoch.statistics.trainTotalNumb.append(helperEpoch.trainTotalNumber)

    def __trainLoopExit__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        helperEpoch.loopsState.imprint(numb=helper.batchNumber, isEnd=helper.loopEnded)

    def trainLoopTearDown(self):
        self.trainHelper = None

    def trainLoop(self, model: 'Model', helperEpoch: 'EpochDataContainer', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        """
        Główna logika pętli treningowej.
        """
        total = 0.0
        correct = 0.0

        startNumb = helperEpoch.loopsState.decide()
        if(startNumb is None):
            self.trainLoopTearDown()
            return # loop already ended. This state can occur when framework was loaded from file.

        if(self.trainHelper is None): # jeżeli nie było wznowione; nowe wywołanie
            self.trainHelper = self.setTrainLoop(model=model, modelMetadata=modelMetadata, dataMetadata=dataMetadata, metadata=metadata)
        
        self.trainHelper.loopTimer.clearTime()
        self.__beforeTrainLoop__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        metadata.stream.print("Starting train batch at: {}".format(startNumb), "debug:0")

        self.trainHelper.loopTimer.start(cudaDeviceSynch=smoothingMetadata.device)
        for batch, (inputs, labels) in enumerate(self.trainloader):
            if(batch < startNumb): # already iterated
                continue

            self.trainHelper.inputs = inputs
            self.trainHelper.labels = labels
            self.trainHelper.batchNumber = batch
            if(enabledSaveAndExit()):
                metadata.stream.print("Triggered SAVE_AND_EXIT_FLAG.", "debug:0")
                self.__trainLoopExit__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                self.trainLoopTearDown()
                return

            if(StaticData.TEST_MODE and batch >= StaticData.MAX_DEBUG_LOOPS):
                metadata.stream.print("In test mode, triggered max loops which is {} iteration. Breaking train loop.".format(StaticData.MAX_DEBUG_LOOPS), "debug:0")
                if(StaticData.TEST_MODE and helperEpoch.epochNumber >= StaticData.MAX_EPOCH_DEBUG_LOOPS):
                    helperEpoch.endEpoches = True
                break

            helperEpoch.trainTotalNumber += 1
            self.__beforeTrain__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

            self.trainHelper.timer.clearTime()
            self.trainHelper.timer.start(cudaDeviceSynch=smoothingMetadata.device)
            
            self.__train__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

            self.trainHelper.timer.end(cudaDeviceSynch=smoothingMetadata.device)
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
                    metadata.stream.print("Successful first smoothing call while training at batch {}, epoch {}".format(batch, helperEpoch.epochNumber), ['model:0', 'debug:0'])
                    helperEpoch.firstSmoothingSuccess = True
                else:
                    metadata.stream.print("Successful smoothing call while training at batch {}, epoch {}".format(batch, helperEpoch.epochNumber), 'debug:0')

            self.__afterTrain__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

            if(self.trainHelper.smoothingSuccess and smoothing.__isSmoothingGoodEnough__(
                helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, 
                modelMetadata=modelMetadata, metadata=metadata, smoothingMetadata=smoothingMetadata)
            ):
                helperEpoch.endEpoches = True
                break

            self.trainHelper.smoothingSuccess = False

            total += labels.size(0)
            correct += torch.argmax(self.trainHelper.outputs, dim=1).eq(self.trainHelper.labels.data).cpu().sum()
            

        self.trainHelper.loopTimer.end(cudaDeviceSynch=smoothingMetadata.device)
        self.trainHelper.loopTimer.addToStatistics()
        self.trainHelper.loopEnded = True

        metadata.stream.print("Train epoch accuracy: {}%".format(100.*((correct/total) if total != 0 else 0)), "model:0")

        self.__afterTrainLoop__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        self.__trainLoopExit__(helperEpoch=helperEpoch, helper=self.trainHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

        helperEpoch.statistics.trainLoopTimerSum.append(self.trainHelper.loopTimer.getTimeSum())
        metadata.stream.print('Train time;', alias='stat')
        metadata.stream.print(str(helperEpoch.statistics.trainLoopTimerSum[-1]) + ';', alias='stat')

        self.trainLoopTearDown()
        
    def __test__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        """
        Główna logika testu modelu. Następuje pomiar czasu dla wykonania danej metody.
        """
        
        helper.pred = model.getNNModelModule()(helper.inputs)
        helper.test_loss = model.__getLossFun__()(helper.pred, helper.labels).item()

    def setTestLoop(self, model: 'Model', modelMetadata: 'Model_Metadata', dataMetadata: 'dataMetadata', metadata: 'Metadata'):
        helper = TestDataContainer()
        metadata.prepareOutput()
        helper.size = (StaticData.MAX_DEBUG_LOOPS * dataMetadata.batchTestSize if test_mode.isActive() else len(self.testloader.dataset))
        helper.test_loss, helper.test_correct = 0, 0
        helper.timer = Timer()
        helper.loopTimer = Timer()
        return helper

    def __beforeTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        model.getNNModelModule().eval()

    def __beforeTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        helper.inputs = helper.inputs.to(modelMetadata.device)
        helper.labels = helper.labels.to(modelMetadata.device)

    def __afterTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        helper.testLossSum += helper.test_loss
        helper.test_correct = (helper.pred.argmax(1) == helper.labels).type(torch.float).sum().item()
        helper.testCorrectSum += helper.test_correct

    def __afterTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'): 
        if(helperEpoch.averaged):
            helperEpoch.statistics.smthTestLossSum.append(helper.testLossSum)
            helperEpoch.statistics.smthTestCorrectSum.append(helper.testCorrectSum)
            helperEpoch.statistics.smthPredSizeSum.append(helper.predSizeSum)
            
            helperEpoch.statistics.smthLossRatio.append(helper.testLossSum / helper.predSizeSum)
            helperEpoch.statistics.smthCorrectRatio.append(helper.testCorrectSum / helper.predSizeSum)
            
            metadata.stream.print(f"\nTest summary: \n Accuracy: {(100*helperEpoch.statistics.smthCorrectRatio[-1]):>6f}%, Avg loss: {helperEpoch.statistics.smthLossRatio[-1]:>8f}", ['model:0'])
            metadata.stream.print(f" Average test execution time in a loop ({helper.timer.getUnits()}): {helper.timer.getAverage():>3f}", ['model:0'])
            metadata.stream.print(f" Time to complete the entire loop ({helper.timer.getUnits()}): {helper.loopTimer.getTimeSum():>3f}\n", ['model:0'])

            helperEpoch.statistics.smthTestTimeLoop.append(helper.loopTimer.getTimeSum())
            helperEpoch.statistics.smthAvgTestTimeLoop.append(helper.timer.getAverage())
            helperEpoch.statistics.smthTestTimeUnits.append(helper.timer.getUnits())

        else:
            helperEpoch.statistics.testLossSum.append(helper.testLossSum)
            helperEpoch.statistics.testCorrectSum.append(helper.testCorrectSum)
            helperEpoch.statistics.predSizeSum.append(helper.predSizeSum)
            
            helperEpoch.statistics.lossRatio.append(helper.testLossSum / helper.predSizeSum)
            helperEpoch.statistics.correctRatio.append(helper.testCorrectSum / helper.predSizeSum)
            
            metadata.stream.print(f"\nTest summary: \n Accuracy: {(100*helperEpoch.statistics.correctRatio[-1]):>6f}%, Avg loss: {helperEpoch.statistics.lossRatio[-1]:>8f}", ['model:0'])
            metadata.stream.print(f" Average test execution time in a loop ({helper.timer.getUnits()}): {helper.timer.getAverage():>3f}", ['model:0'])
            metadata.stream.print(f" Time to complete the entire loop ({helper.timer.getUnits()}): {helper.loopTimer.getTimeSum():>3f}\n", ['model:0'])

            helperEpoch.statistics.testTimeLoop.append(helper.loopTimer.getTimeSum())
            helperEpoch.statistics.avgTestTimeLoop.append(helper.timer.getAverage())
            helperEpoch.statistics.testTimeUnits.append(helper.timer.getUnits())

    def __testLoopExit__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        helperEpoch.loopsState.imprint(numb=helper.batchNumber, isEnd=helper.loopEnded)
    
    def testLoopTearDown(self):
        self.testHelper = None

    def testLoop(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        startNumb = helperEpoch.loopsState.decide()
        if(startNumb is None):
            self.testLoopTearDown()
            return # loop already ended. This state can occur when framework was loaded from file.
        
        if(self.testHelper is None): # jeżeli nie było wznowione; nowe wywołanie
            self.testHelper = self.setTestLoop(model=model, modelMetadata=modelMetadata, dataMetadata=dataMetadata, metadata=metadata)

        self.testHelper.loopTimer.clearTime()
        #torch.cuda.empty_cache()
        self.__beforeTestLoop__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

        with torch.no_grad():
            self.testHelper.loopTimer.start(cudaDeviceSynch=smoothingMetadata.device)
            for batch, (inputs, labels) in enumerate(self.testloader):
                if(batch < startNumb): # already iterated
                    continue
                self.testHelper.inputs = inputs
                self.testHelper.labels = labels
                self.testHelper.batchNumber = batch

                if(enabledSaveAndExit()):
                    self.__testLoopExit__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                    self.testLoopTearDown()
                    return

                if(StaticData.TEST_MODE and batch >= StaticData.MAX_DEBUG_LOOPS):
                    break
                
                helperEpoch.testTotalNumber += 1
                self.__beforeTest__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

                self.testHelper.timer.clearTime()
                self.testHelper.timer.start(cudaDeviceSynch=smoothingMetadata.device)
                self.__test__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                self.testHelper.timer.end(cudaDeviceSynch=smoothingMetadata.device)
                if(helperEpoch.currentLoopTimeAlias is None and warnings()):
                    Output.printBash("Alias for test loop file was not set. Variable helperEpoch.currentLoopTimeAlias may be set" +
                    " as:\n\t'loopTestTime_normal'\n\t'loopTestTime_smooothing'\n\t'loopTrainTime'\n", 'warn')
                else:
                    metadata.stream.print(self.testHelper.timer.getDiff() , helperEpoch.currentLoopTimeAlias)
                self.testHelper.timer.addToStatistics()

                self.testHelper.predSizeSum += labels.size(0)
                self.__afterTest__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

            self.testHelper.loopTimer.end(cudaDeviceSynch=smoothingMetadata.device)
            self.testHelper.loopTimer.addToStatistics()
            self.testHelper.loopEnded = True

        self.__afterTestLoop__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        self.__testLoopExit__(helperEpoch=helperEpoch, helper=self.testHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        helperEpoch.statistics.testLoopTimerSum.append(self.testHelper.loopTimer.getTimeSum())
        self.testLoopTearDown()

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

    def _updateTotalNumbLoops(self, dataMetadata: 'Data_Metadata'):
        if(test_mode.isActive()):
            self.epochHelper.maxTrainTotalNumber = self.__howManyTrainInvInOneEpoch__() * (dataMetadata.epoch if dataMetadata.epoch < StaticData.MAX_EPOCH_DEBUG_LOOPS else StaticData.MAX_EPOCH_DEBUG_LOOPS) * StaticData.MAX_DEBUG_LOOPS
            self.epochHelper.maxTestTotalNumber = self.__howManyTestInvInOneEpoch__() * (dataMetadata.epoch if dataMetadata.epoch < StaticData.MAX_EPOCH_DEBUG_LOOPS else StaticData.MAX_EPOCH_DEBUG_LOOPS) * StaticData.MAX_DEBUG_LOOPS
        else:
            self.epochHelper.maxTrainTotalNumber = self.__howManyTrainInvInOneEpoch__() * dataMetadata.epoch * len(self.trainloader)
            self.epochHelper.maxTestTotalNumber = self.__howManyTestInvInOneEpoch__() * dataMetadata.epoch * len(self.testloader)

    def setEpochLoop(self, metadata: 'Metadata'):
        epochHelper = EpochDataContainer()
        epochHelper.statistics.logFolder = metadata.stream.root
        epochHelper.statistics.rootInputFolder = metadata.stream.root
        epochHelper.trainTotalNumber = 0
        epochHelper.testTotalNumber = 0
        epochHelper.addNormalMode()
        return epochHelper

    def epochLoopTearDown(self):
        del self.epochHelper
        self.epochHelper = None

    def epochLoop(self, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        metadata.prepareOutput()
        if(self.epochHelper is None):
            self.epochHelper = self.setEpochLoop(metadata)

        self.__beforeEpochLoop__(helperEpoch=self.epochHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

        for ep, (loopEpoch) in enumerate(range(dataMetadata.epoch), start=1):  # loop over the dataset multiple times
            if(ep < self.epochHelper.epochNumber): # already iterated
                continue
            if(StaticData.TEST_MODE and ep >= StaticData.MAX_EPOCH_DEBUG_LOOPS):
                metadata.stream.print("\nEnding debug epoch loop at epoch {}\n-------------------------------".format(ep))
                self.epochHelper.endEpoches = True
                break
            self.epochHelper.epochNumber = ep
            self._updateTotalNumbLoops(dataMetadata=dataMetadata)
            metadata.stream.print(f"\nEpoch {loopEpoch+1}\n-------------------------------")
            metadata.stream.flushAll()
            
            self.__epoch__(helperEpoch=self.epochHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
            metadata.stream.print(f"\nEpoch End\n-------------------------------")
            if(self.epochHelper.endEpoches):
                metadata.stream.print("\nEnding epoch loop sooner at epoch {}\n-------------------------------".format(ep))
                break

            model.schedulerStep(epochNumb=ep, metadata=metadata, shtypes=self.epochHelper.modes, metrics=self.epochHelper.statistics.testLossSum[-1]) # get lasts sum of losses

            if(enabledSaveAndExit()):
                self.__epochLoopExit__(helperEpoch=self.epochHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                self.epochLoopTearDown()
                return

        self.__afterEpochLoop__(helperEpoch=self.epochHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        self.__epochLoopExit__(helperEpoch=self.epochHelper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

        a = metadata.stream.getFileName('loopTrainTime')
        b = metadata.stream.getFileName('loopTestTime_normal')
        c = metadata.stream.getFileName('loopTestTime_smooothing')
        self.epochHelper.statistics.plotBatches[('loopTimeTrain', 'liczba iteracji pętli treningowej', 'czas (s)')] = [a]
        self.epochHelper.statistics.plotBatches[('loopTimeTest', 'liczba iteracji pętli treningowej', 'czas (s)')] = [b, c]

        a = metadata.stream.getFileName('statLossTrain')
        b = metadata.stream.getFileName('statLossTest_normal')
        c = metadata.stream.getFileName('statLossTest_smooothing')
        self.epochHelper.statistics.plotBatches[('lossTrain', 'liczba iteracji pętli treningowej', 'strata modelu')] = [a]
        self.epochHelper.statistics.plotBatches[('lossTest', 'liczba iteracji pętli treningowej', 'strata modelu')] = [b, c]

        a = metadata.stream.getFileName('weightsSumTrain')
        self.epochHelper.statistics.plotBatches[('weightsSumTrain', 'liczba iteracji pętli treningowej', 'suma wag modelu')] = [a]

        self.resetEpochState()
        metadata.stream.flushAll()
        stat = self.epochHelper.statistics
        self.epochLoopTearDown()
        return stat

    def resetEpochState(self):
        self.epochHelper.loopsState.clear()

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

    def setModelNormalWeights(self, model, helperEpoch, weights, metadata):
        model._Private_setWeights(weights=weights, metadata=metadata)
        helperEpoch.averaged = False

    def setModelSmoothedWeights(self, model, helperEpoch, weights, metadata):
        model._Private_setWeights(weights=weights, metadata=metadata)
        helperEpoch.averaged = True


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

        self.savedWeightsState = {}

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
        Jeżeli zwraca None, to wygładzanie nie zostało poprawnie skonfigurowane.
        Gdy istnieje możliwość zwrócenia wag, zwraca pusty słownik.
        """
        if(self.enabled == False):
            return None
        else:
            return {}

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata) -> bool:
        """
            Zostaje wywołane tylko wtedy, gdy w danej iteracji pętli pomyślnie wywołano wygładzanie (__call__)
        """
        raise Exception("Not implemented.")

    def getWeights(self, key, toDevice=None, copy = False):
        """
            Zwróć wcześniej zapisane wagi.
        """
        if(key in self.savedWeightsState.keys()):
            if(copy):
                return cloneTorchDict(self.savedWeightsState[key], toDevice)
            else:
                return moveToDevice(self.savedWeightsState[key], toDevice)
        else:
            Output.printBash("Smoothing: could not find key '{}' while searching for weights.".format(key), 'warn')
            return None

    def __setDictionary__(self, smoothingMetadata, dictionary):
        """
        Used to map future weights into internal sums.
        """
        self.enabled = True

    def saveWeights(self, weights, key, canOverride = True, toDevice = None):
        """
            Zapisz wagi modelu.
        """
        with torch.no_grad():
            if(canOverride):
                self.savedWeightsState[key] = cloneTorchDict(weights, toDevice)
            elif(key in self.savedWeightsState.keys()):
                Output.printBash("The given key '{}' was found during the cloning of the scales. No override flag was specified.".format(key), 'warn')
                self.savedWeightsState[key] = cloneTorchDict(weights, toDevice)

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata=metadata, suffix=StaticData.SMOOTHING_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.SMOOTHING_SUFFIX

    def canUpdate(self = None):
        return True

class SchedulerContainer():
    def __init__(self, schedType: str, importance: int):
        """
            W tym obiekcie może mieścić się wiele schedulerów, które posiadają tą samą wartość 'schedType' oraz 'importance'.
            Po wywołaniu step(), wszystkie schedulery znajdujące się w tym obiekcie zostaną wywołane, jeżeli spełnią określone warunki,
            w tym zgodność z ich typem oraz numerem epoki.
            Wywołanie schedulera jest oznaczone określonym komentarzem w logu 'model:0'.
            Trzeba zaznaczyć, iż sam scheduler nie zawsze musi zmieniać wartość learning rate, dlatego wywołanie go nie 
            jest równoznaczne ze zmianą tego parametru.

            importance - w jakiej kolejności wywołać obiekty typu SchedulerContainer
            Im większa wartość, tym bardziej ważny jest dany obiekt. Interpretacja tego parametru nie jest zależna od tej klasy.
            Stosuje się ją wraz z klasą SchedulerManager.
        """
        if(schedType != 'smoothing' and schedType != 'normal'):
            raise Exception("Scheduler type can only be 'smoothing' or 'normal'.")
        if(not isinstance(importance, int)):
            raise Exception("Scheduler importance can be only of the type int.")
        self.schedType = schedType
        self.schedulers = []
        self.importance = importance

    def getType(self):
        return self.schedType

    def getImportance(self):
        return self.importance

    def add(self, schedule: list, scheduler, metric: bool):
        """
            metric - czy podany scheduler przyjmuje metrykę. Jest ono podane jako scheduler.step(metrics=metrics). 
            schedule - dla jakich epok wywołać dany scheduler
        """
        self.schedulers.append((schedule, scheduler, metric))
        return self

    def _schedulerStep(self, epochNumb, epochStep):
        return epochStep is None \
            or not epochStep \
            or epochNumb in epochStep

    def step(self, shtypes: [str, list], epochNumb: int, metadata, metrics):
        """
            Zwraca True, gdy typ schedulera zgadza się z wartością lub którąś z wartości w zmiennej 'shtypes'.
            Wartość True zostaje również zwrócona nawet, kiedy żaden scheduler nie został wywołany.
        """
        if( (isinstance(shtypes, list) and self.schedType in shtypes) or self.schedType == shtypes):
            for epochStep, scheduler, metric in self.schedulers:
                if(self._schedulerStep(epochNumb=epochNumb, epochStep=epochStep)):
                    if(metric): # jeżeli podano, że chce się użyć metryki, metric != metrics
                        scheduler.step(metrics=metrics)
                    else:
                        scheduler.step()

                    tmpstr = 'learning rate:\t'
                    tmpstr = str([group['lr'] for group in scheduler.optimizer.param_groups]) + ',\t'
                    metadata.stream.print("Set learning rate to {} of a scheduler {} in mode: {} metric: {}".format(
                        tmpstr, str(type(scheduler)), self.schedType, str(metrics)), ['model:0'])
            return True
        return False
        
class SchedulerManager():
    def __init__(self):
        self.schedulers = []

    def add(self, schedContainer):
        if(not isinstance(schedContainer, SchedulerContainer)):
            raise Exception("Variable schedContainer is not the type of SchedulerContainer.")

        self.schedulers.append(schedContainer) 
        self.schedulers.sort(key=lambda x : x.getImportance())

    def schedulerStep(self, epochNumb, metadata, shtypes: Union[str, list], metrics):
        """
            Jeżeli któryś z obiektów klasy SchedulerContainer zostanie pomyślnie wywołany,
            wtedy inne obiekty tego typu nie zostaną wywołane.
            Każdy z obiektów SchedulerContainer jest posortowany po jego wazności.
        """
        if(self.schedulers is not None):
            for schedulerObj in self.schedulers:
                if(schedulerObj.step(shtypes=shtypes, epochNumb=epochNumb, metadata=metadata, metrics=metrics)):
                    return

class __BaseModel(nn.Module, SaveClass, BaseMainClass, BaseLogicClass):
    def __init__(self):
        super().__init__()

        self.loss_fn = None
        self.optimizer = None
        self.schedulers = SchedulerManager()

    def prepare(self, lossFunc, optimizer, schedulers: list=None):
        """
            Set optimizer and loss function.

            schedulers - a list of objects of the type SchedulerContainer.
        """
        self.loss_fn = lossFunc
        self.optimizer = optimizer
        if(schedulers is not None):
            for s in schedulers:
                self.schedulers.add(s)

    def _step(self, metadata, scheduler):
        scheduler.step()
        metadata.stream.print("Set learning rate to {}".format(scheduler.get_last_lr()), ['model:0'])

    def schedulerStep(self, epochNumb, metadata, shtypes: Union[str, list], metrics):
        """
            Jeżeli któryś z obiektów klasy SchedulerContainer zostanie pomyślnie wywołany,
            wtedy inne obiekty tego typu nie zostaną wywołane.
            Każdy z obiektów SchedulerContainer jest posortowany po jego wazności.
        """
        self.schedulers.schedulerStep(epochNumb=epochNumb, metadata=metadata, shtypes=shtypes, metrics=metrics)

    def canUpdate(self = None):
        return True

    def __getOptimizer__(self):
        return self.optimizer

    def __getLossFun__(self):
        return self.loss_fn


class Model(__BaseModel):
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
        self.getNNModelModule().to(modelMetadata.device)

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

    def _Private_setWeights(self, weights, metadata):
        self.load_state_dict(weights)

    def getWeights(self):
        return self.state_dict()
    
    def __update__(self, modelMetadata):
        raise Exception("Not implemented")

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata=metadata, suffix=StaticData.MODEL_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.MODEL_SUFFIX

    def getNNModelModule(self):
        """
        Używany, gdy chcemy skorzystać z funckji modułu nn.Module. Zwraca obiekt dla którego jest pewność, że implementuje klasę nn.Module. 
        """
        return self

class PredefinedModel(__BaseModel):
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
    def __init__(self, obj: 'modelObject', modelMetadata, name):
        """
        Metoda powinna posiadać zmienne\n
        self.loss_fn = ...\n
        self.optimizer = torch.optim...\n
        """

        if(not isinstance(obj, nn.Module)):
            raise Exception("Object do not implement nn.Module class.")
        super().__init__()
        self.modelObj = obj
        self.name = name
        self.modelObj.to(modelMetadata.device)

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

    def _Private_setWeights(self, weights, metadata):
        self.modelObj.load_state_dict(weights)

    def getWeights(self):
        return self.modelObj.state_dict()

    def __update__(self, modelMetadata):
        raise Exception("Not implemented")

    def __strAppend__(self):
        return "Model name:\t{}\n".format(self.name)

    def trySave(self, metadata, onlyKeyIngredients = False, temporaryLocation = False):
        return super().trySave(metadata=metadata, suffix=StaticData.PREDEFINED_MODEL_SUFFIX, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

    def getFileSuffix(self = None):
        return StaticData.PREDEFINED_MODEL_SUFFIX

    def getNNModelModule(self):
        """
        Używany, gdy chcemy skorzystać z funckji modułu nn.Module. Zwraca obiekt dla którego jest pewność, że implementuje klasę nn.Module. 
        """
        return self.modelObj


def tryLoad(tupleClasses: list, metadata, temporaryLocation = False):
    """
        Wczytuje podane w klasie metadata dane do odczytania klas zawartych w tupleClasses. Zwraca słownik z wczytanymi obiektami.
        tupleClasses - posiada dwie zmienne: klasa metadanych oraz klasa implementująca logikę, korzystająca z podanej klasy metadanych.
        metadata - ogólna klasa metadanych

    """
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
    """
        Zapisuje podane w słowniku obiekty. Miejsce do zapisu znajduje się w obiekcie Metadata, 
        który również musi zostać podany.
    """
    dictObjs['Metadata'].trySave(onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)
    md = dictObjs['Metadata']
    
    for key, obj in dictObjs.items():
        if(key != 'Metadata'):
            obj.trySave(metadata=md, onlyKeyIngredients=onlyKeyIngredients, temporaryLocation=temporaryLocation)

def commandLineArg(metadata, dataMetadata, modelMetadata, argv, enableLoad = True, enableSave = True):
    """
        Deprecated.
    """
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

def averageStatistics(statistics: list, filePaths: dict=None, outputRelativeRootFolder = None, outputFolderNameSuffix = None) -> Statistics:
    """
        Funkcja dla podanych logów w formacie csv uśrednia je oraz zwraca obiekt statystyk. Logi muszą być w odpowiednim formacie, czyli:
            - jeden plik jest przeznaczony dla jednego rodzaju danych w formacie liczby, zapisywanych kolejnych
                liniach
            - plik nie może zawierać innych danych oprócz liczby, która zostanie przedstawiona w formacie
                zmiennoprzecinkowym

        statistics - lista obiektów typu Statistics. To z nich będą pobierane dane. Powinny one reprezentować to samo zjawisko.
            Funkcja uśrednia wartości względem danego indeksu z csv. Przykładowo, podając kilka powtórzeń tego samego eksperymentu, 
            funkcja zwraca uśrednioną klasę statystyk.

        filePaths - wartość domyślna dla None - dict = {
            ('loopTestTime', 'liczba iteracji pętli treningowej', 'czas (s)') : ['loopTestTime_normal.csv', 'loopTestTime_smooothing.csv'], 
            ('loopTrainTime', 'liczba iteracji pętli treningowej', 'czas (s)') : ['loopTrainTime.csv'], 
            ('lossTest', 'liczba iteracji pętli treningowej', 'strata modelu') : ['statLossTest_normal.csv', 'statLossTest_smooothing.csv'], 
            ('lossTrain', 'liczba iteracji pętli treningowej', 'strata modelu') : ['statLossTrain.csv'], 
            ('weightsSumTrain', 'liczba iteracji pętli treningowej', 'suma wag modelu') : ['weightsSumTrain.csv']}

            Zmienna ta przechowuje dane odnośnie plików csv. W nowo stworzonej klasie statystyk ta struktura
            jest do niej kopiowana bez zmian. Funkcja tworzy nowy folder, w którym pliki zawarte w listach tej zmiennej są zapisywane. 
            
            Nowy, uśredniony plik tworzony jest na podstawie wszystkich plików o tej samej nazwie. 
            Jeżeli któryś z plików danej kategorii nie istnieje lub jest pusty, to domyślnymi jego wartościami będą 0.0. 

        outputRelativeRootFolder - folder w którym zostaną zapisane uśrednione logi. Jeżeli None, to folder zostanie stworzony 
            względem folderu w zmiennej StaticData.LOG_FOLDER. Zaleca się go podanie.
        outputFolderNameSuffix - sufiks nowego folderu. Jeżeli None, to przyjmuje wartość 'averaging_files'
    """
    if(outputFolderNameSuffix is None):
        outputFolderNameSuffix = "averaging_files"

    filePaths = filePaths if filePaths is not None else {
            ('loopTestTime', 'liczba iteracji pętli treningowej', 'czas (s)') : ['loopTestTime_normal.csv', 'loopTestTime_smooothing.csv'], 
            ('loopTrainTime', 'liczba iteracji pętli treningowej', 'czas (s)') : ['loopTrainTime.csv'], 
            ('lossTest', 'liczba iteracji pętli treningowej', 'strata modelu') : ['statLossTest_normal.csv', 'statLossTest_smooothing.csv'], 
            ('lossTrain', 'liczba iteracji pętli treningowej', 'strata modelu') : ['statLossTrain.csv'], 
            ('weightsSumTrain', 'liczba iteracji pętli treningowej', 'suma wag modelu') : ['weightsSumTrain.csv']}

    def addLast(to, fromObj, mayBeEmpty=False):
        if(fromObj):
            to.append(fromObj[-1])
        elif(mayBeEmpty):
            return

    def addAll(to, fromObj):
        to += fromObj

    def setFirst(fromObj, default):
        if(fromObj):
            return fromObj[0]
        else:
            return default

    def setTimeUnit(to, fromObj, default):
        if(not fromObj):
            return default
        if(to is not None and fromObj and to != fromObj[0]):
            Output.printBash("Statistics have different time units when averaging. The time units displayed may be incorrect", 'warn')
        return setFirst(fromObj, default)

    def safeDivide(va, toDiv, default):
        if(toDiv == 0 or toDiv == 0.0):
            return default
        return va / toDiv
    
    def removeDuplicates(seq: list):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def addToBuffer(fileToOpen, newVals: list, index: int):
        with open(fileToOpen) as fh:
            #badSize = False
            resizeSize = []
            rows = [float(l.rstrip("\n")) for l in fh] # wczytaj liczby do listy


            if(len(rows) < len(newVals[index])): # jeżeli trzeba rozszerzyć bufor rows
                resize = len(newVals[index]) - len(rows)
                resizeSize.append(resize)
                rows = rows + [0.0 for _ in range(resize)]
                #badSize = True
            if(len(rows) > len(newVals[index])): # jeżeli trzeba rozszerzyć bufor newVals[index]
                resize = len(rows) - len(newVals[index])
                resizeSize.append(resize)
                newVals[index] = newVals[index] + [0.0 for _ in range(resize)]
                #badSize = True
            # else len == len: OK

            # dodaj do bufora
            newVals[index] = list(map(operator.add, rows, newVals[index]))
            
            # warunek niepotrzebny, ponieważ newVals[index] na samym początku może być pustą listą
            #if(badSize):
            #    Output.printBash("averageStatistics - one of the files have bad size: {}\nThe buffer has been increased by the needed values {}. ".format(fileToOpen, resizeSize), 'warn')
    
    
    flattedNewVals = []
    config = [] # lista plików wziętych pod uwagę wraz ze ścieżkami
    flattedFilePaths = []
    tmp_testLossSum = []
    tmp_testCorrectSum = []
    tmp_predSizeSum = []

    tmp_smthTestLossSum = []
    tmp_smthTestCorrectSum = []
    tmp_smthPredSizeSum = []
    numOfAvgFiles = len(statistics)

    if(numOfAvgFiles == 0):
        raise Exception("Cannot average statistics. No statistics to average.")

    tmp_trainTimeLoop    = []
    tmp_trainTimeUnits       = None
    tmp_avgTrainTimeLoop = []
    tmp_avgTrainTimeLoopCount = 0
    tmp_trainTotalNumb   = []

    tmp_testTimeLoop        = []
    tmp_avgTestTimeLoop     = []
    tmp_avgTestTimeLoopCount = 0
    tmp_testTimeUnits       = None
    tmp_smthTestTimeLoop    = []
    tmp_smthAvgTestTimeLoop = []
    tmp_smthAvgTestTimeLoopCount = 0
    tmp_smthTestTimeUnits   = None

    # potrzebne są tylko informacje odnośnie samych plików
    for f in filePaths.values():
        flattedFilePaths += f

    flattedFilePaths = removeDuplicates(flattedFilePaths)

    for _ in range(len(flattedFilePaths)):
        flattedNewVals.append([])

    for st in statistics: # przechodź kolejno po wszystkich folderach z obiektu statystyk
        for index, ffile in enumerate(flattedFilePaths): # iteruj po wszystkich plikach z danego folderu
            openPath = os.path.join(st.logFolder, ffile)
            config.append(openPath)
            addToBuffer(fileToOpen=openPath, newVals=flattedNewVals, index=index)

        # dodaj do statystyk sumy
        addLast(tmp_testLossSum, st.testLossSum, True)
        addLast(tmp_testCorrectSum, st.testCorrectSum, True)
        addLast(tmp_predSizeSum, st.predSizeSum, True)

        addLast(tmp_smthTestLossSum, st.smthTestLossSum, True)
        addLast(tmp_smthTestCorrectSum, st.smthTestCorrectSum, True)
        addLast(tmp_smthPredSizeSum, st.smthPredSizeSum, True)

        # sprawdź jednostkę czasu
        tmp_trainTimeUnits = setTimeUnit(tmp_trainTimeUnits, st.trainTimeUnits, "s")
        tmp_testTimeUnits = setTimeUnit(tmp_testTimeUnits, st.testTimeUnits, "s")
        tmp_smthTestTimeUnits = setTimeUnit(tmp_smthTestTimeUnits, st.smthTestTimeUnits, "s")

        # dodaj do statystyk czasy
        addAll(tmp_trainTimeLoop, st.trainTimeLoop)
        addAll(tmp_avgTrainTimeLoop, st.avgTrainTimeLoop)
        tmp_avgTrainTimeLoopCount += len(st.avgTrainTimeLoop)
        addLast(tmp_trainTotalNumb, st.trainTotalNumb)

        addAll(tmp_testTimeLoop, st.testTimeLoop)
        addAll(tmp_avgTestTimeLoop, st.avgTestTimeLoop)
        tmp_avgTestTimeLoopCount += len(st.avgTestTimeLoop)

        addAll(tmp_smthTestTimeLoop, st.smthTestTimeLoop)
        addAll(tmp_smthAvgTestTimeLoop, st.smthAvgTestTimeLoop)
        tmp_smthAvgTestTimeLoopCount += len(st.smthAvgTestTimeLoop)

    newStats = Statistics(
        testLossSum=torch.mean(torch.as_tensor(tmp_testLossSum, dtype=torch.float64)).item(),
        testCorrectSum=torch.mean(torch.as_tensor(tmp_testCorrectSum, dtype=torch.float64)).item(),
        predSizeSum=torch.mean(torch.as_tensor(tmp_predSizeSum, dtype=torch.float64)).item(),

        lossRatio=torch.std_mean(torch.as_tensor([x / y for x, y in zip(tmp_testLossSum, tmp_predSizeSum)], dtype=torch.float64)),
        correctRatio=torch.std_mean(torch.as_tensor([x / y for x, y in zip(tmp_testCorrectSum, tmp_predSizeSum)], dtype=torch.float64)),

        smthTestLossSum=torch.mean(torch.as_tensor(tmp_smthTestLossSum, dtype=torch.float64)).item(),
        smthTestCorrectSum=torch.mean(torch.as_tensor(tmp_smthTestCorrectSum, dtype=torch.float64)).item(),
        smthPredSizeSum=torch.mean(torch.as_tensor(tmp_smthPredSizeSum, dtype=torch.float64)).item(),

        smthLossRatio=torch.std_mean(torch.as_tensor([x / y for x, y in zip(tmp_smthTestLossSum, tmp_smthPredSizeSum)], dtype=torch.float64)),
        smthCorrectRatio=torch.std_mean(torch.as_tensor([x / y for x, y in zip(tmp_smthTestCorrectSum, tmp_smthPredSizeSum)], dtype=torch.float64)),

        trainTimeLoop=safeDivide(sum(tmp_trainTimeLoop), numOfAvgFiles, 0.0),
        trainTimeUnits=tmp_trainTimeUnits,
        avgTrainTimeLoop=safeDivide(safeDivide(sum(tmp_avgTrainTimeLoop), numOfAvgFiles, 0.0), tmp_avgTrainTimeLoopCount, 0.0),
        trainTotalNumb=safeDivide(sum(tmp_trainTotalNumb), numOfAvgFiles, 0.0),

        testTimeLoop=safeDivide(sum(tmp_testTimeLoop), numOfAvgFiles, 0.0),
        avgTestTimeLoop=safeDivide(safeDivide(sum(tmp_avgTestTimeLoop), numOfAvgFiles, 0.0), tmp_avgTestTimeLoopCount, 0.0),
        testTimeUnits=tmp_testTimeUnits,

        smthTestTimeLoop=safeDivide(sum(tmp_smthTestTimeLoop), numOfAvgFiles, 0.0),
        smthAvgTestTimeLoop=safeDivide(safeDivide(sum(tmp_smthAvgTestTimeLoop), numOfAvgFiles, 0.0), tmp_smthAvgTestTimeLoopCount, 0.0),
        smthTestTimeUnits=tmp_smthTestTimeUnits
    )

    newOutLogFolder = Output.createLogFolder(folderSuffix=outputFolderNameSuffix, relativeRoot=outputRelativeRootFolder)[0]
    

    # podziel
    for arrFile in flattedNewVals:
        for idx, obj in enumerate(arrFile):
            arrFile[idx] = arrFile[idx] / numOfAvgFiles

    
    # zapisz uśrednione wyniki do odpowiednich logów
    for index, ffile in enumerate(flattedFilePaths):
        with open(os.path.join(newOutLogFolder, ffile), "w") as fh:
            for obj in flattedNewVals[index]:
                fh.write(str(obj) + "\n")
        
    # zapisz konfigurację
    with open(os.path.join(newOutLogFolder, 'config.txt'), "w") as fh:
        fh.write("Used files:\n")
        for obj in config:
            fh.write(obj + "\n")

    def trySetOrNan(objList: list):
        return objList[0] if objList else "Nan"

    # zapisz średnie dokładności modelu
    with open(os.path.join(newOutLogFolder, 'model_summary.txt'), "w") as fh:
        fh.write("\nModel averaged\n")

        fh.write("Train summary:\n")
        fh.write(f" Average train time ({newStats.trainTimeUnits[0]}): {newStats.avgTrainTimeLoop[0]}\n")
        fh.write(f" Loop train time ({newStats.trainTimeUnits[0]}): {newStats.trainTimeLoop[0]}\n")
        fh.write(f" Number of batches done in total: {newStats.trainTotalNumb[0]}\n")

        fh.write("\nNormal model averaged\nTest summary:\n Average accuracy: {:>6f}% +- {:>6f}, Avg loss: {:>8f} +- {:>8f}\n".format(
            100*(newStats.correctRatio[0]), trySetOrNan(newStats.correctRatioStd),
            newStats.lossRatio[0], trySetOrNan(newStats.lossRatioStd)))
        fh.write(f" Average test execution time in a loop ({newStats.testTimeUnits[0]}): {newStats.avgTestTimeLoop[0]:>3f}\n")
        fh.write(f" Time to complete the entire loop ({newStats.testTimeUnits[0]}): {newStats.testTimeLoop[0]:>3f}\n")

        fh.write("\nSmoothed model averaged\n")
        fh.write("Test summary:\n Average accuracy: {:>6f}% +- {:>6f}, Avg loss: {:>8f} +- {:>8f}\n".format(
            100*(newStats.smthCorrectRatio[0]), trySetOrNan(newStats.smthCorrectRatioStd),
            newStats.smthLossRatio[0], trySetOrNan(newStats.smthLossRatioStd)))
        fh.write(f" Average test execution time in a loop ({newStats.smthTestTimeUnits[0]}): {newStats.smthAvgTestTimeLoop[0]:>3f}\n")
        fh.write(f" Time to complete the entire loop ({newStats.smthTestTimeUnits[0]}): {newStats.smthTestTimeLoop[0]:>3f}\n")

    newStats.logFolder = newOutLogFolder
    newStats.rootInputFolder = newOutLogFolder
    newStats.plotBatches = filePaths
    newStats.rootInputFolder = newOutLogFolder

    return newStats

def printClassToLog(metadata, *obj):
    """
        Zapisuje do ['debug:0', 'model:0'] całą klasę. Klasa ta musi przeciążać metodę __str__().
    """
    where = ['debug:0', 'model:0']
    metadata.stream.print(str(metadata), where)
    for o in obj:
        if(o is not None):
            metadata.stream.print(str(o), where)

def prettyStr(d, indent=0) -> str:
    """
        Formatuje dany obiekt według konwencji json.
    """
    return str(json.dumps(d, indent=2))

def runObjs(metadataObj, dataMetadataObj, modelMetadataObj, smoothingMetadataObj, smoothingObj, dataObj, modelObj, folderLogNameSuffix = None, 
    folderRelativeRoot = None, logData: dict=None):
    """
        Domyślna funkcja służąca do uruchomienia eksperymentu.
        Przygotowuje logi, zapisuje do logów metadane klas oraz uruchamia pętlę epok. 
    """
    metadataObj.prepareOutput()

    if(folderLogNameSuffix is not None):
        metadataObj.logFolderSuffix = folderLogNameSuffix

    metadataObj.relativeRoot = folderRelativeRoot

    printClassToLog(metadataObj, modelMetadataObj, dataObj,
        dataMetadataObj,  modelObj, smoothingObj, smoothingMetadataObj, "Other data:\n" + prettyStr(logData))

    modelTotalParams = sum(p.numel() for p in modelObj.parameters())
    metadataObj.stream.print("\nNumber of parameters of the model: {}\n".format(modelTotalParams), 'model:0')

    stats = dataObj.epochLoop(
        model=modelObj, dataMetadata=dataMetadataObj, modelMetadata=modelMetadataObj, 
        metadata=metadataObj, smoothing=smoothingObj, smoothingMetadata=smoothingMetadataObj
        )

    return stats

def modelRun(Metadata_Class, Data_Metadata_Class, Model_Metadata_Class, Smoothing_Metadata_Class, Data_Class, Model_Class, Smoothing_Class, 
    modelObj = None, load = True, save = True, folderLogNameSuffix = None, folderRelativeRoot = None):
    """
        Deprecated.
    """
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

    dictObjs[Metadata_Class.__name__].relativeRoot = folderRelativeRoot

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

        dictObjs[Smoothing_Class.__name__].__setDictionary__(smoothingMetadata=dictObjs[Smoothing_Metadata_Class.__name__], dictionary=dictObjs[Model_Class.__name__].getNNModelModule().state_dict().items())

    stat = dictObjs[Data_Class.__name__].epochLoop(
        model=dictObjs[Model_Class.__name__], dataMetadata=dictObjs[Data_Metadata_Class.__name__], modelMetadata=dictObjs[Model_Metadata_Class.__name__], 
        metadata=dictObjs[Metadata_Class.__name__], smoothing=dictObjs[Smoothing_Class.__name__], smoothingMetadata=dictObjs[Smoothing_Metadata_Class.__name__]
        )
    
    printClassToLog(dictObjs[Metadata_Class.__name__], dictObjs[Model_Metadata_Class.__name__], dictObjs[Smoothing_Metadata_Class.__name__], dictObjs[Data_Metadata_Class.__name__],
        dictObjs[Data_Class.__name__], dictObjs[Model_Class.__name__], dictObjs[Smoothing_Class.__name__])
    
    dictObjs[Metadata_Class.__name__].printEndModel()

    trySave(dictObjs=dictObjs)

    clearStat()
    return stat

#########################################
# inne funkcje
def cloneTorchDict(weights: dict, toDevice = None):
    """
        Kopiuje podane tensory do słownika oraz je zwraca. Jednocześnie można sprecyzować docelowe urządzenie.
        Zmienna weights nie musi być typu dict, wystarczy, że można po niej iterować oraz, że posiada parę (key, val).
    """
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
    """
        Przenosi wszystkie podane wagi do danego urządzenia.
    """ 
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
        Oblicza sumę wszyskich wartości bezwzględnych dostarczonych wag.
    """
    with torch.no_grad():
        sumArray = []
        for val in weights.values():
            sumArray.append(torch.sum(torch.abs(val)))
        absSum = torch.sum(torch.stack(sumArray)).item()
        return absSum

def checkStrCUDA(string):
    """
        Sprawdza, czy podany napis zaczyna się od 'cuda'.
    """
    return string.startswith('cuda')

def trySelectCUDA(device, metadata):
    """
        Zwraca 'cuda', jeżeli system je wspiera. W przeciwnym wypadku zwraca 'cpu'.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if(metadata.debugInfo):
        Output.printBash('Using {} torch CUDA device version\nCUDA avaliable: {}\nCUDA selected: {}'.format(torch.version.cuda, torch.cuda.is_available(), self.device == 'cuda'),
        'debug')
    return device

def checkForEmptyFile(filePath):
    """
        Sprawdza, czy dla podanej ścieżki plik istnieje i czy nie jest on pusty.
    """
    return os.path.isfile(filePath) and os.path.getsize(filePath) > 0

def plot(filePath: list, xlabel, ylabel, name = None, plotsNames: list = None, plotInputRoot = None, plotOutputRoot = None, 
    fileFormat = '.png', dpi = 900, widthTickFreq = 0.08, aspectRatio = 0.3, 
    startAt: float = None, endAt: float = None, highAt: float = None, lowAt: float = None, 
    startScale: float = None, endScale: float = None, highScale: float = None, lowScale: float = None, 
    resolutionInches = 11.5, fontSize = 13):
    """
        Generuje plik wykresu, który może zawierać wiele wykresów.
        Rozmiar wyjściowej grafiki jest podana wzorem [resolutionInches; resolutionInches / aspectRatio]

        filePath - ścieżki do wykresów
        name - nazwa wyjściowego pliku. Jeżeli None, to wykres zostanie pokazany, zamiast zapisany.
        plotsNames - nazwy, które pojawią się na wykresie oraz będą odpowiadały plikom z filePath. Jeżeli nie podano lub None w liście, 
            to nazwa danego wykreu będzie nazwą pliku z którego ją zaczerpnięto.

        resolutionInches - rozmiar wykresu. Jest to jego wymiar w przeliczeniu na centymetry. Jest to inna własność niż dpi.
        widthTickFreq - jak często na osi X powinny pojawiać się wartości [0; 1]
        dpi - rozdzielczość. Nie wpływa na proporcje wykresu, jedynie jego ostrość (i rozmiar pliku).
        aspectRatio - proprcje szerokość - resolutionInches/aspectRatio, wysokość - resolutionInches
        fontSize - rozmiar wszystkich czcionek na wykresie.
        fileFormat - typ wykresu [.png, .svg] itp.

        startAt - gdzie wykres na osi X ma się zacząć. Może to być pozytywna lub negatywna wartość. Musi być mniejsze od startAt != None.
        endAt - gdzie wykres na osi X ma się skończyć. Może to być pozytywna lub negatywna wartość. Musi być większe od endAt != None.
        highAt - górna granica na osi Y. Musi być większe od lowAt != None.
        lowAt - dolna granica na osi Y. Musi być mniejsze od highAt != None.

        *Scale - tego typu zmienne są przydatne, kiedy nie ustawia się zmiennyc htypu *At, a mimo to chce się
        mieć szerszy lub wyższy wykres. Przykładowo matplotlib działa tak, że wysokość wykresu daje mniej więcej 
        na wartość największej wartości w danych.

        startScale - skalowana lewa granica osi X.
        endScale - skalowana prawa granica osi X.
        highScale - skalowana górna granica osi Y.
        lowScale - skalowana dolna granica osi Y.
    """
    if(test_mode().isActive()):
        print("\nPlot parameters")
        print("filePath", filePath)
        print("name", name)
        print("filePath", plotInputRoot)
        print("plotOutputRoot", plotOutputRoot)
        print("fileFormat", fileFormat)
        print("dpi", dpi)
        print("widthTickFreq", widthTickFreq)
        print("aspectRatio", aspectRatio)
        print("startAt", startAt)
        print("endAt", endAt)
        print("resolutionInches", resolutionInches)
        print("\n")

    if(isinstance(filePath, str)):
        filePath = [filePath]

    if(len(filePath) == 0):
        Output.printBash("Could not create plot. Input files names are empty.", 'warn')
        return
    
    if(plotsNames is not None and (len(plotsNames) != len(filePath))):
        Output.printBash("Could not create plot. Input target plot names do not match file paths.", 'warn')
        return

    if(plotsNames is not None and not plotsNames):
        plotsNames = None

    if(aspectRatio <= 0):
        Output.printBash("Could not create plot. Bad aspect ratio {}".format(aspectRatio), 'warn')
        return

    if(startAt is not None and endAt is not None and startAt >= endAt):
        Output.printBash("Cannot plot any file. The startAt or endAt arguments do not follow the requirement [startAt < endAt]: \
        startAt: {}; endAt: {}.".format(startAt, endAt), 'warn')
        return

    if(lowAt is not None and highAt is not None and lowAt >= highAt):
        Output.printBash("Cannot plot any file. The lowAt or highAt arguments do not follow the requirement [lowAt < highAt]: \
        lowAt: {}; highAt: {}.".format(lowAt, highAt), 'warn')
        return
        
    def setBorder(axes, ytop, ybottom, startAt, endAt, startScale, endScale, highScale, lowScale):
        xmin = startAt
        xmax = endAt
        ymin = ybottom
        ymax = ytop

        if(startAt is not None):
            axes.set_xlim(xmin=startAt)
        if(startScale is not None):
            axes.set_xlim(xmin=axes.get_xlim()[0] * startScale)

        if(endAt is not None):
            axes.set_xlim(xmax=endAt)
        if(endScale is not None):
            axes.set_xlim(xmax=axes.get_xlim()[1] * endScale)

        if(ybottom is not None):
            axes.set_ylim(ymin=ybottom)
        if(lowScale is not None):
            axes.set_ylim(ymin=axes.get_ylim()[0] * lowScale)

        if(ytop is not None):
            axes.set_ylim(ymax=ytop)
        if(highScale is not None):
            axes.set_ylim(ymax=axes.get_ylim()[1] * highScale)

        
    fp = []
    xleft, xright = [], []
    ybottom, ytop = [], []
    dataStartAt = 0
    dataEndAt = -1

    if(startAt is not None and startAt > 0):
        dataStartAt = startAt
    if(endAt is not None and endAt < 0):
        dataEndAt = endAt

    sampleMaxSize = 0
    ax = plt.gca()
    fig = plt.gcf()
    if(plotInputRoot is not None):
        for fn in filePath:
            fp.append(os.path.join(plotInputRoot, fn))
    else:
        fp = filePath

    for idx, fn in enumerate(fp):
        if(not checkForEmptyFile(fn)):
            Output.printBash("Cannot plot file '{}'. File is empty or does not exist.".format(fn), 'warn')
            continue
        data = pd.read_csv(fn, header=None)
        if(len(data) > sampleMaxSize):
            sampleMaxSize = len(data)

        plotLabel = os.path.basename(fn)
        if(plotsNames is not None and plotsNames[idx] is not None):
            plotLabel = plotsNames[idx]
        plt.plot(data[dataStartAt:dataEndAt], label=plotLabel)

        xleft2, xright2 = ax.get_xlim()
        xleft.append(xleft2)
        xright.append(xright2)
        ybottom2, ytop2 = ax.get_ylim()
        ybottom.append(ybottom2)
        ytop.append(ytop2)

    if(not xleft or not xright or not ybottom or not ytop):
        Output.printBash("Cannot plot any file.", 'warn')
        return 

    xleft = min(xleft)
    xright = max(xright)
    ybottom = min(ybottom)
    ytop = max(ytop)
    fig.set_size_inches(resolutionInches/aspectRatio, resolutionInches)

    #aspect = abs((xright-xleft)/(ybottom-ytop))*aspectRatio
    #fig.figaspect(ratio)
    #ax.set_aspect(aspect)
    setBorder(axes=ax, ytop=highAt, ybottom=lowAt, startAt=startAt, endAt=endAt, 
        startScale = startScale, endScale = endScale, highScale = highScale, lowScale = lowScale)
    tmp = sampleMaxSize / widthTickFreq
    ax.xaxis.set_ticks(numpy.arange(ax.get_xlim()[0], ax.get_xlim()[1], (sampleMaxSize*widthTickFreq)*aspectRatio))
    plt.legend(fontsize=fontSize)
    plt.xlabel(xlabel=xlabel, fontsize=fontSize)
    plt.ylabel(ylabel=ylabel, fontsize=fontSize)
    plt.grid()


    if(name is not None):
        if(plotOutputRoot is not None):
            plt.savefig(os.path.join(plotOutputRoot, name + fileFormat), bbox_inches='tight', dpi=dpi)
        else:
            plt.savefig(name + fileFormat, bbox_inches='tight', dpi=dpi)
        plt.clf()
        return
    plt.show()

def isCuda(device):
    return device if device != 'cpu' else None

#########################################
# test   

def modelDetermTest(Metadata_Class, Data_Metadata_Class, Model_Metadata_Class, Data_Class, Model_Class, Smoothing_Class, modelObj = None):
    """
        Deprecated.
        
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

        dictObjs[Smoothing_Class.__name__].__setDictionary__(dictObjs[Model_Class.__name__].getNNModelModule().state_dict().items())

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


##############################################

if(test_mode.isActive()):
    Output.printBash("TEST MODE is enabled", 'info')