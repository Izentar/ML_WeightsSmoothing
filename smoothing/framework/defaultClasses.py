import torch
import torchvision
import torch.optim as optim
from framework import smoothingFramework as sf
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CircularList():
    def __init__(self, maxCapacity):
        self.array = []
        self.arrayIndex = 0
        self.arrayMax = maxCapacity

    def pushBack(self, number):
        if(self.arrayIndex < len(self.array)):
            del self.array[self.arrayIndex] # trzeba usunąć, inaczej insert zachowa w liście obiekt
        self.array.insert(self.arrayIndex, number)
        self.arrayIndex = (1 + self.arrayIndex) % self.arrayMax

    def getAverage(self):
        l = len(self.array)
        return sum(self.array) / l if l else 0

    def __setstate__(self):
        self.__dict__.update(state)
        self.arrayIndex = self.arrayIndex % self.arrayMax

    def reset(self):
        del self.array
        self.array = []
        self.arrayIndex = 0

    def __iter__(self):
        lo = list(range(self.arrayIndex))
        hi = list(range(self.arrayIndex, len(self.array)))
        lo.reverse()
        hi.reverse()
        self.indexArray = lo + hi
        return self

    def __next__(self):
        if(self.indexArray):
            idx = self.indexArray.pop(0)
            return self.array[idx]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.array)

class DefaultModel_Metadata(sf.Model_Metadata):
    def __init__(self):
        super().__init__()
        self.device = 'cuda:0'
        self.learning_rate = 1e-3
        self.momentum = 0.9

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Learning rate:\t{}\n'.format(self.learning_rate))
        tmp_str += ('Momentum:\t{}\n'.format(self.momentum))
        tmp_str += ('Model device :\t{}\n'.format(self.device))
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

        self.fromGrayToRGB = True

        # batch size * howOftenPrintTrain
        if(sf.test_mode.isActivated()):
            self.howOftenPrintTrain = 200
        else:
            self.howOftenPrintTrain = 2000

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Resize data from Gray to RGB:\t{}\n'.format(self.fromGrayToRGB))
        return tmp_str


class DefaultModelSimpleConv(sf.Model):
    def __init__(self, modelMetadata):
        super().__init__(modelMetadata)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(212 * 212, 212)
        self.linear2 = nn.Linear(212, 120)
        self.linear3 = nn.Linear(120, 84)
        self.linear4 = nn.Linear(84, 10)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.getNNModelModule().parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)
        #self.optimizer = optim.AdamW(self.parameters(), lr=modelMetadata.learning_rate)

        self.getNNModelModule().to(modelMetadata.device)

    def forward(self, x):
        x = self.pool(F.hardswish(self.conv1(x)))
        x = self.pool(F.hardswish(self.conv2(x)))
        # 16 * 212 * 212 może zmienić rozmiar tensora na [1, 16 * 212 * 212] co nie zgadza się z rozmiarem batch_number 1 != 16. Wtedy należy dać [-1, 212 * 212] = [16, 212 * 212]
        # ogółem ta operacja nie jest bezpieczna przy modyfikacji danych.
        x = x.view(-1, 212 * 212)   
        x = F.hardswish(self.linear1(x))
        x = F.hardswish(self.linear2(x))
        x = F.hardswish(self.linear3(x))
        x = self.linear4(x)
        return x

    def __update__(self, modelMetadata):
        self.getNNModelModule().to(modelMetadata.device)
        self.optimizer = optim.SGD(self.getNNModelModule().parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)

    def __initializeWeights__(self):
        for m in model.modules():
            if(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d))):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif(isinstance(m, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



class DefaultModelPredef(sf.PredefinedModel):
    def __init__(self, obj, modelMetadata):
        super().__init__(obj, modelMetadata)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.getNNModelModule().parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)
        #self.optimizer = optim.AdamW(self.parameters(), lr=modelMetadata.learning_rate)

        self.getNNModelModule().to(modelMetadata.device)

    def __update__(self, modelMetadata):
        self.getNNModelModule().to(modelMetadata.device)
        self.optimizer = optim.SGD(self.getNNModelModule().parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)


class DisabledSmoothing(sf.Smoothing):
    def __init__(self):
        super().__init__()

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata):
        return False

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata):
        super().__call__(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata)
        return

    def __getSmoothedWeights__(self, metadata):
        super().__getSmoothedWeights__(metadata)
        return {}

    def __setDictionary__(self, dictionary):
        return

class DefaultSmoothingBorderline(sf.Smoothing):
    """
    Włącza wygładzanie po przejściu przez określoną ilość iteracji pętli.
    Wygładzanie polega na liczeniu średnich tensorów.
    Wygładzanie włączane jest od momentu wykonania określonej ilości pętli oraz jest liczone od końca iteracji.
    """
    def __init__(self):
        super().__init__()

        self.device = 'cpu'

        if(sf.test_mode.isActivated()):
            self.numbOfBatchAfterSwitchOn = 10
        else:
            self.numbOfBatchAfterSwitchOn = 3000 # dla 50000 / 32 ~= 1500, 50000 / 16 ~= 3000

        self.sumWeights = {}
        self.previousWeights = {}
        # [torch.tensor(0.0) for x in range(100)] # add more to array than needed
        self.countWeights = 0
        self.counter = 0

        self.mainWeights = None

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata):
        False

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
                    cpuArg = arg.to(self.device)
                    self.sumWeights[key].to(self.device).add_(cpuArg)
                    #helper.substract[key] = arg.sub(self.previousWeights[key])
                    helper.substract[key] = self.previousWeights[key].sub_(cpuArg).multiply_(-1)
                    self.previousWeights[key].detach().copy_(cpuArg.detach())

    def __getSmoothedWeights__(self, metadata):
        average = super().__getSmoothedWeights__(metadata)
        if(average is not None):
            return average
        average = {}
        if(self.countWeights == 0):
            return average
        for key, arg in self.sumWeights.items():
            average[key] = self.sumWeights[key].to(self.device) / self.countWeights
        return average

    def __setDictionary__(self, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(dictionary)
        with torch.no_grad():
            for key, values in dictionary:
                self.sumWeights[key] = torch.zeros_like(values, requires_grad=False, device=self.device)
                self.previousWeights[key] = torch.zeros_like(values, requires_grad=False, device=self.device)

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

class SmoothingOscilationBase(sf.Smoothing):
    """
    Włącza wygładzanie gdy zostaną spełnione określone warunki:
    - po przekroczeniu pewnej minimalnej ilości iteracji pętli oraz 
        gdy różnica pomiędzy średnią N ostatnich strat treningowych modelu, 
        a średnią średnich N ostatnich strat treningowych modelu jest mniejsza niż epsilon.
    - po przekroczeniu pewnej maksymalnej ilości iteracji pętli.

    Liczy średnią arytmetyczną dla wag.
    """
    def __init__(self):
        super().__init__()

        #------- dane do konfiguracji

        self.device = 'cpu'

        # jak częto powinno się zapisywać średnią odnoszącą się do akumulatora straty.
        # ta średnia jest później porównywana ze średnią N ostatnich zapisanych strat, 
        # która mówi o tym czy nie znaleziono otoczenia minimum lokalnego
        self.avgOfAvgUpdateFreq = 10 

        # jak często powinno się sprawdzać, czy wygładzanie jest dostatecznie dobre. Min val = 1
        self.endSmoothingFreq = 10

        # margines błędu mówiący ile razy __isSmoothingGoodEnough__ powinno dawać pozytywną informację, aznim można
        # będzie uznać, że wygładzanie jest dostatecznie dobre. Funkcjonuje jako pojemność akumulatora.
        self.softMarginAdditionalLoops = 20

        if(sf.test_mode.isActivated()):
            # po ilu maksymalnie iteracjach wygładzanie powinno zostać włączone
            self.numbOfBatchMaxStart = sf.StaticData.MAX_TEST_LOOPS - 10 if sf.StaticData.MAX_TEST_LOOPS - 10 >= 0 else 0

            # minimalna ilość iteracji po których wygładzanie może zostać włączone
            # zapobiega problemom: mała ilośc zgromadzonych danych => niemiarodajne wyniki dla średnich =? zbyt wczesne zatrzymanie treningu
            self.numbOfBatchMinStart = sf.StaticData.MAX_TEST_LOOPS - 30 if sf.StaticData.MAX_TEST_LOOPS - 30 >= 0 else 0
            self.epsilon = 1e-2 # kiedy można uzać, że wygładzania powinno zostać włączone 
            self.weightsEpsilon = 1e-4 # kiedy można uznać, że wygładzanie powinno zostać zakończone wraz z zakończeniem pętli treningowej.
        else:
            self.numbOfBatchMaxStart = 3100
            self.numbOfBatchMinStart = 500
            self.epsilon = 1e-3
            self.weightsEpsilon = 1e-6
        ###############################

        self.lossContainer = CircularList(50)
        self.lastKLossAverage = CircularList(40)
        
        self.countWeights = 0
        self.counter = 0
        self.tensorPrevSum_1 = CircularList(int(self.endSmoothingFreq))
        self.tensorPrevSum_2 = CircularList(int(self.endSmoothingFreq))
        self.divisionCounter = 0
        self.goodEnoughCounter = 0

        self.mainWeights = None

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Update frequency of average of average:\t{}\n'.format(self.avgOfAvgUpdateFreq))
        tmp_str += ('Frequency of checking to end smoothing:\t{}\n'.format(self.endSmoothingFreq))
        tmp_str += ('Max loops to start:\t{}\n'.format(self.numbOfBatchMaxStart))
        tmp_str += ('Min loops to start:\t{}\n'.format(self.numbOfBatchMinStart))
        tmp_str += ('Epsilon to check when start compute weights:\t{}\n'.format(self.epsilon))
        tmp_str += ('Epsilon to check when weights are good enough to stop:\t{}\n'.format(self.weightsEpsilon))
        tmp_str += ('Loop soft margin:\t{}\n'.format(self.softMarginAdditionalLoops))
        return tmp_str

    def canComputeWeights(self):
        """
        - Jeżeli wartość bezwzględna różnicy średnich N ostatnich strat f. celu, a średnią K średnich N ostatnich strat f. celu będzie mniejsza niż epsilon
        i program przeszedł przez minimalną liczbę pętli
        - lub program wykonał już pewną graniczną ilość pętli
        to metoda zwróci True.
        W przeciwnym wypadku zwróci False.
        """
        if(self.counter % self.avgOfAvgUpdateFreq == int(self.avgOfAvgUpdateFreq / 2)): # ponieważ liczenie kolejnej średniej od razu po dodaniu nowego elementu dużo nie da, lepiej zrobić to w połowie
            return bool(
                (abs(self.lossContainer.getAverage() - self.lastKLossAverage.getAverage()) < self.epsilon and self.counter > self.numbOfBatchMinStart) 
                or self.counter > self.numbOfBatchMaxStart
            )
        return False

    def _saveAvgSum(self, sum):
        # avg_1 będzie zawierała ogółem starsze wartości niż avg_2, 
        # dlatego przy dodaniu wartości do avg_2 różnica będzie mniejsza niż przy dodaniu wartości do avg_1
        if(self.divisionCounter % 2 == 0):
            self.tensorPrevSum_1.pushBack(sum)
        else:
            self.tensorPrevSum_2.pushBack(sum)
        return self.tensorPrevSum_1.getAverage(), self.tensorPrevSum_2.getAverage()

    def _sumAllWeights(self, metadata):
        smWg = self.__getSmoothedWeights__(metadata)
        sumArray = []
        for val in smWg.values():
            sumArray.append(torch.sum(torch.abs(val)))
        absSum = torch.sum(torch.stack(sumArray)).item()
        return smWg

    def _smoothingGoodEnoughCheck(self, avg_1, avg_2):
        ret = bool(abs(avg_1 - avg_2) < self.weightsEpsilon)
        if(ret):
            if(self.softMarginAdditionalLoops > self.goodEnoughCounter):
                self.goodEnoughCounter += 1
                return False
            return ret

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata):
        """
        Sprawdza jak bardzo wygładzone wagi różnią się od siebie w kolejnych iteracjach.
        Jeżeli różnica będzie wystarczająco mała, metoda zwróci True. W przeciwnym wypadku zwraca false.

        Algorytm:
        - sumowanie wszystkich wag dla których wcześniej oblicza się wartość bezwzględną
        - dodanie obliczonej wartości do jednego z 2 buforów cyklicznych ustawionych w kolejce, jeden za drugim
        - obliczenie oddzielnie średniej w dwóch buforach i wykonanie abs(r1 - r2)
        - jeżeli dana różnica bezwzględna jest mniejsza od weightsEpsilon, to zwraca True
        - w przeciwnym wypadku metoda zwraca False
        """
        if(self.countWeights > 0 and self.counter % self.endSmoothingFreq == 0):
            self.divisionCounter += 1
            smWg = self._sumAllWeights(metadata)

            avg_1, avg_2 = self._saveAvgSum(absSum)

            metadata.stream.print("Sum debug:" + str(absSum), 'debug:0')
            metadata.stream.print("Weight avg debug:" + str(abs(avg_1 - avg_2)), 'debug:0')
            metadata.stream.print("Weight avg bool:" + str(bool(abs(avg_1 - avg_2) < self.weightsEpsilon)), 'debug:0')
            
            return self._smoothingGoodEnoughCheck(avg_1, avg_2)
        return False

class DefaultSmoothingOscilationGeneralizedMean(SmoothingOscilationBase):
    """
    Włącza wygładzanie gdy zostaną spełnione określone warunki:
    - po przekroczeniu pewnej minimalnej ilości iteracji pętli oraz 
        gdy różnica pomiędzy średnią N ostatnich strat treningowych modelu, 
        a średnią średnich N ostatnich strat treningowych modelu jest mniejsza niż epsilon.
    - po przekroczeniu pewnej maksymalnej ilości iteracji pętli.

    Liczy średnią generalizowaną dla wag.
    """
    def __init__(self):
        super().__init__()

        # dane do konfiguracji

        # stopieć średniej generalizowanej. Dla = 1 jest to średnia arytmetyczna
        self.generalizedMeanPower = 1 # tylko dla 1 dobrze działa, reszta daje gorsze wyniki, info do opisania
        ###############################

        self.sumWeights = {}
        self.lossContainer = CircularList(50)
        self.lastKLossAverage = CircularList(40)

        self.methodPow = None
        self.methodDiv = None

        if(self.generalizedMeanPower == 1):
            self.methodPow = self.methodPow_1
            self.methodDiv = self.methodDiv_1
        else:
            self.methodPow = self.methodPow_
            self.methodDiv = self.methodDiv_

    def methodPow_(self, key, arg):
        self.sumWeights[key].add_(arg.to(self.device).pow(self.generalizedMeanPower))

    def methodPow_1(self, key, arg):
        self.sumWeights[key].add_(arg.to(self.device))

    def methodDiv_(self, arg):
        return (arg / self.countWeights).pow(1/self.generalizedMeanPower)

    def methodDiv_1(self, arg):
        return (arg / self.countWeights)

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Generalized mean power:\t{}\n'.format(self.generalizedMeanPower))
        return tmp_str

    def calcAvgArithmeticMean(self, model):
        with torch.no_grad():
            for key, arg in model.getNNModelModule().named_parameters():
                self.methodPow(key, arg)

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata):
        super().__call__(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata)
        self.counter += 1
        self.lossContainer.pushBack(helper.loss.item())
        avg = self.lossContainer.getAverage()
        if(self.counter % self.avgOfAvgUpdateFreq):
            self.lastKLossAverage.pushBack(avg)
        metadata.stream.print("Loss avg debug:" + str(abs(avg - self.lastKLossAverage.getAverage())), 'debug:0')
        if(self.canComputeWeights()):
            self.countWeights += 1
            self.calcAvgArithmeticMean(model)

    def __getSmoothedWeights__(self, metadata):
        average = super().__getSmoothedWeights__(metadata)
        if(average is not None):
            return average
        average = {}
        if(self.countWeights == 0):
            return average
        for key, arg in self.sumWeights.items():
            average[key] = self.methodDiv(arg)
        return average

    def __setDictionary__(self, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(dictionary)
        with torch.no_grad():
            for key, values in dictionary:
                self.sumWeights[key] = torch.zeros_like(values, requires_grad=False, device=self.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['countWeights']
            del state['counter']
            del state['mainWeights']
            del state['sumWeights']
            del state['enabled']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.countWeights = 0
            self.counter = 0
            self.mainWeights = None
            self.sumWeights = {}
            self.enabled = False
            self.lossContainer.reset()
            self.lastKLossAverage.reset()
            self.tensorPrevSum_1.reset()
            self.tensorPrevSum_2.reset()
            self.divisionCounter = 0

class DefaultSmoothingOscilationMovingMean(SmoothingOscilationBase):
    """
    Włącza wygładzanie gdy zostaną spełnione określone warunki:
    - po przekroczeniu pewnej minimalnej ilości iteracji pętli oraz 
        gdy różnica pomiędzy średnią N ostatnich strat treningowych modelu, 
        a średnią średnich N ostatnich strat treningowych modelu jest mniejsza niż epsilon.
    - po przekroczeniu pewnej maksymalnej ilości iteracji pętli.

    Liczy średnią ważoną dla wag. Wagi są nadawane względem starości zapamiętanej wagi. Im starsza tym ma mniejszą wagę.
    """
    def __init__(self):
        super().__init__()

        # dane do konfiguracji

        # jest to parametr 'a' dla wzoru: S = ax + (1-a)S
        movingAvgParam = 0.27

        ###############################

        # trzeba uważać na zajętość w pamięci. Zapamiętuje wszystkie wagi modelu, co może sumować się do dużych rozmiarów
        #self.weightsArray = CircularList(20) 
        self.weightsSum = {}
        self.lossContainer = CircularList(50)
        self.lastKLossAverage = CircularList(40)

        self.setMovingAvgParam(movingAvgParam)

    def setMovingAvgParam(self, value):
        if(value >= 1.0 or value <= 0.0):
            raise Exception("Value of {}.movingAvgParam can only be in the range [0; 1]".format(self.__name__))

        self.movingAvgTensorLow = torch.tensor(value).to(self.device)
        self.movingAvgTensorHigh = torch.tensor(1 - value).to(self.device)

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Moving average parameter:\t{}\n'.format(self.movingAvgTensorLow.item()))
        return tmp_str

    def calcAvgWeightedMean(self, model):
        with torch.no_grad():
            for key, val in model.getNNModelModule().named_parameters():
                self.weightsSum[key] = self.weightsSum[key].mul_(self.movingAvgTensorHigh).add_(val.to(self.device).mul_(self.movingAvgTensorLow))

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata):
        super().__call__(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata)
        self.counter += 1
        self.lossContainer.pushBack(helper.loss.item())
        avg = self.lossContainer.getAverage()
        if(self.counter % self.avgOfAvgUpdateFreq):
            self.lastKLossAverage.pushBack(avg)
        metadata.stream.print("Loss avg debug:" + str(abs(avg - self.lastKLossAverage.getAverage())), 'debug:0')
        if(self.canComputeWeights()):
            self.countWeights += 1
            self.calcAvgWeightedMean(model)

    def __getSmoothedWeights__(self, metadata):
        average = super().__getSmoothedWeights__(metadata)
        if(average is not None):
            return average # {}
        average = {}
        if(self.countWeights == 0):
            return average # {}

        return sf.cloneTorchDict(self.weightsSum)

    def __setDictionary__(self, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(dictionary)
        with torch.no_grad():
            for key, values in dictionary:
                # ważne jest, aby skopiować początkowe wagi, a nie stworzyć tensor zeros_like
                # w przeciwnym wypadku średnia będzie dawała bardzo złe wyniki
                self.weightsSum[key] = torch.clone(values).to(self.device).requires_grad_(False) 

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['countWeights']
            del state['counter']
            del state['mainWeights']
            del state['enabled']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.countWeights = 0
            self.counter = 0
            self.mainWeights = None
            self.enabled = False
            self.lossContainer.reset()
            self.lastKLossAverage.reset()
            self.tensorPrevSum_1.reset()
            self.tensorPrevSum_2.reset()
            self.divisionCounter = 0

class DefaultSmoothingOscilationWeightedMean(SmoothingOscilationBase):
    """
    Włącza wygładzanie gdy zostaną spełnione określone warunki:
    - po przekroczeniu pewnej minimalnej ilości iteracji pętli oraz 
        gdy różnica pomiędzy średnią N ostatnich strat treningowych modelu, 
        a średnią średnich N ostatnich strat treningowych modelu jest mniejsza niż epsilon.
    - po przekroczeniu pewnej maksymalnej ilości iteracji pętli.

    Liczy średnią ważoną dla wag. Wagi są nadawane względem starości zapamiętanej wagi. Im starsza tym ma mniejszą wagę.
    """
    def __init__(self):
        super().__init__()

        # dane do konfiguracji

        # jak bardzo następne wagi w kolejce mają stracić na wartości. Kolejne wagi dzieli się przez wielokrotność tej wartości.
        self.weightDecay = 1.5

        ###############################

        # trzeba uważać na zajętość w pamięci. Zapamiętuje wszystkie wagi modelu, co może sumować się do dużych rozmiarów
        self.weightsArray = CircularList(20) 
        self.lossContainer = CircularList(50)
        self.lastKLossAverage = CircularList(40)

    def __strAppend__(self):
        return super().__strAppend__()

    def calcAvgWeightedMean(self, model):
        with torch.no_grad():
            tmpDict = {}
            for key, val in model.getNNModelModule().named_parameters():
                tmpDict[key] = val.clone().to(self.device)
            self.weightsArray.pushBack(tmpDict)

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata):
        super().__call__(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata)
        self.counter += 1
        self.lossContainer.pushBack(helper.loss.item())
        avg = self.lossContainer.getAverage()
        if(self.counter % self.avgOfAvgUpdateFreq):
            self.lastKLossAverage.pushBack(avg)
        metadata.stream.print("Loss avg debug:" + str(abs(avg - self.lastKLossAverage.getAverage())), 'debug:0')
        if(self.canComputeWeights()):
            self.countWeights += 1
            self.calcAvgWeightedMean(model)

    def __getSmoothedWeights__(self, metadata):
        average = super().__getSmoothedWeights__(metadata)
        if(average is not None):
            return average # {}
        average = {}
        if(self.countWeights == 0):
            return average # {}


        for wg in self.weightsArray:
            for key, val in wg.items():
                average[key] = torch.zeros_like(val, requires_grad=False, device=self.device)
            break

        weight = 1.0
        wgSum = 0.0
        for wg in self.weightsArray:
            for key, val in wg.items(): 
                average[key] += val.mul(weight)
            wgSum += weight 
            weight /= self.weightDecay

        for key, val in average.items():
            val.div_(wgSum)
        return average

    def __setDictionary__(self, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(dictionary)

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['countWeights']
            del state['counter']
            del state['mainWeights']
            del state['weightsArray']
            del state['enabled']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.countWeights = 0
            self.counter = 0
            self.mainWeights = None
            self.weightsArray.reset()
            self.enabled = False
            self.lossContainer.reset()
            self.lastKLossAverage.reset()
            self.tensorPrevSum_1.reset()
            self.tensorPrevSum_2.reset()
            self.divisionCounter = 0



class DefaultData(sf.Data):
    def __init__(self):
        super().__init__()
        self.statLogName = 'statLossTest'

    def __customizeState__(self, state):
        super().__customizeState__(state)

    def __setstate__(self, state):
        super().__setstate__(state)

    def lambdaGrayToRGB(x):
        return x.repeat(3, 1, 1)

    def NoneTransform(obj):
        '''Nic nie robi. Używana w wyrażeniach warunkowych dotyczących tranformacji lambda.'''
        return obj

    def __setInputTransform__(self, dataMetadata):
        ''' self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        '''

        self.trainTransform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(DefaultData.lambdaGrayToRGB if dataMetadata.fromGrayToRGB else DefaultData.NoneTransform),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
        ])
        self.testTransform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Lambda(DefaultData.lambdaGrayToRGB if dataMetadata.fromGrayToRGB else DefaultData.NoneTransform),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __prepare__(self, dataMetadata):
        raise NotImplementedError("def __prepare__(self, dataMetadata)")

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

            metadata.stream.print(helper.loss.item(), ['statLossTrain'])

            if(bool(metadata.debugInfo) and dataMetadata.howOftenPrintTrain is not None and (helper.batchNumber % dataMetadata.howOftenPrintTrain == 0 or sf.test_mode.isActivated())):
                sf.DefaultMethods.printLoss(metadata, helper)                
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
        #metadata.stream.print('\n\nTest loss', ['statLossTest'])

    def __beforeTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        helper.inputs = helper.inputs.to(modelMetadata.device)
        helper.labels = helper.labels.to(modelMetadata.device)

    def __afterTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        super().__afterTest__(helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothing)
        metadata.stream.print(helper.test_loss, self.statLogName)

    def __afterTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        lossRatio = helper.testLossSum / helper.predSizeSum
        correctRatio = helper.testCorrectSum / helper.predSizeSum
        metadata.stream.print(f"\nTest summary: \n Accuracy: {(100*correctRatio):>0.1f}%, Avg loss: {lossRatio:>8f}", ['model:0'])
        metadata.stream.print(f" Average test execution time in a loop ({helper.timer.getUnits()}): {helper.timer.getAverage():>3f}", ['model:0'])
        metadata.stream.print(f" Time to complete the entire loop ({helper.timer.getUnits()}): {helper.loopTimer.getDiff():>3f}\n", ['model:0'])
        metadata.stream.print(f"{helper.timer.getAverage()};{helper.loopTimer.getDiff()};{(correctRatio):>0.0001f};{lossRatio:>8f}", ['stat'])

    def __beforeEpochLoop__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing'):
        metadata.stream.open(outputType="formatedLog", alias='statLossTrain', pathName='statLossTrain')
        metadata.stream.open(outputType="formatedLog", alias='statLossTest', pathName='statLossTest')
        metadata.stream.open(outputType="formatedLog", alias='statLossTestSmoothing', pathName='statLossTestSmoothing')

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
                wg = smoothing.__getSmoothedWeights__(metadata)
                if(wg):
                    #metadata.stream.print("Smoothing:", 'statLossTest')
                    model.setWeights(wg)
                    #metadata.stream.write("Smoothing weights, ")
                    #metadata.stream.write("Smoothing weights;", 'stat')
                    self.statLogName = 'statLossTestSmoothing'
                    self.testLoop(helperEpoch, model, dataMetadata, modelMetadata, metadata, smoothing)
                    model.setWeights(smoothing.getWeights('main'))
                else:
                    sf.Output.printBash('Smoothing is not enabled. Test does not executed.', 'info')
            # model.linear1.weight = torch.nn.parameter.Parameter(model.average)
            # model.linear1.weight = model.average

class DefaultDataMNIST(DefaultData):
    def __init__(self):
        super().__init__()

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__(dataMetadata)

        #self.trainset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform)
        #self.testset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform)
        self.trainset = torchvision.datasets.MNIST(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform, download=dataMetadata.download)
        self.testset = torchvision.datasets.MNIST(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform, download=dataMetadata.download)

        self.trainSampler = sf.BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = sf.BaseSampler(len(self.testset), dataMetadata.batchTrainSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

class DefaultDataEMNIST(DefaultData):
    def __init__(self):
        super().__init__()

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__(dataMetadata)

        #self.trainset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform)
        #self.testset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform)
        self.trainset = torchvision.datasets.EMNIST(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform, split='digits', download=dataMetadata.download)
        self.testset = torchvision.datasets.EMNIST(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform, split='digits', download=dataMetadata.download)

        self.trainSampler = sf.BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = sf.BaseSampler(len(self.testset), dataMetadata.batchTrainSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

class DefaultDataCIFAR10(DefaultData):
    def __init__(self):
        super().__init__()

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__(dataMetadata)

        #self.trainset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform)
        #self.testset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform)
        self.trainset = torchvision.datasets.CIFAR10(root, split)(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform, download=dataMetadata.download)
        self.testset = torchvision.datasets.CIFAR10(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform, download=dataMetadata.download)

        self.trainSampler = sf.BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = sf.BaseSampler(len(self.testset), dataMetadata.batchTrainSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

class DefaultDataCIFAR100(DefaultData):
    def __init__(self):
        super().__init__()

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__(dataMetadata)

        #self.trainset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform)
        #self.testset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform)
        self.trainset = torchvision.datasets.CIFAR100(root, split)(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform, download=dataMetadata.download)
        self.testset = torchvision.datasets.CIFAR100(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform, download=dataMetadata.download)

        self.trainSampler = sf.BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = sf.BaseSampler(len(self.testset), dataMetadata.batchTrainSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)






if(__name__ == '__main__'):
    torch.backends.cudnn.benchmark = True
    obj = models.alexnet(pretrained=True)

    #sf.useDeterministic()
    #sf.modelDetermTest(sf.Metadata, DefaultData_Metadata, DefaultModel_Metadata, DefaultData, VGG16Model, DefaultSmoothingBorderline)
    stat = sf.modelRun(sf.Metadata, DefaultData_Metadata, DefaultModel_Metadata, DefaultDataMNIST, DefaultModel, DefaultSmoothingBorderline, obj)

    #plt.plot(stat.trainLossArray)
    #plt.xlabel('Train index')
    #plt.ylabel('Loss')
    #plt.show()




