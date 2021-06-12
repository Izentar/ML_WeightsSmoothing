import torch
import torchvision
import torch.optim as optim
from framework import smoothingFramework as sf
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os

import copy

class ConfigClass():
    STD_NAN = 1e+10


class CircularList():
    class CircularListIter():
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

    def pushBack(self, number):
        if(self.arrayIndex < len(self.array)):
            del self.array[self.arrayIndex] # trzeba usunąć, inaczej insert zachowa w liście obiekt
        self.array.insert(self.arrayIndex, number)
        self.arrayIndex = (1 + self.arrayIndex) % self.arrayMax

    def getAverage(self, startAt=0):
        """
            Zwraca średnią.
            Argument startAt mówi o tym, od którego momentu w kolejce należy liczyć średnią.

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

    def __setstate__(self):
        self.__dict__.update(state)
        self.arrayIndex = self.arrayIndex % self.arrayMax

    def reset(self):
        del self.array
        self.array = []
        self.arrayIndex = 0

    def __iter__(self):
        return CircularList.CircularListIter(self)

    def __len__(self):
        return len(self.array)

class DefaultWeightDecay():
    def __init__(self, weightDecay = 1.1):
        self.weightDecay = weightDecay

    def __iter__(self):
        self.currentWd = 1.0
        return self

    def __next__(self):
        x = self.currentWd
        self.currentWd /= self.weightDecay
        return x

    def __str__(self):
        tmp_str = ('Weight decay:\t{}\n'.format(self.weightDecay))
        return tmp_str


# model classes
class DefaultModel_Metadata(sf.Model_Metadata):
    def __init__(self,
        device = 'cuda:0', learning_rate = 1e-3, momentum = 0.9):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Learning rate:\t{}\n'.format(self.learning_rate))
        tmp_str += ('Momentum:\t{}\n'.format(self.momentum))
        tmp_str += ('Model device :\t{}\n'.format(self.device))
        return tmp_str


class DefaultModelSimpleConv(sf.Model):
    """
    Z powodu jego prostoty i słabych wyników zaleca się go używać podczas testowania sieci.
    """
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
        self.__initializeWeights__()

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
        for m in self.modules():
            if(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d))):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif(isinstance(m, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def createDefaultMetadataObj(self):
        return DefaultModel_Metadata()

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

    def createDefaultMetadataObj(self):
        return None

# disabled smoothing
class DisabledSmoothing_Metadata(sf.Smoothing_Metadata):
    def __init__(self):
        super().__init__()

class DisabledSmoothing(sf.Smoothing):
    def __init__(self, smoothingMetadata):
        super().__init__(smoothingMetadata=smoothingMetadata)

        if(not isinstance(smoothingMetadata, DisabledSmoothing_Metadata)):
            raise Exception("Metadata class '{}' is not the type of '{}'".format(type(smoothingMetadata), DisabledSmoothing_Metadata.__name__))
        
    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        return False

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        super().__call__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothingMetadata=smoothingMetadata)
        return False

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        super().__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        return {}

    def __setDictionary__(self, smoothingMetadata, dictionary):
        super().__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=dictionary)
        return

    def createDefaultMetadataObj(self):
        return DisabledSmoothing_Metadata()

# oscilation base
class _SmoothingOscilationBase_Metadata(sf.Smoothing_Metadata):
    def __init__(self, 
        device = 'cpu',
        weightSumContainerSize = 10, weightSumContainerSizeStartAt=5, softMarginAdditionalLoops = 20, 
        batchPercentMaxStart = 0.9988, batchPercentMinStart = 0.02, 
        epsilon = 1e-6, hardEpsilon=1e-8, weightsEpsilon = 1e-7,
        lossContainer=50, lossContainerDelayedStartAt = 25):
        """
            device - urządzenie na którym ma działać wygładzanie
            weightSumContainerSize - wielkość kontenera dla przechowywania sumy wag.
            softMarginAdditionalLoops - margines błędu mówiący ile razy __isSmoothingGoodEnough__ powinno dawać pozytywną informację, 
                zanim można będzie uznać, że wygładzanie jest dostatecznie dobre. Funkcjonuje jako pojemność akumulatora.
            lossContainerSize - rozmiar kontenera, który trzyma N ostatnich strat modelu.
            lossContainerDelayedStartAt - dla jak bardzo starych wartości powinno się policzyć średnią przy 
                porównaniu ze średnią z całego kontenera strat.
            batchPercentMaxStart - po ilu maksymalnie iteracjach wygładzanie powinno zostać włączone. Wartość w przedziale [0; 1], gdzie
                większa wskazuje na większą liczbę wykonanych iteracji.
            batchPercentMinStart - minimalna ilość iteracji po których wygładzanie może zostać włączone. Wartość w przedziale [0; 1], gdzie
                większa wskazuje na większą liczbę wykonanych iteracji.
            epsilon - kiedy można uzać, że wygładzania powinno zostać włączone dla danej iteracji. Jeżeli warunek nie zostanie
                spełniony, to możliwe jest, że wygładzanie nie uzwględni nowych wag.
            hardEpsilon - kiedy wygładzanie powinno zostać włączone pernamentnie, niezależnie od parametru 'epsilon'.
            weightsEpsilon - kiedy można uznać, że wygładzanie powinno zostać zakończone wraz z zakończeniem pętli treningowej.

            Wartości z dopiskiem test_ są uzywane tylko, gdy włączony jest tryb testowy. 
            Stworzone jest to po to, aby w łatwy sposób móc przetestować klasę, bez wcześniejszego definiowania od nowa wartości.
        """

        super().__init__()

        self.device = device
        self.weightSumContainerSize = weightSumContainerSize
        self.softMarginAdditionalLoops = softMarginAdditionalLoops
        self.lossContainerSize = lossContainer
        self.lossContainerDelayedStartAt = lossContainerDelayedStartAt
        self.batchPercentMaxStart = batchPercentMaxStart
        self.batchPercentMinStart = batchPercentMinStart
        self.epsilon = epsilon
        self.hardEpsilon = hardEpsilon 
        self.weightsEpsilon = weightsEpsilon
        self.weightSumContainerSizeStartAt = weightSumContainerSizeStartAt

        # data validation
        if(self.weightSumContainerSize <= weightSumContainerSizeStartAt):
            raise Exception("lossContainerDelayedStartAt cannot be greater than weightSumContainerSize.\nlossContainerDelayedStartAt: {}\nweightSumContainerSize: {}".format(
                self.lossContainerDelayedStartAt, self.weightSumContainerSize))
        if(self.hardEpsilon > self.epsilon):
            raise Exception("Hard epsilon cannot be greater than epsilon.\nhard epsilon: {}\nepsilon: {}".format(
                self.hardEpsilon, self.epsilon))
    
    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Size of the weight sum container:\t{}\n'.format(self.weightSumContainerSize))
        tmp_str += ('Weight sum container delayed start:\t{}\n'.format(self.weightSumContainerSizeStartAt))
        tmp_str += ('Max loops to start (%):\t{}\n'.format(self.batchPercentMaxStart))
        tmp_str += ('Min loops to start (%):\t{}\n'.format(self.batchPercentMinStart))
        tmp_str += ('Epsilon to check when to start compute weights:\t{}\n'.format(self.epsilon))
        tmp_str += ('Hard epsilon to check to when start compute weights:\t{}\n'.format(self.hardEpsilon))
        tmp_str += ('Epsilon to check when weights are good enough to stop:\t{}\n'.format(self.weightsEpsilon))
        tmp_str += ('Loop soft margin:\t{}\n'.format(self.softMarginAdditionalLoops))
        tmp_str += ('Loss container size:\t{}\n'.format(self.lossContainerSize))
        tmp_str += ('Loss container delayed start:\t{}\n'.format(self.lossContainerDelayedStartAt))
        tmp_str += ('Device:\t{}\n'.format(self.device))
        return tmp_str

class _Test_SmoothingOscilationBase_Metadata(_SmoothingOscilationBase_Metadata):
    def __init__(self,
        test_device = 'cpu',
        test_weightSumContainerSize = 10, test_weightSumContainerSizeStartAt=5, test_softMarginAdditionalLoops = 3, 
        test_batchPercentMaxStart = 0.85, test_batchPercentMinStart = 0.1, 
        test_epsilon = 1e-4, test_hardEpsilon=1e-9, test_weightsEpsilon = 1e-5,
        test_lossContainer=5, test_lossContainerDelayedStartAt = 2):
        """
            Klasa z domyślnymi testowymi parametrami.
        """

        super().__init__(device=device,
        weightSumContainerSize=test_weightSumContainerSize, weightSumContainerSizeStartAt=test_weightSumContainerSizeStartAt, 
        softMarginAdditionalLoops=test_softMarginAdditionalLoops, batchPercentMaxStart=test_batchPercentMaxStart,
        batchPercentMinStart=test_batchPercentMinStart, epsilon=test_epsilon, hardEpsilon=test_hardEpsilon, weightsEpsilon=test_weightsEpsilon,
        lossContainer=test_lossContainer, lossContainerDelayedStartAt=test_lossContainerDelayedStartAt)

class _SmoothingOscilationBase(sf.Smoothing):
    """
    Włącza wygładzanie gdy zostaną spełnione określone warunki:
    - po przekroczeniu pewnej minimalnej ilości iteracji pętli oraz 
        gdy różnica pomiędzy średnią N ostatnich strat treningowych modelu, 
        a średnią średnich N ostatnich strat treningowych modelu jest mniejsza niż epsilon.
    - po przekroczeniu pewnej maksymalnej ilości iteracji pętli.

    Liczy średnią arytmetyczną dla wag.
    """
    def __init__(self, smoothingMetadata):
        super().__init__(smoothingMetadata=smoothingMetadata)

        if(not isinstance(smoothingMetadata, _SmoothingOscilationBase_Metadata)):
            raise Exception("Metadata class '{}' is not the type of '{}'".format(type(smoothingMetadata), _SmoothingOscilationBase_Metadata.__name__))
        
        self.countWeights = 0
        self.tensorPrevSum = CircularList(int(smoothingMetadata.weightSumContainerSize))
        self.divisionCounter = 0
        self.goodEnoughCounter = 0
        self.alwaysOn = False

        self.lossContainer = CircularList(smoothingMetadata.lossContainerSize)
        
    def canComputeWeights(self, helper, helperEpoch, dataMetadata, smoothingMetadata, metadata):
        """
        - Jeżeli wartość bezwzględna różnicy średnich N ostatnich strat f. celu, a średnią K średnich N ostatnich strat f. celu będzie mniejsza niż epsilon
        i program przeszedł przez minimalną liczbę pętli, to metoda zwróci True.
        W przeciwnym wypadku zwróci False.
        """
        avg_1 = self.lossContainer.getAverage()
        avg_2 = self.lossContainer.getAverage(smoothingMetadata.lossContainerDelayedStartAt)
        metadata.stream.print("Loss average: {} : {}".format(avg_1, avg_2), 'debug:0')
        absAvgDiff = abs(avg_1 - avg_2)
        minStart = smoothingMetadata.batchPercentMinStart * helperEpoch.maxTrainTotalNumber

        if(absAvgDiff < smoothingMetadata.hardEpsilon and helperEpoch.trainTotalNumber > minStart):
            self.alwaysOn = True
            metadata.stream.print("Reached hard epsilon.", 'debug:0')
        return bool(
            (
                # jeżeli osiągnięto dostatecznie małą różnicę średnich strat oraz wykonano minimalną liczbę pętli
                absAvgDiff < smoothingMetadata.epsilon and helperEpoch.trainTotalNumber >= minStart
            )
            or (
                # wykonano maksymalną liczbę pętli przed włączeniem wygładzania
                helperEpoch.trainTotalNumber > (smoothingMetadata.batchPercentMaxStart * helperEpoch.maxTrainTotalNumber)
            )
        )

    def _sumAllWeights(self, smoothingMetadata, metadata):
        smWg = self.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        return sf.sumAllWeights(smWg)

    def _smoothingGoodEnoughCheck(self, val, smoothingMetadata):
        ret = bool(val < smoothingMetadata.weightsEpsilon)
        if(ret):
            if(smoothingMetadata.softMarginAdditionalLoops > self.goodEnoughCounter):
                self.goodEnoughCounter += 1
                return False
            return ret
        return False

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        """
        Sprawdza jak bardzo wygładzone wagi różnią się od siebie w kolejnych iteracjach.
        Jeżeli różnica będzie wystarczająco mała, metoda zwróci True. W przeciwnym wypadku zwraca false.

        Algorytm:
        - sumowanie wszystkich wag dla których wcześniej oblicza się wartość bezwzględną
        - dodanie obliczonej wartości do bufora cyklicznego
        - obliczenie średniej z bufora cyklicznego
        - obliczenie drugiej średniej z bufora cyklicznego, dla którgo obliczanie średniej zaczyna się począwszy od K ostatniej wagi
        - podanie obliczonej bezwzględnej różnicy do ewaluacji
        """
        if(self.countWeights > 0):
            self.divisionCounter += 1
            absSum = self._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=metadata)

            self.tensorPrevSum.pushBack(absSum)
            avg_1, avg_2 = self.tensorPrevSum.getAverage(), self.tensorPrevSum.getAverage(smoothingMetadata.weightSumContainerSizeStartAt)

            metadata.stream.print("Sum debug:" + str(absSum), 'debug:0')
            metadata.stream.print("Weight avg diff: " + str(abs(avg_1 - avg_2)), 'debug:0')
            metadata.stream.print("Weight avg diff bool: " + str(bool(abs(avg_1 - avg_2) < smoothingMetadata.weightsEpsilon)), 'debug:0')
            
            return self._smoothingGoodEnoughCheck(val=abs(avg_1 - avg_2), smoothingMetadata=smoothingMetadata)
        return False

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        super().__call__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothingMetadata=smoothingMetadata)
        self.lossContainer.pushBack(helper.loss.item())
        metadata.stream.print("Loss avg diff : " + 
            str(abs(self.lossContainer.getAverage() - self.lossContainer.getAverage(smoothingMetadata.lossContainerDelayedStartAt))), 'debug:0')

        if(self.alwaysOn or self.canComputeWeights(helperEpoch=helperEpoch, helper=helper, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, metadata=metadata)):
            self.countWeights += 1
            self.calcMean(model=model, smoothingMetadata=smoothingMetadata)
            return True

        return False

    def createDefaultMetadataObj(self):
        return _SmoothingOscilationBase_Metadata()

# borderline smoothing
class DefaultSmoothingBorderline_Metadata(sf.Smoothing_Metadata):
    def __init__(self, device = 'cpu',
        numbOfBatchAfterSwitchOn = 3000):
        super().__init__()

        self.device = device
        self.numbOfBatchAfterSwitchOn = numbOfBatchAfterSwitchOn # dla 50000 / 32 ~= 1500, 50000 / 16 ~= 3000

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Device:\t{}\n'.format(self.device))
        return tmp_str

class Test_DefaultSmoothingBorderline_Metadata(DefaultSmoothingBorderline_Metadata):
    def __init__(self, test_device = 'cpu',test_numbOfBatchAfterSwitchOn = 10):
        """
            Klasa z domyślnymi testowymi parametrami.
        """
        super().__init__(device=test_device, numbOfBatchAfterSwitchOn=test_numbOfBatchAfterSwitchOn)

class DefaultSmoothingBorderline(sf.Smoothing):
    """
    Włącza wygładzanie po przejściu przez określoną ilość iteracji pętli.
    Wygładzanie polega na liczeniu średnich tensorów.
    Wygładzanie włączane jest od momentu wykonania określonej ilości pętli oraz jest liczone od końca iteracji.
    Liczy średnią arytmetyczną.
    """
    def __init__(self, smoothingMetadata):
        super().__init__(smoothingMetadata=smoothingMetadata)

        if(not isinstance(smoothingMetadata, DefaultSmoothingBorderline_Metadata)):
            raise Exception("Metadata class '{}' is not the type of '{}'".format(type(smoothingMetadata), DefaultSmoothingBorderline_Metadata.__name__))

        self.sumWeights = {}
        self.previousWeights = {}
        self.countWeights = 0

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        return False

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        super().__call__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothingMetadata=smoothingMetadata)
        if(helperEpoch.trainTotalNumber > smoothingMetadata.numbOfBatchAfterSwitchOn):
            self.countWeights += 1
            if(hasattr(helper, 'substract')):
                del helper.substract
            helper.substract = {}
            with torch.no_grad():
                for key, arg in model.getNNModelModule().named_parameters():
                    cpuArg = arg.to(smoothingMetadata.device)
                    self.sumWeights[key].to(smoothingMetadata.device).add_(cpuArg)
                    #helper.substract[key] = arg.sub(self.previousWeights[key])
                    helper.substract[key] = self.previousWeights[key].sub_(cpuArg).multiply_(-1)
                    self.previousWeights[key].detach().copy_(cpuArg.detach())
            return True
        return False

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        average = super().__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        if(average is not None):
            return average
        average = {}
        if(self.countWeights == 0):
            return average
        for key, arg in self.sumWeights.items():
            average[key] = self.sumWeights[key].to(smoothingMetadata.device) / self.countWeights
        return average

    def __setDictionary__(self, smoothingMetadata, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=dictionary)
        with torch.no_grad():
            for key, values in dictionary:
                self.sumWeights[key] = torch.zeros_like(values, requires_grad=False, device=smoothingMetadata.device)
                self.previousWeights[key] = torch.zeros_like(values, requires_grad=False, device=smoothingMetadata.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['previousWeights']
            del state['countWeights']
            del state['counter']
            del state['sumWeights']
            del state['enabled']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.previousWeights = {}
            self.countWeights = 0
            self.sumWeights = {}
            self.enabled = False

    def createDefaultMetadataObj(self):
        return DefaultSmoothingBorderline_Metadata()

# oscilation generalized mean
class DefaultSmoothingOscilationGeneralizedMean_Metadata(_SmoothingOscilationBase_Metadata):
    def __init__(self, generalizedMeanPower = 1,
        device = 'cpu',
        weightSumContainerSize = 10, weightSumContainerSizeStartAt=5, softMarginAdditionalLoops = 20, 
        batchPercentMaxStart = 0.9988, batchPercentMinStart = 0.02, 
        epsilon = 1e-6, hardEpsilon=1e-8, weightsEpsilon = 1e-7,
        lossContainer=50, lossContainerDelayedStartAt = 25):

        super().__init__(device=device,
        weightSumContainerSize=weightSumContainerSize, weightSumContainerSizeStartAt=weightSumContainerSizeStartAt, 
        softMarginAdditionalLoops=softMarginAdditionalLoops, batchPercentMaxStart=batchPercentMaxStart,
        batchPercentMinStart=batchPercentMinStart, epsilon=epsilon, hardEpsilon=hardEpsilon, weightsEpsilon=weightsEpsilon,
        lossContainer=lossContainer, lossContainerDelayedStartAt=lossContainerDelayedStartAt)

        self.generalizedMeanPower = generalizedMeanPower # tylko dla 1 dobrze działa, reszta daje gorsze wyniki, info do opisania

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Generalized mean power:\t{}\n'.format(self.generalizedMeanPower))
        return tmp_str

class Test_DefaultSmoothingOscilationGeneralizedMean_Metadata(DefaultSmoothingOscilationGeneralizedMean_Metadata):
    def __init__(self, test_generalizedMeanPower = 1,
        test_device = 'cpu',
        test_weightSumContainerSize = 10, test_weightSumContainerSizeStartAt=5, test_softMarginAdditionalLoops = 3, 
        test_batchPercentMaxStart = 0.85, test_batchPercentMinStart = 0.1, 
        test_epsilon = 1e-4, test_hardEpsilon=1e-9, test_weightsEpsilon = 1e-5,
        test_lossContainer=5, test_lossContainerDelayedStartAt = 2):

        super().__init__(
            generalizedMeanPower=test_generalizedMeanPower,
            device=test_device,
            weightSumContainerSize=test_weightSumContainerSize, weightSumContainerSizeStartAt=test_weightSumContainerSizeStartAt, 
            softMarginAdditionalLoops=test_softMarginAdditionalLoops, batchPercentMaxStart=test_batchPercentMaxStart,
            batchPercentMinStart=test_batchPercentMinStart, epsilon=test_epsilon, hardEpsilon=test_hardEpsilon, weightsEpsilon=test_weightsEpsilon,
            lossContainer=test_lossContainer, lossContainerDelayedStartAt=test_lossContainerDelayedStartAt
        )

class DefaultSmoothingOscilationGeneralizedMean(_SmoothingOscilationBase):
    """
    Włącza wygładzanie gdy zostaną spełnione określone warunki:
    - po przekroczeniu pewnej minimalnej ilości iteracji pętli oraz 
        gdy różnica pomiędzy średnią N ostatnich strat treningowych modelu, 
        a średnią średnich N ostatnich strat treningowych modelu jest mniejsza niż epsilon.
    - po przekroczeniu pewnej maksymalnej ilości iteracji pętli.

    Liczy średnią generalizowaną dla wag (domyślne średnia arytmetyzna).
    """
    def __init__(self, smoothingMetadata):
        super().__init__(smoothingMetadata=smoothingMetadata)

        self.sumWeights = {}

        # methods
        self.methodPow = None
        self.methodDiv = None

        # set method
        if(smoothingMetadata.generalizedMeanPower == 1):
            self.methodPow = self.methodPow_1
            self.methodDiv = self.methodDiv_1
        else:
            self.methodPow = self.methodPow_
            self.methodDiv = self.methodDiv_

    def methodPow_(self, key, arg, smoothingMetadata):
        self.sumWeights[key].add_(arg.to(smoothingMetadata.device).pow(smoothingMetadata.generalizedMeanPower))

    def methodPow_1(self, key, arg, smoothingMetadata):
        self.sumWeights[key].add_(arg.to(smoothingMetadata.device))

    def methodDiv_(self, arg, smoothingMetadata):
        return (arg / self.countWeights).pow(1/smoothingMetadata.generalizedMeanPower)

    def methodDiv_1(self, arg, smoothingMetadata):
        return (arg / self.countWeights)

    def calcMean(self, model, smoothingMetadata):
        with torch.no_grad():
            for key, arg in model.getNNModelModule().named_parameters():
                self.methodPow(key=key, arg=arg, smoothingMetadata=smoothingMetadata)

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        average = super().__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        if(average is not None):
            return average
        average = {}
        if(self.countWeights == 0):
            return average
        for key, arg in self.sumWeights.items():
            average[key] = self.methodDiv(arg=arg, smoothingMetadata=smoothingMetadata)
        return average

    def __setDictionary__(self, smoothingMetadata, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(dictionary=dictionary, smoothingMetadata=smoothingMetadata)
        with torch.no_grad():
            for key, values in dictionary:
                self.sumWeights[key] = torch.zeros_like(values, requires_grad=False, device=smoothingMetadata.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['countWeights']
            del state['counter']
            del state['sumWeights']
            del state['enabled']
            del state['divisionCounter']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.countWeights = 0
            self.sumWeights = {}
            self.enabled = False
            self.divisionCounter = 0

            self.lossContainer.reset()
            self.tensorPrevSum_1.reset()
            self.tensorPrevSum_2.reset()

    def createDefaultMetadataObj(self):
        return DefaultSmoothingOscilationGeneralizedMean_Metadata()

# oscilation moving mean
class DefaultSmoothingOscilationMovingMean_Metadata(_SmoothingOscilationBase_Metadata):
    def __init__(self, movingAvgParam = 0.27,
        device = 'cpu',
        weightSumContainerSize = 10, weightSumContainerSizeStartAt=5, softMarginAdditionalLoops = 20, 
        batchPercentMaxStart = 0.9988, batchPercentMinStart = 0.02, 
        epsilon = 1e-6, hardEpsilon=1e-8, weightsEpsilon = 1e-7,
        lossContainer=50, lossContainerDelayedStartAt = 25):

        super().__init__(device=device,
        weightSumContainerSize=weightSumContainerSize, weightSumContainerSizeStartAt=weightSumContainerSizeStartAt, 
        softMarginAdditionalLoops=softMarginAdditionalLoops, batchPercentMaxStart=batchPercentMaxStart,
        batchPercentMinStart=batchPercentMinStart, epsilon=epsilon, hardEpsilon=hardEpsilon, weightsEpsilon=weightsEpsilon,
        lossContainer=lossContainer, lossContainerDelayedStartAt=lossContainerDelayedStartAt)

        # movingAvgParam jest parametrem 'a' dla wzoru: S = ax + (1-a)S
        self.movingAvgParam = movingAvgParam

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Moving average parameter a:(ax + (a-1)S):\t{}\n'.format(self.movingAvgParam))
        return tmp_str

class Test_DefaultSmoothingOscilationMovingMean_Metadata(DefaultSmoothingOscilationMovingMean_Metadata):
    def __init__(self, test_movingAvgParam = 0.27,
        test_device = 'cpu',
        test_weightSumContainerSize = 10, test_weightSumContainerSizeStartAt=5, test_softMarginAdditionalLoops = 3, 
        test_batchPercentMaxStart = 0.85, test_batchPercentMinStart = 0.1, 
        test_epsilon = 1e-4, test_hardEpsilon=1e-9, test_weightsEpsilon = 1e-5,
        test_lossContainer=5, test_lossContainerDelayedStartAt = 2):

        super().__init__(movingAvgParam=test_movingAvgParam, device=test_device,
        weightSumContainerSize=test_weightSumContainerSize, weightSumContainerSizeStartAt=test_weightSumContainerSizeStartAt, 
        softMarginAdditionalLoops=test_softMarginAdditionalLoops, batchPercentMaxStart=test_batchPercentMaxStart,
        batchPercentMinStart=test_batchPercentMinStart, epsilon=test_epsilon, hardEpsilon=test_hardEpsilon, weightsEpsilon=test_weightsEpsilon,
        lossContainer=test_lossContainer, lossContainerDelayedStartAt=test_lossContainerDelayedStartAt)

class DefaultSmoothingOscilationMovingMean(_SmoothingOscilationBase):
    """
    Włącza wygładzanie gdy zostaną spełnione określone warunki:
    - po przekroczeniu pewnej minimalnej ilości iteracji pętli oraz 
        gdy różnica pomiędzy średnią N ostatnich strat treningowych modelu, 
        a średnią średnich N ostatnich strat treningowych modelu jest mniejsza niż epsilon.
    - po przekroczeniu pewnej maksymalnej ilości iteracji pętli.

    Liczy średnią ważoną dla wag. Wagi są nadawane względem starości zapamiętanej wagi. Im starsza tym ma mniejszą wagę.
    """
    def __init__(self, smoothingMetadata):
        super().__init__(smoothingMetadata=smoothingMetadata)

        self.weightsSum = {}
        self.__setMovingAvgParam(value=smoothingMetadata.movingAvgParam, smoothingMetadata=smoothingMetadata)

    def __setMovingAvgParam(self, value, smoothingMetadata):
        if(value >= 1.0 or value <= 0.0):
            raise Exception("Value of {}.movingAvgParam can only be in the range [0; 1]".format(self.__name__))

        self.movingAvgTensorLow = torch.tensor(value).to(smoothingMetadata.device)
        self.movingAvgTensorHigh = torch.tensor(1 - value).to(smoothingMetadata.device)

    def calcMean(self, model, smoothingMetadata):
        with torch.no_grad():
            for key, val in model.getNNModelModule().named_parameters():
                valDevice = val.device
                # S = ax + (1-a)S
                self.weightsSum[key] = self.weightsSum[key].mul_(self.movingAvgTensorHigh).add_(
                    (val.mul(self.movingAvgTensorLow.to(valDevice))).to(smoothingMetadata.device)
                    )

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        average = super().__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        if(average is not None):
            return average # {}
        average = {}
        if(self.countWeights == 0):
            return average # {}

        return sf.cloneTorchDict(self.weightsSum)

    def __setDictionary__(self, smoothingMetadata, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=dictionary)
        with torch.no_grad():
            for key, values in dictionary:
                # ważne jest, aby skopiować początkowe wagi, a nie stworzyć tensor zeros_like
                # w przeciwnym wypadku średnia będzie dawała bardzo złe wyniki
                self.weightsSum[key] = torch.clone(values).to(smoothingMetadata.device).requires_grad_(False) 

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['countWeights']
            del state['counter']
            del state['enabled']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.countWeights = 0
            self.enabled = False
            self.divisionCounter = 0

            self.lossContainer.reset()
            self.tensorPrevSum_1.reset()
            self.tensorPrevSum_2.reset()

    def createDefaultMetadataObj(self):
        return DefaultSmoothingOscilationMovingMean_Metadata()

# oscilation weighted mean
smoothingEndCheckTypeDict = [
    'std',
    'wgsum'
]


class DefaultSmoothingOscilationWeightedMean_Metadata(_SmoothingOscilationBase_Metadata):
    def __init__(self, weightIter = DefaultWeightDecay(), weightsArraySize=20, smoothingEndCheckType='std',
        device = 'cpu',
        weightSumContainerSize = 10, weightSumContainerSizeStartAt=5, softMarginAdditionalLoops = 20, 
        batchPercentMaxStart = 0.9988, batchPercentMinStart = 0.02, 
        epsilon = 1e-6, hardEpsilon=1e-8, weightsEpsilon = 1e-7,
        lossContainer=50, lossContainerDelayedStartAt = 25):

        super().__init__(device=device,
        weightSumContainerSize=weightSumContainerSize, weightSumContainerSizeStartAt=weightSumContainerSizeStartAt, 
        softMarginAdditionalLoops=softMarginAdditionalLoops, batchPercentMaxStart=batchPercentMaxStart,
        batchPercentMinStart=batchPercentMinStart, epsilon=epsilon, hardEpsilon=hardEpsilon, weightsEpsilon=weightsEpsilon,
        lossContainer=lossContainer, lossContainerDelayedStartAt=lossContainerDelayedStartAt)

        # jak bardzo następne wagi w kolejce mają stracić na wartości. Kolejne wagi dzieli się przez wielokrotność tej wartości.
        self.weightIter = weightIter
        self.weightsArraySize=weightsArraySize
        self.smoothingEndCheckType=smoothingEndCheckType

        # validate
        if(self.smoothingEndCheckType not in smoothingEndCheckTypeDict):
            raise Exception("Unknown type of smoothingEndCheckType: {}\nPossible values:\n{}".format(
                self.smoothingEndCheckType, smoothingEndCheckTypeDict))

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('\nStart inner {} class\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'.format(type(self.weightIter).__name__))
        tmp_str += str(self.weightIter)
        tmp_str += ('Weight array size:\t{}\n'.format(self.weightsArraySize))
        tmp_str += ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nEnd inner {} class\n'.format(type(self.weightIter).__name__))
        return tmp_str

class Test_DefaultSmoothingOscilationWeightedMean_Metadata(DefaultSmoothingOscilationWeightedMean_Metadata):
    def __init__(self, test_weightIter = DefaultWeightDecay(), test_weightsArraySize=20, test_smoothingEndCheckType='std',
        test_device = 'cpu',
        test_weightSumContainerSize = 10, test_weightSumContainerSizeStartAt=5, test_softMarginAdditionalLoops = 3, 
        test_batchPercentMaxStart = 0.85, test_batchPercentMinStart = 0.1, 
        test_epsilon = 1e-4, test_hardEpsilon=1e-9, test_weightsEpsilon = 1e-5,
        test_lossContainer=5, test_lossContainerDelayedStartAt = 2
    ):
        super().__init__(weightIter=test_weightIter, weightsArraySize=test_weightsArraySize, smoothingEndCheckType=test_smoothingEndCheckType,
        device=test_device,
        weightSumContainerSize=test_weightSumContainerSize, weightSumContainerSizeStartAt=test_weightSumContainerSizeStartAt, 
        softMarginAdditionalLoops=test_softMarginAdditionalLoops, batchPercentMaxStart=test_batchPercentMaxStart,
        batchPercentMinStart=test_batchPercentMinStart, epsilon=test_epsilon, hardEpsilon=test_hardEpsilon, weightsEpsilon=test_weightsEpsilon,
        lossContainer=test_lossContainer, lossContainerDelayedStartAt=test_lossContainerDelayedStartAt)

class DefaultSmoothingOscilationWeightedMean(_SmoothingOscilationBase):
    """
    Włącza wygładzanie gdy zostaną spełnione określone warunki:
    - po przekroczeniu pewnej minimalnej ilości iteracji pętli oraz 
        gdy różnica pomiędzy średnią N ostatnich strat treningowych modelu, 
        a średnią średnich N ostatnich strat treningowych modelu jest mniejsza niż epsilon.
    - po przekroczeniu pewnej maksymalnej ilości iteracji pętli.

    Liczy średnią ważoną dla wag. Wagi są nadawane względem starości zapamiętanej wagi. Im starsza tym ma mniejszą wagę.
    """
    def __init__(self, smoothingMetadata):
        super().__init__(smoothingMetadata=smoothingMetadata)

        # trzeba uważać na zajętość w pamięci. Zapamiętuje wszystkie wagi modelu, co może sumować się do dużych rozmiarów
        self.weightsArray = CircularList(smoothingMetadata.weightsArraySize) 
        if(smoothingMetadata.smoothingEndCheckType == 'std'):
            self.isSmoothingGoodEnoughMethod = DefaultSmoothingOscilationWeightedMean.__isSmoothingGoodEnough__std
        elif(smoothingMetadata.smoothingEndCheckType == 'wgsum'):
            self.isSmoothingGoodEnoughMethod = _SmoothingOscilationBase.__isSmoothingGoodEnough__
        else:
            raise Exception("Unknown type of smoothingEndCheckType: {}".format(smoothingMetadata.smoothingEndCheckType))
        
    def calcMean(self, model, smoothingMetadata):
        with torch.no_grad():
            tmpDict = {}
            for key, val in model.getNNModelModule().named_parameters():
                tmpDict[key] = val.clone().to(smoothingMetadata.device)
            self.weightsArray.pushBack(tmpDict)

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        average = super().__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        if(average is not None):
            return average # {}
        average = {}
        if(self.countWeights == 0):
            return average # {}

        for wg in self.weightsArray:
            for key, val in wg.items():
                average[key] = torch.zeros_like(val, requires_grad=False, device=smoothingMetadata.device)
            break

        iterWg = iter(smoothingMetadata.weightIter)
        wgSum = 0.0
        for wg in self.weightsArray:
            weight = next(iterWg)
            for key, val in wg.items(): 
                average[key] += val.mul(weight)
            wgSum += weight 

        for key, val in average.items():
            val.div_(wgSum)
        return average

    def __setDictionary__(self, smoothingMetadata, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=dictionary)

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['countWeights']
            del state['counter']
            del state['weightsArray']
            del state['enabled']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.countWeights = 0
            self.weightsArray.reset()
            self.enabled = False
            self.divisionCounter = 0

            self.lossContainer.reset()
            self.tensorPrevSum_1.reset()
            self.tensorPrevSum_2.reset()

    def createDefaultMetadataObj(self):
        return DefaultSmoothingOscilationWeightedMean_Metadata()

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        return self.isSmoothingGoodEnoughMethod(self, helperEpoch=helperEpoch, helper=helper, model=model, 
            dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothingMetadata=smoothingMetadata)

    def _sumWeightsToArrayStd(self, smWg):
        sumOfDiff = []
        for wg in self.weightsArray:
            diffSumDict = 0.0
            for key, tens in wg.items():
                diffSumDict += torch.sum(torch.abs(tens.sub(smWg[key])))
            sumOfDiff.append(diffSumDict)
        
        sumOfDiff = torch.Tensor(sumOfDiff)
        std = torch.std(sumOfDiff)
        del sumOfDiff
        torch.cuda.empty_cache()
        if(std.isnan()):
            return torch.tensor(ConfigClass.STD_NAN)
        return std

    def __isSmoothingGoodEnough__std(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        if(self.countWeights > 0):
            self.divisionCounter += 1
            smWg = self.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
            stdDev = self._sumWeightsToArrayStd(smWg=smWg)

            metadata.stream.print("Standard deviation:" + str(stdDev), 'debug:0')
            metadata.stream.print("Standard deviation bool: " + str(bool(stdDev < smoothingMetadata.weightsEpsilon)), 'debug:0')
            
            return self._smoothingGoodEnoughCheck(val=stdDev, smoothingMetadata=smoothingMetadata)
        return False

# data classes
class DefaultData_Metadata(sf.Data_Metadata):
    def __init__(self, worker_seed = 8418748, download = True, pin_memoryTrain = False, pin_memoryTest = False,
        epoch = 1, batchTrainSize = 16, batchTestSize = 16, fromGrayToRGB = True,
        test_howOftenPrintTrain = 200, howOftenPrintTrain = 2000):

        super().__init__(worker_seed = worker_seed, train = True, download = download, pin_memoryTrain = pin_memoryTrain, pin_memoryTest = pin_memoryTest,
            epoch = epoch, batchTrainSize = batchTrainSize, batchTestSize = batchTestSize, howOftenPrintTrain = howOftenPrintTrain)

        self.fromGrayToRGB = fromGrayToRGB

        # batch size * howOftenPrintTrain
        if(sf.test_mode.isActive()):
            self.howOftenPrintTrain = test_howOftenPrintTrain
        else:
            self.howOftenPrintTrain = howOftenPrintTrain

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Resize data from Gray to RGB:\t{}\n'.format(self.fromGrayToRGB))
        return tmp_str

class DefaultData(sf.Data):
    def __init__(self, dataMetadata):
        super().__init__(dataMetadata=dataMetadata)
        self.testAlias = 'statLossTest_normal'

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

    def __beforeTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        pass

    def __beforeTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__beforeTrain__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

    def __afterTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__afterTrain__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

    def __afterTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__afterTrainLoop__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        with torch.no_grad():
            if(helper.diff is not None):
                diffKey = list(helper.diff.keys())[-1]
                metadata.stream.print("\n\ntrainLoop;\nAverage train time;Loop train time;Weight difference of last layer average;divided by;", ['stat'])
                metadata.stream.print(f"{helper.timer.getAverage()};{helper.loopTimer.getDiff()};{helper.diff[diffKey].sum() / helper.diff[diffKey].numel()};{helper.diff[diffKey].numel()}", ['stat'])
                del diffKey

    def __beforeTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        metadata.stream.print("\n\ntestLoop;\nAverage test time;Loop test time;Accuracy;Avg loss", ['stat'])

    def __beforeTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__beforeTest__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

    def __afterTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__afterTest__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        metadata.stream.print(helper.test_loss, self.testAlias)

    def __afterTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__afterTestLoop__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        metadata.stream.print(f"{helper.timer.getAverage()};{helper.loopTimer.getDiff()};{(helperEpoch.statistics.correctRatio[-1]):>0.0001f};{helperEpoch.statistics.lossRatio[-1]:>8f}", ['stat'])

    def __howManyTestInvInOneEpoch__(self):
        return 2

    def __howManyTrainInvInOneEpoch__(self):
        return 1

    def __epoch__(self, helperEpoch: 'EpochDataContainer', model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        if(metadata.shouldTrain()):
            helperEpoch.currentLoopTimeAlias = 'loopTrainTime'
            self.trainLoop(model=model, helperEpoch=helperEpoch, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        
        if(sf.enabledSaveAndExit()):
            return 

        with torch.no_grad():
            if(metadata.shouldTest()):
                helperEpoch.currentLoopTimeAlias = 'loopTestTime_normal'
                self.testLoop(model=model, helperEpoch=helperEpoch, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                smoothing.saveWeights(weights=model.getNNModelModule().named_parameters(), key='main')
                wg = smoothing.__getSmoothedWeights__(metadata=metadata, smoothingMetadata=smoothingMetadata)
                if(wg):
                    self.setModelSmoothedWeights(model, helperEpoch, wg)
                    helperEpoch.currentLoopTimeAlias = 'loopTestTime_smooothing'
                    self.testAlias = 'statLossTest_smooothing'
                    helperEpoch.averaged = True
                    self.testLoop(model=model, helperEpoch=helperEpoch, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                    self.setModelNormalWeights(model, helperEpoch, smoothing.getWeights(key='main'))
                else:
                    sf.Output.printBash('Smoothing is not enabled. Test does not executed.', 'info')

    def createDefaultMetadataObj(self):
        return DefaultData_Metadata()

class DefaultDataMNIST(DefaultData):
    def __init__(self, dataMetadata):
        super().__init__(dataMetadata=dataMetadata)

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__(dataMetadata)

        #self.trainset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform)
        #self.testset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform)
        self.trainset = torchvision.datasets.MNIST(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform, download=dataMetadata.download)
        self.testset = torchvision.datasets.MNIST(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform, download=dataMetadata.download)

        self.trainSampler = sf.BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = sf.BaseSampler(len(self.testset), dataMetadata.batchTestSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

class DefaultDataEMNIST(DefaultData):
    def __init__(self, dataMetadata):
        super().__init__(dataMetadata=dataMetadata)

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__(dataMetadata)

        #self.trainset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform)
        #self.testset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform)
        self.trainset = torchvision.datasets.EMNIST(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform, split='digits', download=dataMetadata.download)
        self.testset = torchvision.datasets.EMNIST(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform, split='digits', download=dataMetadata.download)

        self.trainSampler = sf.BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = sf.BaseSampler(len(self.testset), dataMetadata.batchTestSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

class DefaultDataCIFAR10(DefaultData):
    def __init__(self, dataMetadata):
        super().__init__(dataMetadata=dataMetadata)

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__(dataMetadata)

        #self.trainset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform)
        #self.testset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform)
        self.trainset = torchvision.datasets.CIFAR10(root, split)(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform, download=dataMetadata.download)
        self.testset = torchvision.datasets.CIFAR10(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform, download=dataMetadata.download)

        self.trainSampler = sf.BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = sf.BaseSampler(len(self.testset), dataMetadata.batchTestSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

class DefaultDataCIFAR100(DefaultData):
    def __init__(self, dataMetadata):
        super().__init__(dataMetadata=dataMetadata)

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__(dataMetadata)

        #self.trainset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform)
        #self.testset = torchvision.datasets.ImageNet(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform)
        self.trainset = torchvision.datasets.CIFAR100(root, split)(root=sf.StaticData.DATA_PATH, train=True, transform=self.trainTransform, download=dataMetadata.download)
        self.testset = torchvision.datasets.CIFAR100(root=sf.StaticData.DATA_PATH, train=False, transform=self.testTransform, download=dataMetadata.download)

        self.trainSampler = sf.BaseSampler(len(self.trainset), dataMetadata.batchTrainSize)
        self.testSampler = sf.BaseSampler(len(self.testset), dataMetadata.batchTestSize)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=self.trainSampler,
                                          shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=self.testSampler,
                                         shuffle=False, num_workers=2, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)


ModelMap = {
    'simpleConvModel': DefaultModelSimpleConv,
    'predefModel': DefaultModelPredef
}

DataMap = {
    'MINST': DefaultDataMNIST,
    'CIFAR10': DefaultDataCIFAR10,
    'CIFAR100': DefaultDataCIFAR100,
    'EMNIST': DefaultDataEMNIST
}

SmoothingMap = {
    'disabled': DisabledSmoothing,
    'borderline': DefaultSmoothingBorderline,
    'generalizedMean': DefaultSmoothingOscilationGeneralizedMean,
    'movingMean': DefaultSmoothingOscilationMovingMean,
    'weightedMean': DefaultSmoothingOscilationWeightedMean
}

def run(modelType, dataType, smoothingType, metadataObj, modelMetadata, dataMetadata, smoothingMetadata, modelPredefObj = None, 
    numbOfRepetition = 1):

    listStat = []
    folderRelativeRoot = None

    for experimentNumber in range(numbOfRepetition):
        if(modelPredefObj is None and modelType == 'predefModel'):
            sf.Output.printBash("Cannot run test, because model '{}' does not have provided object.".format(modelType), 'warn')
            return
        if(modelType not in ModelMap):
            sf.Output.printBash("Cannot run test, because model '{}' not found.".format(modelType), 'warn')
            return
        if(dataType not in DataMap):
            sf.Output.printBash("Cannot run test, because data '{}' not found.".format(dataType), 'warn')
            return
        if(smoothingType not in SmoothingMap):
            sf.Output.printBash("Cannot run test, because smoothing '{}' not found.".format(smoothingType), 'warn')
            return
        if(modelPredefObj is not None and modelType != 'predefModel'):
            sf.Output.printBash("Cannot run test, because modelObj is not None and choosed model type '{}'.".format(modelType), 'warn')
            return

        metadataObj.resetOutput() 

        logFolderSuffix = modelType + '_' + dataType + '_' + smoothingType
        if(numbOfRepetition != 1):
            if(folderRelativeRoot is None):
                folderRelativeRoot = os.path.basename(sf.Output.createLogFolder(logFolderSuffix + "_set"))

        metadataObj.name = logFolderSuffix
        metadataObj.logFolderSuffix = logFolderSuffix
        metadataObj.relativeRoot = folderRelativeRoot

        metadataObj.prepareOutput()
        
        model = None
        data = DataMap[dataType](dataMetadata)
        smoothing = SmoothingMap[smoothingType](smoothingMetadata)
        statistics = None

        if(modelPredefObj is None):
            model = ModelMap[modelType](modelMetadata=modelMetadata)
        else:
            model = ModelMap[modelType](obj=modelPredefObj, modelMetadata=modelMetadata)
        smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().named_parameters())

        metadataObj.printStartNewModel()

        statistics = sf.runObjs(metadataObj=metadataObj, dataMetadataObj=dataMetadata, modelMetadataObj=modelMetadata, 
                smoothingMetadataObj=smoothingMetadata, smoothingObj=smoothing, dataObj=data, modelObj=model, folderLogNameSuffix=logFolderSuffix, 
                folderRelativeRoot=folderRelativeRoot)

        metadataObj.printEndModel()

        listStat.append(statistics)

    if(numbOfRepetition == 1):
        return listStat[0]

    return listStat



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




