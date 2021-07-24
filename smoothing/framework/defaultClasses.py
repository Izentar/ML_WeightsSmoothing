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

class ConfigClass():
    STD_NAN = 1e+10 # standard value if NaN


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
        tmp_str = ('\nStart inner {} class\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'.format(type(self).__name__))
        tmp_str += ('Weight decay:\t{}\n'.format(self.weightDecay))
        tmp_str += ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\nEnd inner {} class\n'.format(type(self).__name__))
        return tmp_str

# model classes
class DefaultModel_Metadata(sf.Model_Metadata):
    def __init__(self, lossFuncDataDict=None, optimizerDataDict=None,
        device = 'cuda:0'):
        """
            lossFuncDataDict - domyślnie {} dla None
            optimizerDataDict - domyślnie {} dla None
        """
        super().__init__()
        self.device = device
        self.lossFuncDataDict = lossFuncDataDict if lossFuncDataDict is not None else {}
        self.optimizerDataDict = optimizerDataDict if optimizerDataDict is not None else {}

    def prepare(self, lossFunc, optimizer):
        self.loss_fn = lossFunc
        self.optimizer = optimizer

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Model device :\t{}\n'.format(self.device))
        tmp_str += ('Loss function name:\t{}\n'.format(str(type(self.loss_fn))))
        tmp_str += ('Loss function values:\n{}\n'.format(self.lossFuncDataDict))
        tmp_str += ('Optimizer name:\t{}\n'.format(str(type(self.optimizer))))
        tmp_str += ('Optimizer values:\n')
        if(hasattr(self.optimizer, "defaults")):
            tmp_str += ('{}\n'.format((self.optimizer.defaults)))
        else:
            tmp_str += ('Not found\n')
        tmp_str += ('Optimizer provided data:\n{}\n'.format((self.optimizerDataDict)))
        return tmp_str

class DefaultModelSimpleConv(sf.Model):
    """
    Z powodu jego prostoty i słabych wyników zaleca się go używać podczas testowania sieci.
    """
    def __init__(self, modelMetadata):
        super().__init__(modelMetadata=modelMetadata)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.linear1 = nn.Linear(16*6*6, 212)
        self.linear2 = nn.Linear(212, 120)
        self.linear3 = nn.Linear(120, 84)
        self.linear4 = nn.Linear(84, 10)

        #self.loss_fn = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(self.getNNModelModule().parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)
        #self.optimizer = optim.AdamW(self.parameters(), lr=modelMetadata.learning_rate)

        self.getNNModelModule().to(modelMetadata.device)
        self.__initializeWeights__()

    def forward(self, x):
        x = self.pool(F.hardswish(self.conv1(x)))
        x = self.pool(F.hardswish(self.conv2(x)))
        # 16 * 212 * 212 może zmienić rozmiar tensora na [1, 16 * 212 * 212] co nie zgadza się z rozmiarem batch_number 1 != 16. Wtedy należy dać [-1, 212 * 212] = [16, 212 * 212]
        # ogółem ta operacja nie jest bezpieczna przy modyfikacji danych.
        #x = x.view(x.size(0), -1)   
        x = self.avgpool(x)
        print(x.size())
        x = x.view(x.size(0), -1)   
        print(x.size())
        x = F.hardswish(self.linear1(x))
        x = F.hardswish(self.linear2(x))
        x = F.hardswish(self.linear3(x))
        x = self.linear4(x)
        return x

    def __update__(self, modelMetadata):
        self.getNNModelModule().to(modelMetadata.device)
        #self.optimizer = optim.SGD(self.getNNModelModule().parameters(), lr=modelMetadata.learning_rate, momentum=modelMetadata.momentum)

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
    def __init__(self, obj, modelMetadata, name):
        super().__init__(obj=obj, modelMetadata=modelMetadata, name=name)
        self.getNNModelModule().to(modelMetadata.device)

    def __update__(self, modelMetadata):
        self.getNNModelModule().to(modelMetadata.device)

    def createDefaultMetadataObj(self):
        return DefaultModel_Metadata()

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
                spełniony, to możliwe jest, że wygładzanie nie uzwględni nowych wag. Jest to różnica uśrednionych strat modelu 
                z listy cyklicznej
            hardEpsilon - kiedy wygładzanie powinno zostać włączone pernamentnie, niezależnie od parametru 'epsilon'. Liczy się tak samo
                jak epsilon.
            weightsEpsilon - kiedy można uznać, że wygładzanie powinno zostać zakończone wraz z zakończeniem pętli treningowej. Jest to
                różnica uśrednionych wag z listy cyklicznej
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
        device = 'cpu',
        weightSumContainerSize = 10, weightSumContainerSizeStartAt=5, softMarginAdditionalLoops = 3, 
        batchPercentMaxStart = 0.85, batchPercentMinStart = 0.1, 
        epsilon = 1e-4, hardEpsilon=1e-9, weightsEpsilon = 1e-5,
        lossContainer=5, lossContainerDelayedStartAt = 2):
        """
            Klasa z domyślnymi testowymi parametrami.
        """

        super().__init__(device=device,
        weightSumContainerSize=weightSumContainerSize, weightSumContainerSizeStartAt=weightSumContainerSizeStartAt, 
        softMarginAdditionalLoops=softMarginAdditionalLoops, batchPercentMaxStart=batchPercentMaxStart,
        batchPercentMinStart=batchPercentMinStart, epsilon=epsilon, hardEpsilon=hardEpsilon, weightsEpsilon=weightsEpsilon,
        lossContainer=lossContainer, lossContainerDelayedStartAt=lossContainerDelayedStartAt)

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
        
        self.countWeights = 0 # liczba wywołań calcMean()
        self.countWeightsInaRow = 0 # liczba wywołań calcMean() z rzędu. Resetowana do 0 gdy nie zostanie wywołana
        self.tensorPrevSum = sf.CircularList(int(smoothingMetadata.weightSumContainerSize))
        self.divisionCounter = 0
        self.goodEnoughCounter = 0
        self.alwaysOn = False # gdy osiągnie warunek twardego epsilona
        self.weightsComputed = False

        self.lossContainer = sf.CircularList(smoothingMetadata.lossContainerSize)
        
    def canComputeWeights(self, helper, helperEpoch, dataMetadata, smoothingMetadata, metadata):
        """
        - Jeżeli wartość bezwzględna różnicy średnich N ostatnich strat f. celu, a średnią K średnich N ostatnich strat f. celu będzie mniejsza niż epsilon
        i program przeszedł przez minimalną liczbę pętli, to metoda zwróci True.
        W przeciwnym wypadku zwróci False.
        """
        avg_1 = self.lossContainer.getAverage()
        avg_2 = self.lossContainer.getAverage(smoothingMetadata.lossContainerDelayedStartAt)
        metadata.stream.print("Loss average: {} : {}".format(avg_1, avg_2), 'debug:0')

        # oblicz bezwzględną różnicę dwóch średnich strat
        absAvgDiff = abs(avg_1 - avg_2)

        # czy spełniono waruek na twardy epsilon
        minStart = smoothingMetadata.batchPercentMinStart * helperEpoch.maxTrainTotalNumber
        if(absAvgDiff < smoothingMetadata.hardEpsilon and helperEpoch.trainTotalNumber > minStart and not self.alwaysOn):
            self.alwaysOn = True
            helperEpoch.addSmoothingMode()
            metadata.stream.print("Reached hard epsilon. Average losses are: {}; {}\nSmoothing scheduler enabled.".format(
                avg_1, avg_2), ['debug:0', 'model:0'])

        # czy wykonano maksymalną liczbę pętli przed włączeniem wygładzania
        return bool(
                # jeżeli osiągnięto twardy epsilon
                (self.alwaysOn)
            or (
                # jeżeli osiągnięto dostatecznie małą różnicę średnich strat oraz wykonano minimalną liczbę pętli
                absAvgDiff < smoothingMetadata.epsilon and helperEpoch.trainTotalNumber >= minStart
            )
            or (
                # jeżeli ilość wykonanych pętli treningowych przekracza maksymalną wartość, po której wygładzanie powinno zostać
                # włączone niezależnie od innych czynników
                helperEpoch.trainTotalNumber > (smoothingMetadata.batchPercentMaxStart * helperEpoch.maxTrainTotalNumber)
            )
        )

    def _sumAllWeights(self, smoothingMetadata, metadata):
        smWg = self.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        return sf.sumAllWeights(smWg)

    def _smoothingGoodEnoughCheck(self, val, smoothingMetadata):
        ret = bool(val < smoothingMetadata.weightsEpsilon)
        if(ret):
            # jeżeli dotknięto soft margin, to zwróć True w przypadku jego przekroczenia. Inaczej False oraz zwiększ licznik.
            if(smoothingMetadata.softMarginAdditionalLoops >= self.goodEnoughCounter):
                self.goodEnoughCounter += 1
                return False
            return True
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
        if(self.countWeightsInaRow > smoothingMetadata.softMarginAdditionalLoops): 
            self.divisionCounter += 1
            absSum = self._sumAllWeights(smoothingMetadata=smoothingMetadata, metadata=metadata)

            self.tensorPrevSum.pushBack(absSum)
            avg_1, avg_2 = self.tensorPrevSum.getAverage(), self.tensorPrevSum.getAverage(smoothingMetadata.weightSumContainerSizeStartAt)

            metadata.stream.print("Sum debug:" + str(absSum), 'debug:0')
            metadata.stream.print("Weight avg diff: " + str(abs(avg_1 - avg_2)), 'debug:0')
            metadata.stream.print("Weight avg diff bool: " + str(bool(abs(avg_1 - avg_2) < smoothingMetadata.weightsEpsilon)), 'debug:0')
            
            return self._smoothingGoodEnoughCheck(val=abs(avg_1 - avg_2), smoothingMetadata=smoothingMetadata)
        return False

    def calcMean(self, model, smoothingMetadata):
        self.countWeights += 1
        self.countWeightsInaRow += 1

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        super().__call__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothingMetadata=smoothingMetadata)
        # dodaj stratę do listy cyklicznej
        self.lossContainer.pushBack(helper.loss.item())

        metadata.stream.print("Loss avg diff : " + 
            str(abs(self.lossContainer.getAverage() - self.lossContainer.getAverage(smoothingMetadata.lossContainerDelayedStartAt))), 'debug:0')

        self.weightsComputed = self.canComputeWeights(helperEpoch=helperEpoch, helper=helper, dataMetadata=dataMetadata, smoothingMetadata=smoothingMetadata, metadata=metadata)
        if(self.weightsComputed):                
            self.calcMean(model=model, smoothingMetadata=smoothingMetadata)
            return True
        self.countWeightsInaRow = 0
        return False

    def createDefaultMetadataObj(self):
        return _SmoothingOscilationBase_Metadata()

# simple mean smoothing
class DefaultSmoothingSimpleMean_Metadata(sf.Smoothing_Metadata):
    def __init__(self, device = 'cpu',
        batchPercentStart = 0.8):
        super().__init__()

        self.device = device
        self.batchPercentStart = batchPercentStart # dla 50000 / 32 ~= 1500, 50000 / 16 ~= 3000

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Device:\t{}\n'.format(self.device))
        tmp_str += ('Number of batches after smoothing on:\t{}\n'.format(self.batchPercentStart))
        return tmp_str

class Test_DefaultSmoothingSimpleMean_Metadata(DefaultSmoothingSimpleMean_Metadata):
    def __init__(self, device = 'cpu', numbOfBatchAfterSwitchOn = 0.5):
        """
            Klasa z domyślnymi testowymi parametrami.
        """
        super().__init__(device=device, batchPercentStart=numbOfBatchAfterSwitchOn)

class DefaultSmoothingSimpleMean(sf.Smoothing):
    """
    Włącza wygładzanie po przejściu przez określoną ilość iteracji pętli.
    Wygładzanie polega na liczeniu średnich tensorów.
    Wygładzanie włączane jest od momentu wykonania określonej ilości pętli oraz jest liczone od końca iteracji.
    Liczy średnią arytmetyczną.
    """
    def __init__(self, smoothingMetadata):
        super().__init__(smoothingMetadata=smoothingMetadata)

        if(not isinstance(smoothingMetadata, DefaultSmoothingSimpleMean_Metadata)):
            raise Exception("Metadata class '{}' is not the type of '{}'".format(type(smoothingMetadata), DefaultSmoothingSimpleMean_Metadata.__name__))

        self.sumWeights = {}
        self.countWeights = 0
        self.smoothingEnabled = False

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        return False

    def _firstSetup(self, helperEpoch):
        if(not self.smoothingEnabled):
            self.smoothingEnabled = True
            helperEpoch.addSmoothingMode()

    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        super().__call__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothingMetadata=smoothingMetadata)
        if(helperEpoch.trainTotalNumber > smoothingMetadata.batchPercentStart * helperEpoch.maxTrainTotalNumber):
            self.countWeights += 1
            with torch.no_grad():
                for key, arg in model.getNNModelModule().state_dict().items():
                    deviceArg = arg.to(smoothingMetadata.device)
                    self.sumWeights[key].to(smoothingMetadata.device).add_(deviceArg)
            return True
        return False

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        tmpCheck = super().__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        if(tmpCheck is not None):
            return tmpCheck


        average = {}
        if(self.countWeights == 0):
            return average
        for key, arg in self.sumWeights.items():
            average[key] = self.sumWeights[key].to(smoothingMetadata.device).div(self.countWeights)
        return average

    def __setDictionary__(self, smoothingMetadata, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=dictionary)
        with torch.no_grad():
            for key, values in dictionary:
                self.sumWeights[key] = torch.zeros_like(values, requires_grad=False, device=smoothingMetadata.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['countWeights']
            del state['sumWeights']
            del state['enabled']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.countWeights = 0
            self.sumWeights = {}
            self.enabled = False

    def createDefaultMetadataObj(self):
        return DefaultSmoothingSimpleMean_Metadata()

# oscilation generalized mean
class DefaultSmoothingOscilationGeneralizedMean_Metadata(_SmoothingOscilationBase_Metadata):
    def __init__(self, generalizedMeanPower = 1.0,
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
    def __init__(self, generalizedMeanPower = 1.0,
        device = 'cpu',
        weightSumContainerSize = 10, weightSumContainerSizeStartAt=5, softMarginAdditionalLoops = 3, 
        batchPercentMaxStart = 0.85, batchPercentMinStart = 0.1, 
        epsilon = 1e-4, hardEpsilon=1e-9, weightsEpsilon = 1e-5,
        lossContainer=5, lossContainerDelayedStartAt = 2):

        super().__init__(
            generalizedMeanPower=generalizedMeanPower,
            device=device,
            weightSumContainerSize=weightSumContainerSize, weightSumContainerSizeStartAt=weightSumContainerSizeStartAt, 
            softMarginAdditionalLoops=softMarginAdditionalLoops, batchPercentMaxStart=batchPercentMaxStart,
            batchPercentMinStart=batchPercentMinStart, epsilon=epsilon, hardEpsilon=hardEpsilon, weightsEpsilon=weightsEpsilon,
            lossContainer=lossContainer, lossContainerDelayedStartAt=lossContainerDelayedStartAt
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
        self.mean = None

    def calcMean(self, model, smoothingMetadata):
        super().calcMean(model=model, smoothingMetadata=smoothingMetadata)
        self.mean.addWeights(model.getNNModelModule().state_dict())

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        tmpCheck = super().__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        if(tmpCheck is not None):
            return tmpCheck
        return self.mean.getWeights()

    def __setDictionary__(self, smoothingMetadata, dictionary):
        '''
        Used to map future weights into internal sums.
        '''
        super().__setDictionary__(dictionary=dictionary, smoothingMetadata=smoothingMetadata)
        self.mean = sf.RunningGeneralMeanWeights(initWeights=dictionary, power=smoothingMetadata.generalizedMeanPower, device=smoothingMetadata.device, setToZeros=True)

    def __getstate__(self):
        state = self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def createDefaultMetadataObj(self):
        return DefaultSmoothingOscilationGeneralizedMean_Metadata()

# oscilation moving mean
class DefaultSmoothingOscilationEWMA_Metadata(_SmoothingOscilationBase_Metadata):
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

class Test_DefaultSmoothingOscilationEWMA_Metadata(DefaultSmoothingOscilationEWMA_Metadata):
    def __init__(self, movingAvgParam = 0.27,
        device = 'cpu',
        weightSumContainerSize = 10, weightSumContainerSizeStartAt=5, softMarginAdditionalLoops = 3, 
        batchPercentMaxStart = 0.85, batchPercentMinStart = 0.1, 
        epsilon = 1e-4, hardEpsilon=1e-9, weightsEpsilon = 1e-5,
        lossContainer=5, lossContainerDelayedStartAt = 2):

        super().__init__(movingAvgParam=movingAvgParam, device=device,
        weightSumContainerSize=weightSumContainerSize, weightSumContainerSizeStartAt=weightSumContainerSizeStartAt, 
        softMarginAdditionalLoops=softMarginAdditionalLoops, batchPercentMaxStart=batchPercentMaxStart,
        batchPercentMinStart=batchPercentMinStart, epsilon=epsilon, hardEpsilon=hardEpsilon, weightsEpsilon=weightsEpsilon,
        lossContainer=lossContainer, lossContainerDelayedStartAt=lossContainerDelayedStartAt)

class DefaultSmoothingOscilationEWMA(_SmoothingOscilationBase):
    """
        Liczy średnią EWMA względem wag.
        Wzór to S = ax + (1-a)S, gdzie 
            a - współczynnik z przedziału [0;1]; movingAvgParam
            x - nowe wartości
            S - suma poprzednich wartości
    """
    def __init__(self, smoothingMetadata, dataType=torch.float32):
        super().__init__(smoothingMetadata=smoothingMetadata)

        self.weightsSum = {}
        self.dataType = dataType
        self.__setMovingAvgParam(value=smoothingMetadata.movingAvgParam, smoothingMetadata=smoothingMetadata)

    def __setMovingAvgParam(self, value, smoothingMetadata):
        if(value >= 1.0 or value <= 0.0):
            raise Exception("Value of {}.movingAvgParam can only be in the range [0; 1]".format(self.__name__))
        
        # a
        self.movingAvgTensorLow = torch.tensor(value).to(smoothingMetadata.device).type(self.dataType)
        # 1 - a
        self.movingAvgTensorHigh = torch.tensor(1 - value).to(smoothingMetadata.device).type(self.dataType)

    def calcMean(self, model, smoothingMetadata):
        super().calcMean(model=model, smoothingMetadata=smoothingMetadata)
        with torch.no_grad():
            for key, val in model.getNNModelModule().state_dict().items():
                valDevice = val.device
                # S = ax + (1-a)S
                self.weightsSum[key] = self.weightsSum[key].mul_(self.movingAvgTensorHigh).add_(
                    (val.type(self.dataType).mul(self.movingAvgTensorLow)).to(smoothingMetadata.device)
                    ).type(self.dataType)

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        tmpCheck = super().__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        if(tmpCheck is not None):
            return tmpCheck # {}


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
                self.weightsSum[key] = torch.clone(values).to(smoothingMetadata.device, dtype=self.dataType).requires_grad_(False)

    def __getstate__(self):
        state = self.__dict__.copy()
        if(self.only_Key_Ingredients):
            del state['countWeights']
            del state['enabled']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.countWeights = 0
            self.enabled = False

    def createDefaultMetadataObj(self):
        return DefaultSmoothingOscilationEWMA_Metadata()

# oscilation weighted mean
smoothingEndCheckTypeDict = [
    'std',
    'wgsum'
]


class DefaultSmoothingOscilationWeightedMean_Metadata(_SmoothingOscilationBase_Metadata):
    def __init__(self, weightIter = None, weightsArraySize=20, smoothingEndCheckType='std',
        device = 'cpu',
        weightSumContainerSize = 10, weightSumContainerSizeStartAt=5, softMarginAdditionalLoops = 20, 
        batchPercentMaxStart = 0.9988, batchPercentMinStart = 0.02, 
        epsilon = 1e-6, hardEpsilon=1e-8, weightsEpsilon = 1e-7,
        lossContainer=50, lossContainerDelayedStartAt = 25):
        """
            weightIter - domyślna wartość DefaultWeightDecay() przy None
        """

        super().__init__(device=device,
        weightSumContainerSize=weightSumContainerSize, weightSumContainerSizeStartAt=weightSumContainerSizeStartAt, 
        softMarginAdditionalLoops=softMarginAdditionalLoops, batchPercentMaxStart=batchPercentMaxStart,
        batchPercentMinStart=batchPercentMinStart, epsilon=epsilon, hardEpsilon=hardEpsilon, weightsEpsilon=weightsEpsilon,
        lossContainer=lossContainer, lossContainerDelayedStartAt=lossContainerDelayedStartAt)

        # jak bardzo następne wagi w kolejce mają stracić na wartości. Kolejne wagi dzieli się przez wielokrotność tej wartości.
        self.weightIter = weightIter if weightIter is not None else DefaultWeightDecay()
        self.weightsArraySize=weightsArraySize
        self.smoothingEndCheckType=smoothingEndCheckType

        # validate
        if(self.smoothingEndCheckType not in smoothingEndCheckTypeDict):
            raise Exception("Unknown type of smoothingEndCheckType: {}\nPossible values:\n{}".format(
                self.smoothingEndCheckType, smoothingEndCheckTypeDict))

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += str(self.weightIter)
        tmp_str += ('Weight array size:\t{}\n'.format(self.weightsArraySize))
        tmp_str += ('Smoothing end type:\t{}\n'.format(self.smoothingEndCheckType))
        return tmp_str

class Test_DefaultSmoothingOscilationWeightedMean_Metadata(DefaultSmoothingOscilationWeightedMean_Metadata):
    def __init__(self, weightIter = None, weightsArraySize=20, smoothingEndCheckType='std',
        device = 'cpu',
        weightSumContainerSize = 10, weightSumContainerSizeStartAt=5, softMarginAdditionalLoops = 3, 
        batchPercentMaxStart = 0.85, batchPercentMinStart = 0.1, 
        epsilon = 1e-4, hardEpsilon=1e-9, weightsEpsilon = 1e-5,
        lossContainer=5, lossContainerDelayedStartAt = 2
    ):
        """
            weightIter - domyślna wartość DefaultWeightDecay() przy None
        """

        super().__init__(weightIter=weightIter, weightsArraySize=weightsArraySize, smoothingEndCheckType=smoothingEndCheckType,
        device=device,
        weightSumContainerSize=weightSumContainerSize, weightSumContainerSizeStartAt=weightSumContainerSizeStartAt, 
        softMarginAdditionalLoops=softMarginAdditionalLoops, batchPercentMaxStart=batchPercentMaxStart,
        batchPercentMinStart=batchPercentMinStart, epsilon=epsilon, hardEpsilon=hardEpsilon, weightsEpsilon=weightsEpsilon,
        lossContainer=lossContainer, lossContainerDelayedStartAt=lossContainerDelayedStartAt)

class DefaultSmoothingOscilationWeightedMean(_SmoothingOscilationBase):
    """
        Włącza wygładzanie, gdy zostaną spełnione określone warunki:
        - po przekroczeniu pewnej minimalnej ilości iteracji pętli oraz 
            gdy różnica pomiędzy średnią N ostatnich strat treningowych modelu, 
            a średnią średnich N ostatnich strat treningowych modelu jest mniejsza niż epsilon.
        - po przekroczeniu pewnej maksymalnej ilości iteracji pętli.

        Liczy średnią ważoną dla wag. Wagi są nadawane względem kolejności ich zapamiętywania. 
        Domyślnie, im starsza, tym posiada mniejszą wagę, jednak można podać swoją własną implementację.
        Podana implementacja zużywa proporcjonalnie tyle pamięci, ile wynosi dla niej parametr weightsArraySize.
        Wymaga to trzymania w buforze zapamiętanych wag modelu.

        Implementacja, ze względu na zapamiętywanie wag, dostarcza innej metody sprawdzenia, czy wygładzanie osiągnęło dostatecznie
        dobry wynik. Porównujemy w niej uśrednione wagi z zapisanymi w buforze wagami, licząc jak bardzo się od siebie różnią.
        Pseudokod:
            diff = []
            for weights in savedWeights:
                diffsum = 0
                for weight in weights:
                    diffsum += sum(abs(weight - avgWeight))
                diff.append(diffsum)
            torch.std(diffsum)
    """
    def __init__(self, smoothingMetadata):
        super().__init__(smoothingMetadata=smoothingMetadata)

        self.weightsArray = sf.CircularList(smoothingMetadata.weightsArraySize) 

        if(smoothingMetadata.smoothingEndCheckType == 'std'):
            self.isSmoothingGoodEnoughMethod = DefaultSmoothingOscilationWeightedMean.__isSmoothingGoodEnough__std
        elif(smoothingMetadata.smoothingEndCheckType == 'wgsum'):
            self.isSmoothingGoodEnoughMethod = _SmoothingOscilationBase.__isSmoothingGoodEnough__
        else:
            raise Exception("Unknown type of smoothingEndCheckType: {}".format(smoothingMetadata.smoothingEndCheckType))
        
    def calcMean(self, model, smoothingMetadata):
        super().calcMean(model=model, smoothingMetadata=smoothingMetadata)
        with torch.no_grad():
            tmpDict = {}
            for key, val in model.getNNModelModule().state_dict().items():
                tmpDict[key] = val.clone().to(smoothingMetadata.device)
            self.weightsArray.pushBack(tmpDict)

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        tmpCheck = super().__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        if(tmpCheck is not None):
            return tmpCheck # {}

        average = {}
        if(self.countWeights == 0):
            return average # {}

        # przygotuj słownik dla wag
        for wg in self.weightsArray:
            for key, val in wg.items():
                average[key] = torch.zeros_like(val, requires_grad=False, device=smoothingMetadata.device)
            break

        # sumuj wagi względem współczynnika
        iterWg = iter(smoothingMetadata.weightIter)
        wgSum = 0.0
        for wg in self.weightsArray:
            weight = next(iterWg)
            for key, val in wg.items(): 
                average[key] += val.mul(weight)
            wgSum += weight 

        # podziel otrzymaną sumę przez całkowitą sumę wag średniej ważonej
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
            del state['weightsArray']
            del state['enabled']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.only_Key_Ingredients):
            self.countWeights = 0
            self.weightsArray.reset()
            self.enabled = False

    def createDefaultMetadataObj(self):
        return DefaultSmoothingOscilationWeightedMean_Metadata()

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        return self.isSmoothingGoodEnoughMethod(self, helperEpoch=helperEpoch, helper=helper, model=model, 
            dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothingMetadata=smoothingMetadata)

    def _sumWeightsToArrayStd(self, smoothedWeights):
        sumOfDiff = []
        for weights in self.weightsArray:
            diffSumDict = 0.0
            for key, weightTens in weights.items():
                diffSumDict += torch.sum(torch.abs(weightTens.sub(smoothedWeights[key])))
            sumOfDiff.append(diffSumDict)
        
        std = torch.std(torch.Tensor(sumOfDiff))
        if(std.isnan()):
            return torch.tensor(ConfigClass.STD_NAN)
        return std

    def __isSmoothingGoodEnough__std(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        if(self.countWeights > 0):
            self.divisionCounter += 1
            smWg = self.__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
            stdDev = self._sumWeightsToArrayStd(smoothedWeights=smWg)

            metadata.stream.print("Standard deviation:" + str(stdDev), 'debug:0')
            metadata.stream.print("Standard deviation bool: " + str(bool(stdDev < smoothingMetadata.weightsEpsilon)), 'debug:0')
            
            return self._smoothingGoodEnoughCheck(val=stdDev, smoothingMetadata=smoothingMetadata)
        return False



# pytorch SWA model implementation

class DefaultPytorchAveragedSmoothing_Metadata(sf.Smoothing_Metadata):
    def __init__(self, device = 'cpu', smoothingStartPercent = 0.8):
        super().__init__()
        self.device = device
        self.smoothingStartPercent = smoothingStartPercent

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Device:\t{}\n'.format(str(self.device)))
        tmp_str += ('Smoothing start percent:\t{}\n'.format(str(self.smoothingStartPercent)))
        return tmp_str

class Test_DefaultPytorchAveragedSmoothing_Metadata(DefaultPytorchAveragedSmoothing_Metadata):
    def __init__(self, device = 'cpu', smoothingStartPercent = 0.8):
        super().__init__(device=device, smoothingStartPercent=smoothingStartPercent)

class DefaultPytorchAveragedSmoothing(sf.Smoothing):
    """
        Algorytm SWA. Korzysta z implementacji pytorcha torch.optim.swa_utils.AveragedModel.
        Wygładzanie jest włączane po określonej liczbie iteracji pętli treningowej.
    """
    def __init__(self, smoothingMetadata, model):
        super().__init__(smoothingMetadata)
        self.swaModel = torch.optim.swa_utils.AveragedModel(model.getNNModelModule())
        self.runSmoothing = False # true dla pierszego udanego __call__
    
    def __call__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, smoothingMetadata, metadata):
        super().__call__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothingMetadata=smoothingMetadata)

        if(helperEpoch.trainTotalNumber > (smoothingMetadata.smoothingStartPercent * helperEpoch.maxTrainTotalNumber)):
            self._firstSetup(helperEpoch=helperEpoch)
            self.swaModel.update_parameters(model.getNNModelModule())
            return True
        return False

    def _firstSetup(self, helperEpoch):
        if(not self.runSmoothing):
            self.runSmoothing = True
            helperEpoch.addSmoothingMode()

    def __isSmoothingGoodEnough__(self, helperEpoch, helper, model, dataMetadata, modelMetadata, metadata, smoothingMetadata):
        return False

    def createDefaultMetadataObj(self):
        return DefaultPytorchAveragedSmoothing_Metadata()

    def getSWAModel(self):
        return self.swaModel

    def __getSmoothedWeights__(self, smoothingMetadata, metadata):
        tmpCheck = super().__getSmoothedWeights__(smoothingMetadata=smoothingMetadata, metadata=metadata)
        if(tmpCheck is not None):
            return tmpCheck

        if(self.runSmoothing):
            return sf.cloneTorchDict(self.swaModel.module.state_dict())
        else:
            return {}


# data classes
class DefaultData_Metadata(sf.Data_Metadata):
    """
        startTestAtEpoch - wartość -1, jeżeli przy każdym epochu należy wywołać test albo lista epochy dla których należy wywołać test.
            Przykład 
            startTestAtEpoch = [*range(3)] # wywoła testy tylko dla pierwszych 3 epochy [0, 1, 2]
            startTestAtEpoch = -1 # inaczej [*range(epoch)]
    """
    def __init__(self, worker_seed = 8418748, download = True, pin_memoryTrain = False, pin_memoryTest = False,
        epoch = 1, batchTrainSize = 16, batchTestSize = 16, startTestAtEpoch=-1, 
        howOftenPrintTrain = 2000, testShuffle=True, trainShuffle=True, 
        trainSampler=None, testSampler=None, trainLoaderWorkers=2, testLoaderWorkers=2,
        transformTrain=None, transformTest=None):

        super().__init__(worker_seed = worker_seed, train = True, download = download, pin_memoryTrain = pin_memoryTrain, pin_memoryTest = pin_memoryTest,
            epoch = epoch, batchTrainSize = batchTrainSize, batchTestSize = batchTestSize, howOftenPrintTrain = howOftenPrintTrain)

        self.trainSampler = trainSampler
        self.testSampler = testSampler
        self.testShuffle = testShuffle
        self.trainShuffle = trainShuffle
        self.trainLoaderWorkers = trainLoaderWorkers
        self.testLoaderWorkers = testLoaderWorkers


        # default values
        self.transformTrain = transformTrain if transformTrain is not None else transforms.Compose([
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
            ])

        self.transformTest = transformTest if transformTest is not None else transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        if(startTestAtEpoch == -1):
            self.startTestAtEpoch = [*range(epoch + 1)]
        else:
            self.startTestAtEpoch = startTestAtEpoch # list of epoches where the test should be called

        # batch size * howOftenPrintTrain
        self.howOftenPrintTrain = howOftenPrintTrain

    def __strAppend__(self):
        tmp_str = super().__strAppend__()
        tmp_str += ('Test shuffle:\t{}\n'.format(self.testShuffle))
        tmp_str += ('Train shuffle:\t{}\n'.format(self.trainShuffle))
        tmp_str += ('Train sampler:\t{}\n'.format(str(self.trainSampler)))
        tmp_str += ('Test sampler:\t{}\n'.format(str(self.testSampler)))
        tmp_str += ('Train workers:\t{}\n'.format(self.trainLoaderWorkers))
        tmp_str += ('Test workers:\t{}\n'.format(self.testLoaderWorkers))
        tmp_str += ('Train transform:\t{}\n'.format(str(self.transformTrain)))
        tmp_str += ('Test transform:\t{}\n'.format(str(self.transformTest)))
        return tmp_str

class DefaultData(sf.Data):
    """
        Domyślna klasa na dane. Jeżeli zabrakło pamięci, należy zwrócić uwagę na rozmiar wejściowego obrazka. Można go zmniejszyć
        uswawiając odpowiedni rozmiar w metadanych dla tej klasy argumentem resizeTo. 
    """
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
        self.transformTrain = dataMetadata.transformTrain
        self.transformTest = dataMetadata.transformTest

    def __update__(self, dataMetadata):
        self.__prepare__(dataMetadata)

    def __beforeTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__beforeTrainLoop__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

    def __beforeTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__beforeTrain__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

    def __afterTrain__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__afterTrain__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

    def __afterTrainLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__afterTrainLoop__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        if(isinstance(smoothing, DefaultPytorchAveragedSmoothing)):
            self.trainHelper.timer.clearTime()
            self.trainHelper.loopTimer.clearTime()

            self.trainHelper.loopTimer.start()
            self.trainHelper.timer.start()
            torch.optim.swa_utils.update_bn(loader=self.trainloader, model=smoothing.getSWAModel(), device=modelMetadata.device)
            self.trainHelper.timer.end()
            self.trainHelper.loopTimer.end()

            self.trainHelper.timer.addToStatistics()
            self.trainHelper.loopTimer.addToStatistics()
        
        with torch.no_grad():
            if(helper.diff is not None):
                diffKey = list(helper.diff.keys())[-1]
                metadata.stream.print("\n\ntrainLoop;\nAverage train time;Loop train time;Weight difference of last layer average;divided by;", ['stat'])
                metadata.stream.print(f"{helper.timer.getAverage()};{helper.loopTimer.getTimeSum()};{helper.diff[diffKey].sum() / helper.diff[diffKey].numel()};{helper.diff[diffKey].numel()}", ['stat'])

    def __beforeTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__beforeTestLoop__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        metadata.stream.print("\n\ntestLoop;\nAverage test time;Loop test time;Accuracy;Avg loss", ['stat'])

    def __beforeTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__beforeTest__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)

    def __afterTest__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__afterTest__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        metadata.stream.print(helper.test_loss, self.testAlias)

    def __afterTestLoop__(self, helperEpoch: 'EpochDataContainer', helper, model: 'Model', dataMetadata: 'Data_Metadata', modelMetadata: 'Model_Metadata', metadata: 'Metadata', smoothing: 'Smoothing', smoothingMetadata: 'Smoothing_Metadata'):
        super().__afterTestLoop__(helperEpoch=helperEpoch, helper=helper, model=model, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
        metadata.stream.print(f"{helper.timer.getAverage()};{helper.loopTimer.getTimeSum()};{(helperEpoch.statistics.correctRatio[-1]):>0.0001f};{helperEpoch.statistics.lossRatio[-1]:>8f}", ['stat'])

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
            if((metadata.shouldTest() and (helperEpoch.epochNumber + 1 in dataMetadata.startTestAtEpoch or helperEpoch.epochNumber + 1 == dataMetadata.epoch) )
                or helperEpoch.endEpoches):
                helperEpoch.currentLoopTimeAlias = 'loopTestTime_normal'
                self.testLoop(model=model, helperEpoch=helperEpoch, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                smoothing.saveWeights(weights=model.getNNModelModule().state_dict().items(), key='main')
                wg = smoothing.__getSmoothedWeights__(metadata=metadata, smoothingMetadata=smoothingMetadata)
                if(wg):
                    sf.Output.printBash('Starting smoothing test at epoch {}.'.format(helperEpoch.epochNumber), 'info')
                    metadata.stream.print("Smoothing test", 'model:0')
                    self.setModelSmoothedWeights(model=model, helperEpoch=helperEpoch, weights=wg, metadata=metadata)
                    helperEpoch.currentLoopTimeAlias = 'loopTestTime_smooothing'
                    self.testAlias = 'statLossTest_smooothing'
                    helperEpoch.averaged = True
                    self.testLoop(model=model, helperEpoch=helperEpoch, dataMetadata=dataMetadata, modelMetadata=modelMetadata, metadata=metadata, smoothing=smoothing, smoothingMetadata=smoothingMetadata)
                    self.setModelNormalWeights(model=model, helperEpoch=helperEpoch, weights=smoothing.getWeights(key='main'), metadata=metadata)
                else:
                    sf.Output.printBash('Smoothing is not enabled at epoch {}. Normal test executed. Smoothing test did not executed.'.format(helperEpoch.epochNumber), 'info')

    def createDefaultMetadataObj(self):
        return DefaultData_Metadata()

    def __prepare__(self, dataMetadata):
        self.__setInputTransform__(dataMetadata)

class DefaultDataMNIST(DefaultData):
    def __init__(self, dataMetadata):
        super().__init__(dataMetadata=dataMetadata)

    def __prepare__(self, dataMetadata):
        super().__prepare__(dataMetadata)

        self.trainset = torchvision.datasets.MNIST(root=sf.StaticData.DATA_PATH, train=True, transform=self.transformTrain, download=dataMetadata.download)
        self.testset = torchvision.datasets.MNIST(root=sf.StaticData.DATA_PATH, train=False, transform=self.transformTest, download=dataMetadata.download)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=dataMetadata.trainSampler,
                                          shuffle=dataMetadata.trainShuffle, num_workers=dataMetadata.trainLoaderWorkers, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=dataMetadata.testSampler,
                                         shuffle=dataMetadata.testShuffle, num_workers=dataMetadata.testLoaderWorkers, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

class DefaultDataEMNIST(DefaultData):
    def __init__(self, dataMetadata):
        super().__init__(dataMetadata=dataMetadata)

    def __prepare__(self, dataMetadata):
        super().__prepare__(dataMetadata)

        self.trainset = torchvision.datasets.EMNIST(root=sf.StaticData.DATA_PATH, train=True, transform=self.transformTrain, download=dataMetadata.download, split='digits')
        self.testset = torchvision.datasets.EMNIST(root=sf.StaticData.DATA_PATH, train=False, transform=self.transformTest, download=dataMetadata.download, split='digits')

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=dataMetadata.trainSampler,
                                          shuffle=dataMetadata.trainShuffle, num_workers=dataMetadata.trainLoaderWorkers, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=dataMetadata.testSampler,
                                         shuffle=dataMetadata.testShuffle, num_workers=dataMetadata.testLoaderWorkers, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

class DefaultDataCIFAR10(DefaultData):
    def __init__(self, dataMetadata):
        super().__init__(dataMetadata=dataMetadata)

    def __prepare__(self, dataMetadata):
        super().__prepare__(dataMetadata)

        self.trainset = torchvision.datasets.CIFAR10(root=sf.StaticData.DATA_PATH, train=True, transform=self.transformTrain, download=dataMetadata.download)
        self.testset = torchvision.datasets.CIFAR10(root=sf.StaticData.DATA_PATH, train=False, transform=self.transformTest, download=dataMetadata.download)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=dataMetadata.trainSampler,
                                          shuffle=dataMetadata.trainShuffle, num_workers=dataMetadata.trainLoaderWorkers, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=dataMetadata.testSampler,
                                         shuffle=dataMetadata.testShuffle, num_workers=dataMetadata.testLoaderWorkers, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

class DefaultDataCIFAR100(DefaultData):
    def __init__(self, dataMetadata):
        super().__init__(dataMetadata=dataMetadata)

    def __prepare__(self, dataMetadata):
        super().__prepare__(dataMetadata)

        self.trainset = torchvision.datasets.CIFAR100(root=sf.StaticData.DATA_PATH, train=True, transform=self.transformTrain, download=dataMetadata.download)
        self.testset = torchvision.datasets.CIFAR100(root=sf.StaticData.DATA_PATH, train=False, transform=self.transformTest, download=dataMetadata.download)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=dataMetadata.batchTrainSize, sampler=dataMetadata.trainSampler,
                                          shuffle=dataMetadata.trainShuffle, num_workers=dataMetadata.trainLoaderWorkers, pin_memory=dataMetadata.pin_memoryTrain, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=dataMetadata.batchTestSize, sampler=dataMetadata.testSampler,
                                         shuffle=dataMetadata.testShuffle, num_workers=dataMetadata.testLoaderWorkers, pin_memory=dataMetadata.pin_memoryTest, worker_init_fn=dataMetadata.worker_seed if sf.enabledDeterminism() else None)


ModelMap = {
    'simpleConvModel': DefaultModelSimpleConv,
    'predefModel': DefaultModelPredef
}

DataMap = {
    'MNIST': DefaultDataMNIST,
    'CIFAR10': DefaultDataCIFAR10,
    'CIFAR100': DefaultDataCIFAR100,
    'EMNIST': DefaultDataEMNIST
}

SmoothingMap = {
    'disabled': DisabledSmoothing,
    'simpleMean': DefaultSmoothingSimpleMean,
    'generalizedMean': DefaultSmoothingOscilationGeneralizedMean,
    'EWMA': DefaultSmoothingOscilationEWMA,
    'weightedMean': DefaultSmoothingOscilationWeightedMean,
    'pytorchSWA' : DefaultPytorchAveragedSmoothing
}

def __checkClassExistence(checkedMap, obj):
    for name, i in checkedMap.items():
        if(isinstance(obj, i)):
            ok = True
            return name

    sf.Output.printBash("Cannot run test, because used unregistered class '{}'. Cannot name folder because of that.\nMap: {}: ".format(str(type(obj)), checkedMap), 
        'warn')
    return None


def run(data, model, smoothing, metadataObj, modelMetadata, dataMetadata, smoothingMetadata, optimizer, lossFunc, schedulers: list=None,
    rootFolder = None, startPrintAt = -10, runningAvgSize=1, logData=None):
    """
        Funckja przygotowuje do wywołania eksperymentu. Na końcu działania funkcja tworzy wykresy.
    """

    dataType = __checkClassExistence(checkedMap=DataMap, obj=data)
    modelType = __checkClassExistence(checkedMap=ModelMap, obj=model)
    smoothingType = __checkClassExistence(checkedMap=SmoothingMap, obj=smoothing)

    metadataObj.resetOutput() 
    modelMetadata.prepare(lossFunc=lossFunc, optimizer=optimizer)
    model.prepare(lossFunc=lossFunc, optimizer=optimizer, schedulers=schedulers)

    logFolderSuffix = modelType + '_' + dataType + '_' + smoothingType

    metadataObj.name = logFolderSuffix
    metadataObj.logFolderSuffix = logFolderSuffix
    metadataObj.relativeRoot = rootFolder

    metadataObj.prepareOutput()
    smoothing.__setDictionary__(smoothingMetadata=smoothingMetadata, dictionary=model.getNNModelModule().state_dict().items())

    metadataObj.printStartNewModel()

    statistics = sf.runObjs(metadataObj=metadataObj, dataMetadataObj=dataMetadata, modelMetadataObj=modelMetadata, 
            smoothingMetadataObj=smoothingMetadata, smoothingObj=smoothing, dataObj=data, modelObj=model, 
            folderLogNameSuffix=logFolderSuffix, folderRelativeRoot=rootFolder, logData=logData)

    metadataObj.printEndModel()

    statistics.printPlots(startAt=startPrintAt, runningAvgSize=runningAvgSize)

    return statistics



if(__name__ == '__main__'):
    torch.backends.cudnn.benchmark = True
    obj = models.alexnet(pretrained=True)

    #sf.useDeterministic()
    #sf.modelDetermTest(sf.Metadata, DefaultData_Metadata, DefaultModel_Metadata, DefaultData, VGG16Model, DefaultSmoothingSimpleMean)
    stat = sf.modelRun(sf.Metadata, DefaultData_Metadata, DefaultModel_Metadata, DefaultDataMNIST, DefaultModel, DefaultSmoothingSimpleMean, obj)

    #plt.plot(stat.trainLossArray)
    #plt.xlabel('Train index')
    #plt.ylabel('Loss')
    #plt.show()




