
# Smoothing framework

Framework do wykonywania eksperymentów dotyczących wygładzania wag modelu<br>
Został przetestowany na platformie Ubuntu 18.04 oraz Windows 10. Wykorzystywana wersja python - 3.6.9 64 bitowa.

Obecny folder jest folderem nadrzędnym. Po uruchomieniu skryptu znajdą się w nim
* savedBash - plik służący do zapamiętania aktualnego stanu programu przy jego przerwaniu. Pojawia się przy wywołaniu tasks.sh. Zawiera pojedynczą liczbę wskazującą na komendę, która powinna zostać wywołana. Obecnie nie jest używany z powodu braku zaimplementowania danej funkcjonalności.
* savedLogs - folder w którym zostaną zapamiętane logi wywołania programu. Posiada on hierarchiczną strukturę. Jego położenie zależne jest od miejsca wywołania skryptu, gdyż posługuje się katalogiem roboczym. 
* bash.log - przekierowane standardowe wyjście wywoływanego skryptu tasks.sh.

## Wymagania 

Wymaga się zainstalowania poniższych bibliotek:
* pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
* pip install pandas
* python -m pip install matplotlib

Program wymaga stworzenia w katalogu domowym folderu **dataSmoothing** z uwagi na konieczność pobrania oraz zapisywania wag stworzonych modelów.
Zaleca się stworzenie dla tego folderu dowiązanie symboliczne lub inne podobne działanie w celu wybrania odpowiednio dużego nośnika na zapis.
Jeden model potrafi ważyć ponad 0.5 GB, a dane treningowe oraz walidacyjne od 100 MB do 2 GB.
Domyślnie, jeżeli użytkownik nie posiada tego folderu, zostanie on stworzony automatycznie w jego katalogu domowym.

Wymagania przętowe:
* najmniejsze wymagania sprzętowe, to posiadanie jedynie **CPU** oraz **8 GB pamięci RAM**. Mimo to, na takiej konfiguracji program, w zależności od wykonywanego eksperymentu, będzie wykonywał się dniami, dlatego zaleca się wywoływanie ich na karcie **graficznej z 8 GB pamięci**, jak **RTX 2070**.
* domyślnie program będzie uruchamiany na karcie graficznej **cuda:0**, jednak nie jest ona wymagana (ale mocno zalecana). Do skorzystania z CPU należy zmienić parametr **--device** oraz **--smdevice** na **cpu**. Umożliwia się również wywołanie na drugiej karcie graficznej **cuda:1**, o ile ona istnieje.

## Wywołanie programu

Istnieją 2 sposoby wywołania programu:
1. Uruchomienie przygotowanego przez użytkownika skryptu. <br>
    python smoothing/experiments/exp_pytorch.py
2. Uruchomienie linuksowego skryptu tasks.sh, który można modyfikować i zawrzeć kolejne instrukcje wywołań. Taki sposób uruchomienia zapewnia wywołanie kolejnych skryptów z kolejki nawet, jeżeli któryś z nich będzie posiadał krytyczny błąd.
Możliwe też jest przekazanie argumentów wywołań ze skryptu do każdego wywołania zawartych w nim eksperymentów. Przykład:
    ./tasks.sh --test
Flagi dodane do skryptu bash zostaną przekazane do każdego wywołanego wewnętrznie skryptu.

__UWAGA__ <br>
Domyślne wywołanie ./tasks.sh zakłada posiadanie przynajmniej jednej karty graficznej, aby można było je wykonać. 
W przeciwnym wypadku należy zmienić 'cuda:0' na 'cpu'. <br>
Jednocześnie domyślne wywołania w skrypcie ./tasks.sh nie są tymi, któ©e zostały użyte przy wykonywaniu ekperymentów.
Zmniejszono ich wymagania pamięciowe po to, aby mogły zmieścić się na karcie graficznej z 4GB pamięci.
Wszystkich eksperymentów było ponad 40, dlatego argumenty wywołania znajdują się bezpośrednio w logach wywołań w pliku 'model.log'.


Aby zobaczyć implementacje modeli w tym użyte w nim warstwy, należy wywołać skrypt z danym modelem, który wypisze w konsoli schemat modelu. Nie wszystkie warstwy mogą być widoczne, zaleca się ich analizowanie wraz z kodem źródłowym.
Przykład wywołania: <br>
&emsp;python smoothing/framework/models/wideResNet.py

## Testowanie

W folderze smoothing/framework/test znajdują się:
* dump - folder na logi stworzone w czasie testów.
* removeDump.py - usuwa pliki zawarte w folderze dump.
* utils.py - zbiór przydatnych do testowania funkcji.
* reszta plików dotyczy testowania. 

Uruchomienie testów:
1. wejście do folderu smoothing <br>
2. wywołanie <br>
    &emsp;cd smoothing/ <br>
    &emsp;python -m unittest invokeTests.py

Aby usunąć logi powstałe na wskutek wykonania testu, należy wykonać: <br>
&emsp;python smoothing/framework/test/removeDump.py 

## Inne uwagi

Dla pytorcha w wersji 1.9.0+cu102 występują następujące UserWarning / Warning:
+ torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable... (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.) - jest to wewnętrzny warning pytorcha.
+ torch/nn/functional.py:718: UserWarning: Named tensors (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.) - jest to wewnętrzny warning pytorcha.
+ [W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool) - pojawia się tylko w sytuacji, gdy pin_memory dla torch.utils.data.DataLoader jest ustawiony na True oraz gdy num_workers > 0. Aby go zlikwidować należy ustawić pin_memoryTest oraz pin_memoryTrain na False. Wątek o tym temacie: https://github.com/pytorch/pytorch/issues/57273

<br>
Dla pytorcha w wersji 1.8.1+cu102 nie pojawiają się żadne UserWarning.


## Objaśnienia plików

### **smoothing**

W podanym folderze znajdują się:
* experiments - folder służący do przechowywania skryptów wykonujących określone ekperymenty.
* framework - znajduje się tam główna logika programu.
* averageAgain.py - skrypt służący do uśredniania logów. W linii poleceń przekazuje się ścieżkę do zapisanych obiektów typu 'Statistics'. Logi wywołania znajdą się w folderze savedLogs pod nazwą 'custom_avg_*'.
* invokeTests.py - wykonuje testy zawarte w framework/test. Testy napisane są za pomocą **unittest** z wykorzystaniem **pandas**.
* plot.py - skrypt umożliwiający stworzenie wykresu z jednego lub kilku różnych plików źródłowych. Posiada linię poleceń.

### **smoothing/experiments**

Folder ten służy do tworzenia skryptów dla wykonywanych eksperymentów.
Dla poprawnego działania należy na samym początku zaimportować <br>
&emsp;&emsp;import setup; setup.run()<br>
która zmienia ścieżkę systemową wywołanego skryptu na folder 'smoothing/'. Trzeba uważać na moment wywołania setup.run(), gdyż danie go przed importowaniem pozostałych skryptów zmieni dla nich katalog roboczy.

Użytkownik może wywołać domyślny skrypt smoothing/experiments/exp_pytorch.py, który pozwala na przekazanie argumentów wywołania w linii poleceń. Zawiera on wszystkie konfiguracje potrzebne do wykonania eksperymentów. Jednocześnie użytkownik może stworzyć swój własny skrypt korzystając z dostępnych implementacji lub pisząc własne klasy dziedziczące po domyśłnych klasach frameworka.

Skrypt może się nie wykonać z różnych, czasem niezależnych od użytkownika przyczyn, dlatego aby nie wstrzymywać dalszych eksperymentów ze skryptu zaleca się również, aby wywoływane eksperymenty zamykać w bloku 'try except'.

### **smoothing/framework**

Strojenie logiki klas odbywa się poprzez dziedziczenie określonych i nadpisywanie / dodawanie zmiennych oraz metod.

Klasy można podzielić na 2 rodzaje:
* metadane - trzymają w sobie dane, używane przez klasy implementacyjne. Służą do oddzielenia danych od kodu. Posiadają na końcu w nazwie klasy dopisek '_Metadata'. Mogą trzymać one również hiperparametry.
* klasy implementacyjne - implementują logikę działania. Posiadają metody, które można lub należy przeciążać. Przykładowo plik defaultClasses.py implementuje domyślną logikę wywołań.

Wyjątkami tej regułu są klasy:
* Metadata - trzyma informacje o tym, jak ma się zachowywać cały program. Pobiera informacje z argumentów linii poleceń podawanych przy wywołaniu programu.
* Smoothing - nie posiada swojej klasy metadanych.


### **smoothing/framework/models**

Folder ten posiada implementacje modeli. Więcej informacji znajduje się w poszczególnych plikach modeli.<br>

Umożliwia się wywołanie skryptów bezpośrednio, aby wypisać ich strukturę wraz z liczbą parametrów wykorzystywanych w propagacji wstecznej.
