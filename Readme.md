Framework do testowania wygładzania wag modelu

Na każdym poziomie folderów znajduje się osobny plik Readme wyjaśniający cel zawartych plików.
Obecny folder jest folderem nadrzędnym. Po uruchomieniu skryptu znajdą się w nim
* savedBash - plik służący do zapamiętania aktualnego stanu programu przy jego przerwaniu. Pojawia się przy wywołaniu smoothing/tasks.sh. Zawiera pojedynczą liczbę wskazującą na komendę, która powinna zostać wywołana. Obecnie nie jest używany z powodu braku zaimplementowania danej funkcjonalności.
* savedLogs - folder w którym zostaną zapamiętane logi wywołania programu. Posiada on hierarchiczną strukturę.
* bash.log - przekierowane standardowe wyjście wywoływanego skryptu smoothing/tasks.sh.

Wymaga się zainstalowania poniższych bibliotek:
* pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
* pip install pandas
* python -m pip install matplotlib


Istnieją 2 sposoby wywołania programu:
1. Uruchomienie przygotowanego przez użytkownika skryptu. <br>
    python smoothing/experiments/some_experiment.py
2. Uruchomienie linuksowego skryptu tasks.sh, który można modyfikować i zawrzeć kolejne instrukcje wywołań. Taki sposób uruchomienia zapewnia wywołanie kolejnych skryptów z kolejki nawet, jeżeli któryś z nich będzie posiadał krytyczny błąd.
Możliwe też jest przekazanie argumentów wywołań ze skryptu do każdego wywołania zawartych w nim eksperymentów. Przykład:
    ./smoothing/tasks.sh --debug
    ./smoothing/tasks.sh --test
Flagi dodane do skryptu bash zostaną przekazane do każdego wywołanego wewnętrznie skryptu.

UWAGA
W zależności od miejsca wywołania skryptu savedLogs pojawią się w katalogu wywołania skryptu.<br>
Domyślne wywołanie ./tasks.sh zakłada posiadanie przynajmniej jednej karty graficznej, aby można było je wykonać. 
W przeciwnym wypadku należy zmienić 'cuda:0' na 'cpu'. <br>
Jednocześnie domyślne wywołania w skrypcie ./tasks.sh nie są tymi, któ©e zostały użyte przy wykonywaniu ekperymentów.
Zmniejszono ich zajętość pamięciową po to, aby mogły zmieścić się na karcie graficznej z 4GB pamięci.




Program wymaga stworzenia w katalogu domowym folderu dataSmoothing z uwagi na konieczność pobrania oraz zapisywania wag stworzonych modelów.
Zaleca się stworzenie dla tego folderu dowiązanie symboliczne lub inne podobne działanie w celu wybrania odpowiednio dużego nośnika na zapis.
Jeden model potrafi ważyć ponad 0.5 GB, a dane treningowe oraz walidacyjne od 100 MB do 2 GB.
Domyślnie, jeżeli użytkownik nie posiada tego folderu, zostanie on stworzony automatycznie w jego katalogu domowym.


Aby zobaczyć implementacje modeli w tym ich warstwy, należy wywołać skrypt z danym modelem, który wypisze w konsoli schemat modelu.
Przykład wywołania: <br>
&emsp;python smoothing/framework/models/wideResNet.py

<br>

Uruchomienie testów:
1. wejście do folderu smoothing <br>
2. wywołanie <br>
    &emsp;cd smoothing/ <br>
    &emsp;python -m unittest invokeTests.py

Aby usunąć logi powstałe na wskutek wykonania testu, należy wykonać: <br>
&emsp;python smoothing/framework/test/removeDump.py


<br> <br>
Inne uwagi
Dla pytorcha w wersji 1.9.0+cu102 występują następujące UserWarning / Warning:
+ torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable... (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.) - jest to wewnętrzny warning pytorcha.
+ torch/nn/functional.py:718: UserWarning: Named tensors (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.) - jest to wewnętrzny warning pytorcha.
+ [W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool) - pojawia się tylko w sytuacji, gdy pin_memory dla torch.utils.data.DataLoader jest ustawiony na True oraz gdy num_workers > 0. Aby go zlikwidować należy ustawić pin_memoryTest oraz pin_memoryTrain na False. Wątek o tym temacie: https://github.com/pytorch/pytorch/issues/57273

<br>
Dla pytorcha w wersji 1.8.1+cu102 nie pojawiają się żadne UserWarning.