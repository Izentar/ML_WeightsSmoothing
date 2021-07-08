
Istnieją 2 sposoby wywołania programu:
1. Uruchomienie przygotowanego przez użytkownika skryptu. <br>
    python smoothing/experiments/some_experiment.py
2. Uruchomienie linuksowego skryptu tasks.sh, który można modyfikować i zawrzeć kolejne instrukcje wywołań. Taki sposób uruchomienia zapewnia wywołanie kolejnych skryptów z kolejki nawet, jeżeli któryś z nich będzie posiadał krytyczny błąd.
Możliwe też jest przekazanie argumentów wywołań ze skryptu do każdego wywołania zawartych w nim eksperymentów. Przykład
    ./smoothing/tasks.sh debug
Obecnie domyślnie jedynie istnieje możliwość przekazania flagi argumentu debug do skryptu pythona aby uruchomić eksperymenty w trybie testowym.

Użytkownik musi zdefiniować swoje własne skrypty pythona, które korzystają z danego frameworka. Możliwe jest wykorzystanie domyślnych implementacji korzystających z frameworka. 
Strojenie jakichkolwiek parametrów klas odbywa się poprzez dziedziczenie określonych i nadpisywanie / dodawanie zmiennych do klasy w funckji __init__ bądź innych, według preferencji użytkownika. Mnogość, różnorodność oraz dowolność implementacji skłania do wyboru takiego rozwiązania.

Klasy można podzielić na 2 rodzaje:
* metadane - trzymają w sobie dane, używane przez klasy implementacyjne. Służą do oddzielenia danych od kodu. Posiadają na końcu w nazwie klasy dopisek '_Metadata'. Mogą trzymać one również hiperparametry.
* klasy implementacyjne - implementują logikę działania. Posiadają metody, które można lub należy przeciążać. Przykładowo plik defaultClasses.py implementuje domyślną logikę wywołań.

Wyjątkami tej regułu są klasy:
* Metadata - trzyma informacje o tym, jak ma się zachowywać cały program. Pobiera informacje z argumentów linii poleceń podawanych przy wywołaniu programu.
* Smoothing - nie posiada swojej klasy metadanych.


Program wymaga stworzenia w katalogu domowym folderu .data z uwagi na konieczność pobrania oraz zapisywania wag stworzonych modelów.
Zaleca się stworzenie dla tego folderu dowiązanie symboliczne lub inne podobne działanie w celu wybrania odpowiednio dużego nośnika na zapis.
Jeden model potrafi ważyć ponad 0.5 GB, a dane treningowe oraz walidacyjne od 100 MB do 2 GB.
Domyślnie, jeżeli użytkownik nie posiada tego folderu, zostanie on stworzony automatycznie w jego katalogu domowym.

<br>

Uruchomienie testów:
1. wejście do folderu smoothing <br>
2. wywołanie <br>
    &emsp;cd smoothing/ <br>
    &emsp;python -m unittest invokeTests.py

Aby usunąć logi powstałe na wskutek wykonania testu, należy wykonać: python smoothing/framework/test/removeDump.py
<br>
<br>
Inne uwagi
Dla pytorcha w wersji 1.9.0+cu102 występują następujące UserWarning / Warning:
+ torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable... (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.) - jest to wewnętrzny warning pytorcha.
+ torch/nn/functional.py:718: UserWarning: Named tensors (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.) - jest to wewnętrzny warning pytorcha.
+ [W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool) - pojawia się tylko w sytuacji, gdy pin_memory dla torch.utils.data.DataLoader jest ustawiony na True oraz gdy num_workers > 0. Aby go zlikwidować należy ustawić pin_memoryTest oraz pin_memoryTrain na False. Wątek o tym temacie: https://github.com/pytorch/pytorch/issues/57273

<br>
Dla pytorcha w wersji 1.8.1+cu102 nie pojawiają się żadne UserWarning.