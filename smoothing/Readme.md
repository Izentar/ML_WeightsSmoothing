
Istnieją 2 sposoby wywołania programu:
1. Wpisanie w konsolę bezpośrednio, jak np. <br>
    python smoothing/alexnet_pretrain.py -l alexnet_pretrain_load -s alexnet_pretrain_save --mname alexnet_pretrain --test true --train true --pinTest false -d --debugOutput debug --modelOutput alexnet_pretrain_model --bashOutput true --formatedOutput alexnet_pretrain_formated 
2. Uruchomienie linuksowego skryptu tasks.sh, w którym można zawrzeć kolejne instrukcje wywołań. Taki sposób uruchomienia zapewnia wywołanie kolejnych skryptów z kolejki nawet, jeżeli któryś z nich będzie posiadał krytyczny błąd.

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

<br>

Uruchomienie testów:
1. wejście do folderu smoothing <br>
2. wywołanie <br>
    &emsp;cd smoothing/
    &emsp;python -m unittest invokeTests.py

Aby usunąć logi powstałe na wskutek wykonania testu, należy wykonać: python smoothing/framework/test/removeDump.py