

Strojenie logiki klas odbywa się poprzez dziedziczenie określonych i nadpisywanie / dodawanie zmiennych oraz metod.

Klasy można podzielić na 2 rodzaje:
* metadane - trzymają w sobie dane, używane przez klasy implementacyjne. Służą do oddzielenia danych od kodu. Posiadają na końcu w nazwie klasy dopisek '_Metadata'. Mogą trzymać one również hiperparametry.
* klasy implementacyjne - implementują logikę działania. Posiadają metody, które można lub należy przeciążać. Przykładowo plik defaultClasses.py implementuje domyślną logikę wywołań.

Wyjątkami tej regułu są klasy:
* Metadata - trzyma informacje o tym, jak ma się zachowywać cały program. Pobiera informacje z argumentów linii poleceń podawanych przy wywołaniu programu.
* Smoothing - nie posiada swojej klasy metadanych.