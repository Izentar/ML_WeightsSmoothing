Folder ten służy do tworzenia skryptów dla wykonywanych eksperymentów.
Dla poprawnego działania należy na samym początku zaimportować 
    import experiments
która zmienia ścieżkę systemową wywołanego skryptu na folder 'smoothing/'.

Użytkownik może wywołać domyślny skrypt smoothing/experiments/exp_pytorch.py, który pozwala na przekazanie argumentów wywołania w linii poleceń. Zawiera on wszystkie konfiguracje potrzebne do wykonania eksperymentów. Jednocześnie użytkownik może stworzyć swój własny skrypt korzystając z dostępnych implementacji lub pisząc własne klasy dziedziczące po domyśłnych klasach frameworka.

Skrypt może się nie wykonać z różnych, czasem niezależnych od użytkownika przyczyn, dlatego aby nie wstrzymywać dalszych eksperymentów ze skryptu zaleca się również, aby wywoływane eksperymenty zamykać w bloku 'try except'.