Folder ten służy do tworzenia skryptów dla wykonywanych eksperymentów.
Dla poprawnego działania należy na samym początku zaimportować 
import experiments
która zmienia ścieżkę systemową wywołanego skryptu na folder 'smoothing/'.

Skrypt może się nie wykonać z różnych, czasem niezależnych od użytkownika przyczyn, dlatego aby nie wstrzymywać dalszych eksperymentów ze skryptu zaleca się również, aby wywoływane eksperymenty zamykać w bloku 'try except'.