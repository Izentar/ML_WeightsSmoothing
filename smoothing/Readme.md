
W podanym folderze znajdują się:
* experiments - folder służący do przechowywania skryptów wykonujących określone ekperymenty.
* framework - znajduje się tam główna logika programu.
* averageAgain.py - skrypt służący do uśredniania logów. W linii poleceń przekazuje się ścieżkę do zapisanych obiektów typu 'Statistics'. Logi wywołania znajdą się w folderze savedLogs pod nazwą 'custom_avg_*'.
* invokeTests.py - wykonuje testy zawarte w framework/test. Testy napisane są za pomocą unittest.
