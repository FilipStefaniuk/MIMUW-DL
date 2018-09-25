Głębokie Sieci Neuronowe: Task 1
Autor: Filip Stefaniuk
Nr Indeksu: 361039
-------------------------------------------------

Model który trenowałem oparty jest na architekturze sieci VGG,
do trenowania używałem RMSProp.

Program składa się z następujących elementów:
    - model.py - buduje model
    - batch_normalization.py - moja implementacja batch normalizacji
    - trainer.py - trenuje model
    - main.py - uruchamia program (trenowanie)
    - visualizer.py - tworzy wizualizacje do 2 części zadania
    - Filter Visualization.ipynb - wyświetla wizualizacje

W pliku environment.yml jest środowisko z anacondy w jakim pracowałem.

Program uruchamia się poleceniem:

$ python ./main.py

Wypróbowałem wersje z i bez dropoutu i batch normalizacji.
Logi są w folderze logs i można je obejrzeć w tensorboard.

W najlepszej wersji model osiąga accuracy ~99.3%

Nie wysyłam wytrenowanych modeli bo zajmują za dużo miejsca,
trenowanie nie trwa długo, na GPU ok 2-3 min