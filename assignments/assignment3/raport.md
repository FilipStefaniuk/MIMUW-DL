###### Filip Stefaniuk fs361039
# Assignment 3

### Dataset
W tym zadaniu zdecydowałem się użyć generatora danych, zamiast przygotowywać dane wcześniej i dzielić je na zbiór treningowy i testowy. Główną zaletą tego podejścia jest elastyczność. Pozwala mi ono na bardzo łatwe zmienianie długości nawasowania.

Jestem świadomy problemu jakie takie podejście stwarza. Ponieważ nie ma danych testowych, na którym można by sprawdzać czy model dobrze się generalizuje, a podczas uczenia teorytycznie może zobaczyć wszystkie możliwe nawiasowania istnieje ryzyko, że model wyuczy się ich "na pamięć". Uważam że przy bardzo krótkich wyrażeniach istotnie może to być problemem, ale ponieważ ilość możliwych inputów rośnie wykładniczo względem długości wejśćia to można założyć że możliwości jest na tyle dużo że nie jest możliwe żeby model zapamiętał je wszystkie.

Kolejnym problemem jest to, że podczas losowania nawiasowania o zadanej długości chciałbym aby każde z możliwych nawiasowań było losowane z takim samym prawdopodobieństwem. Gdyby tak nie było, zbiór wszystkich możliwych nawiasowań generowanych przez generator mógłby się znacznie zmniejszyć i powyższe założenie nie byłoby już prawdziwe.

Do generowania nawiasowań używam faktu, że takie poprawne nawiasowanie można traktować jako nieoetykietowane drzewo. Okazuje się, że losowanie takich drzew z równym prawdopodobieństwem nie jest proste. Jedyną możliwością jest wypisanie wszystkich możliwości i losowanie z takiego zbioru. Tego rozwiązania jednak nie jestem w stanie zastoosować dla dużych wielkości drzew. Napisałem więc generator z heurystyką który przetestowałem empirycznie tak żeby próbki były losowane mniej więcej z takim samym prawdopodobieństwem.


### Funkcja Kosztu
Jako funkcję kosztu przyjąłem mean squared error. Uważam że jest to dobra funkcja dla tego problemu...

### Model

Przeteswowałem następujące modele:
- rnn z 64 neuronami 
- lstm 64 neurony
- lstm 2 layers 64 neurony
- lstm 8 neuronów
- bi lstm 64 neurony

### Wyniki dla stałej długości inputu

### Wyniki dla różnych długośći inputu

### Wnioski