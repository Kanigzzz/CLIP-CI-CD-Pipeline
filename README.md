### Własna implementacja CLIP-a zamiast gotowego modelu

**Własna implementacja modelu typu CLIP zamiast gotowego modelu `openai/clip-vit-base-patch32`**

**Kontekst.** Zamiast sięgnąć po gotowy `openai/clip-vit-base-patch32`,
zaimplementowałem własny dwuwieżowy model multimodalny.

**Decyzja i implementacja.** Własna implementacja daje pełną kontrolę nad
każdym elementem: osobną konfigurację obu wież, niezależny dobór strategii
fine-tuningu dla każdej z nich, wymiar wektorów wyjściowych oraz własną
implementację funkcji straty. Gotowy model jest czarną skrzynką — tutaj każdy
element mogłem zmienić i od razu zobaczyć skutek.

Model składa się z dwóch wież:

- **wizyjna** — `efficientnet_b0` z `torchvision`, zamrożony backbone,
  głowica projekcyjna do 256 wymiarów
- **tekstowa** — `distilbert-base-uncased` z `transformers`, mean pooling
  z maską, analogiczna głowica projekcyjna

Obie rzutują na wspólną przestrzeń 256-wymiarową znormalizowaną L2, uczoną
zmodyfikowaną kontrastową funkcją straty.

**Koszt.** Niepodważalnym kosztem tej decyzji jest słabsza jakość dopasowań.
Model wytrenowany na 5 400 zdjęciach jest nieporównywalny z CLIP-em trenowanym
na 400 mln par obraz–tekst — to różnica rzędu 74 000×. Autorski model działa
prawidłowo w obrębie domeny, na której był trenowany; zapytanie spoza niej
skutkuje bardzo słabym dopasowaniem, w praktyce bliskim losowemu. Doliczyć
trzeba też czas treningu, ewaluacji i samodzielnego eksportu do ONNX, którego
gotowe rozwiązanie by nie wymagało.

**Wniosek.** Samodzielna implementacja modelu typu CLIP to decyzja podyktowana
przede wszystkim wartością edukacyjną. W projekcie produkcyjnym byłaby błędem —
ale gdy celem jest zrozumienie architektury, a dopiero potem jej wdrożenie,
zwrot z niej okazał się bardzo konkretny. To dzięki niej wiedziałem, z czego
wynika błąd eksportu BLIP-a: zrozumienie kształtu obliczeń własnego modelu
przełożyło się bezpośrednio na umiejętność zdiagnozowania problemu z innym modelem.

### Porzucenie eksportu BLIP do ONNX

**Eksportowalność do ONNX nie jest cechą modelu, tylko cechą kształtu jego obliczeń.**

**Kontekst.** Autorski model CLIP wyeksportowałem przez `torch.onnx.export` bez
problemów. Założyłem, że z BLIP-em (Bootstrapping Language-Image Pre-training)
pójdzie tak samo — i zbudowałem na nim endpoint `/caption`, zanim sprawdziłem,
czy w ogóle da się go wyeksportować.

**Na czym polegał problem.** `torch.onnx.export` działa metodą *tracingu*:
przepuszcza przez model przykładowy tensor i nagrywa wykonane operacje.
Zapisuje więc jedną, statyczną ścieżkę obliczeń.

Wieża tekstowa CLIP-a to pojedynczy forward pass przez statyczny enkoder —
tracing zapisuje ją w całości i eksport przechodzi za pierwszym razem.

BLIP to enkoder wizyjny połączony z dekoderem-transformerem, a generowanie
opisu odbywa się w `model.generate()`, wewnątrz pętli `while`. Liczba iteracji
zależy od długości generowanego tekstu, więc nie ma jednej ścieżki do nagrania:

    torch.onnx.errors.SymbolicValueError: Dynamic control flow (Python loops /
    if-else) is not supported during ONNX tracing.

Drugi problem to KV-cache. Żeby dekoder nie liczył wektorów całego zdania od
nowa przy każdym kolejnym tokenie, transformer trzyma podręczną pamięć
mechanizmu attention. Ten cache **rośnie z każdą iteracją**, a pojedynczy graf
ONNX ma statyczne kształty wejść i wyjść — nie ma jak tego wyrazić:

    RuntimeError: Tracer cannot trace python objects of type tuple of tuples
    containing dynamic past_key_values.

Optimum obsługuje jedynie architektury, dla których ma zdefiniowaną
konfigurację ONNX. `Salesforce/blip-image-captioning-base` ma typ architektury
`blip`, którego na tej liście nie ma — w wersji `optimum==2.1.0` biblioteka nie
zawiera mapy węzłów obliczeniowych dla tej architektury, więc eksport przerywa
się, zanim w ogóle zacznie:

    ValueError: Asking to export an unsupported architecture: blip.
    No ONNX configuration found for blip in transformers/optimum.

Zostały więc dwie drogi: napisać własną konfigurację ONNX dla tej architektury
albo zmienić model na wspierany.


**Decyzja.** Nie wybrałem żadnej z dwóch opcji. Pisanie własnej konfiguracji
ONNX wykracza poza zakres tego projektu, a zmiana modelu oznaczałaby ponowny
dobór, konfigurację i weryfikację jakości generowanych opisów — przy endpoincie
pobocznym, podczas gdy główna ścieżka (wyszukiwarka) ma już pełną obsługę ONNX
z kwantyzacją. Zamroziłem zakres i zapisałem to jako znany dług. Ścieżka jest
opisana wyżej i wraca jako możliwe rozszerzenie — razem z redukcją obrazu
Dockera, która byłaby jej bezpośrednią konsekwencją.

**Koszt.** BLIP został w PyTorch, więc `torch` i `torchvision` zostały
w `requirements.txt` — mimo że wyszukiwarka ich nie potrzebuje, bo działa na
`onnxruntime`. Warstwa `uv pip install` waży przez to 5,06 GB, przy 151 kB
kodu aplikacji. To bezpośredni, mierzalny koszt niedokończonej migracji.

**Wniosek.**

Techniczny — eksportowalność modelu do formatu ONNX zależy od dwóch rzeczy
naraz: od kształtu obliczeń (czy da się je opisać jednym statycznym grafem)
oraz od tego, czy dana architektura ma gotową konfigurację w narzędziach
eksportu. Pierwsze wymaga zrozumienia architektury modelu i uprzedniej
weryfikacji planu wdrożenia. Drugie sprowadza się do sprawdzenia jednej pozycji
na liście wspieranych architektur.

Procesowy — **wybór modelu jest decyzją wdrożeniową, nie tylko jakościową.**
Sprawdzenie, czy architektura jest wspierana przez narzędzie eksportu, trwa
pięć minut i należy do etapu wyboru modelu — nie do momentu, w którym stoi już
na nim API i przychodzi czas na migrację do ONNX. Dziś przy wyborze modelu
patrzę na trzy rzeczy naraz: jakość, rozmiar i wsparcie w narzędziach eksportu.