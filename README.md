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