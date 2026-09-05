# CLIP Animal Search

> Wyszukiwanie zdjęć zapytaniem tekstowym oraz automatyczne generowanie opisów -
> Od wytrenowanego modelu do skonteneryzowanej aplikacji.

## Co to jest?

Kompletny pipeline wdrozeniowy dla aplikacji ML. Projekt opiera się na problemach operacyjnych,
jakie stawia przez inzynierem wdrozenie aplikacji z domeny ML - drzewo zaleznosci rzedu kilku GB, koniecznosc 
osobnego wersjonowania artefaktów modelu, zimny start modelu az po specyfike asynchronicznosci w obliczeniach modelu.

Projekt koncentruje się na stronie operacyjnej: ciężkim drzewie zależności sięgającym kilku GB, niezależnym wersjonowaniu artefaktów modeli, 
cold starcie, inferencji oraz specyfice uruchamiania obliczeń ML w środowisku produkcyjnym.

## Szybki start 

```bash
git clone https://github.com/Kanigzzz/CLIP-CI-CD-Pipeline.git
cd CLIP-CI-CD-Pipeline
docker compose up --build
```

Aplikacja: <http://localhost>
API: <http://localhost:8000/docs>

## Dziennik decyzji 

### Ładowanie modeli przy starcie kontenera, nie przy ządaniu

### Wersjonowanie artefaktów: HF Hub z przypiętą rewizją zamiast DVC

### ONNX i kwantyzacja w sciezce wyszukiwania

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


### Kacze typowanie i lekka warstwa API zamiast sformalizowanych protokołów

**W mikroserwisie ML warstwa API powinna być odchudzona – formalizm typowania
nie musi wymuszać importu bibliotek ML.**

**Kontekst.** Routery `/search` i `/caption` wymagają instancji silników
inferencyjnych (`CLIPSearcher`, `ImageCaptioner`). Instancje powstają raz,
w `lifespan` aplikacji, i trafiają do routerów przez `Depends` czytające
`request.app.state`. W architekturze, która stawia na czyste typowanie,
naturalnym posunięciem jest wykorzystanie formalnych interfejsów
(np. `typing.Protocol`) oraz weryfikacja zgodności za pomocą lintera
(np. `mypy`) w pipeline CI.

**Na czym polegał problem.** Routery importowały klasy modeli **wyłącznie po
to, żeby użyć ich jako adnotacji typu** w `Depends`. Adnotacja niepotrzebna
w runtime ciągnęła za sobą cały stos ML, przez co testy jednostkowe API nie
przechodziły nawet fazy zbierania:

    tests/test_api.py → src.api.routers.search → src.inference.clip_search
    ModuleNotFoundError: No module named 'faiss'

Narzut samego importu to 2,3 s i 233 MB RSS na każde uruchomienie testów.
Warto zaznaczyć, czego problemem nie było: wagi modeli nigdy się w testach nie
ładowały. `TestClient(app)` używany bez `with` nie odpala `lifespan`, więc
konstruktory `CLIPSearcher()` i `ImageCaptioner()` nie były wołane. Kosztem był
sam import bibliotek, nie zużycie pamięci przez modele.

Sformalizowanie tego protokołami byłoby przy obecnej architekturze
overengineeringiem: kontrakt jest wąski (dwie metody — `.search()`,
`.generate_caption()`), stabilny i ma jedną implementację produkcyjną.

**Decyzja i implementacja.** Zrezygnowałem z formalizmu na rzecz kaczego
typowania (*duck typing*) oraz wstrzykiwania zależności (`FastAPI Depends`):

- W routerach zastosowałem strażnika `if TYPE_CHECKING:` do importu klas
  modeli, a adnotacje ująłem w cudzysłowy. Cudzysłowy są tu obowiązkowe, nie
  kosmetyczne — Python nie zamienia adnotacji na stringi samoczynnie, więc bez
  nich `def get_searcher(...) -> CLIPSearcher` rzuciłoby `NameError` już przy
  definiowaniu funkcji.
- W `src/api/main.py` sam `TYPE_CHECKING` nie wystarcza, bo klasy są tam
  *wołane*, nie tylko adnotowane. Import wędruje do wnętrza `lifespan`, czyli
  wykonuje się przy starcie serwera, nigdy przy imporcie modułu.
- W testach jednostkowych (`test_api.py`) atrapy `FakeSearcher`
  i `FakeCaptioner` implementują jedynie oczekiwane sygnatury metod i są
  wstrzykiwane przez `app.dependency_overrides`.

FastAPI **próbuje** rozwinąć taką adnotację i przeżywa dzięki dwóm
zabezpieczeniom. `_get_signature()` woła `inspect.signature(call, eval_str=True)`,
łapie `NameError` i schodzi do wariantu zostawiającego adnotacje surowymi
stringami — komentarz w źródle mówi wprost: *„Handle type annotations with
if TYPE_CHECKING, not used by FastAPI"*. Następnie `get_typed_annotation()`
rozwija string przez pydantic'owe `try_eval_type()`, które przy nieznanej
nazwie zwraca nierozwiązany `ForwardRef` zamiast rzucić wyjątkiem. Na końcu
`analyze_param()` widzi `Depends(...)` i nie robi z parametru pola pydantic,
więc `ForwardRef` nigdy nie musi zostać rozwiązany.

Dzięki temu testy jednostkowe API wykonują się błyskawicznie i nie zależą od
`faiss`, `torch` ani `onnxruntime`.

**Koszt.** Nie ma statycznej weryfikacji zgodności atrap z prawdziwymi klasami
— i trzeba to nazwać dokładnie: nie jest to rezygnacja z czegoś, co projekt
miał. W repozytorium nie ma `mypy` ani w `requirements.txt`, ani w konfiguracji
`pyproject.toml`, a `.github/workflows/ci.yml` uruchamia wyłącznie `pytest` —
nie ma w nim kroku type-checkingu. Protokół bez lintera, który go sprawdza,
byłby komentarzem o lepszej składni. Gdyby metoda w modelu
zmieniła sygnaturę (np. argumenty w `.search()`), a atrapa w testach pozostała
stara, testy jednostkowe API pozostałyby zielone. Nie złapałyby tego również
testy integracyjne — `tests/integration/conftest.py` wycina je przez
`collect_ignore_glob`, gdy brakuje `faiss`, `onnxruntime` lub `transformers`,
czyli dokładnie w środowisku, w którym uruchamia się testy jednostkowe. Rozjazd
wyszedłby dopiero przy realnym starcie aplikacji.

Drugim kosztem jest warunek brzegowy: `TYPE_CHECKING` działa tu **wyłącznie dla
parametrów wstrzykiwanych przez `Depends`**. Ta sama adnotacja bez `Depends`
staje się polem do walidacji i psuje się późno — import przechodzi w ciszy,
request zwraca mylące `422 Field required`, a `/openapi.json` wywraca się na
`PydanticUserError: TypeAdapter[...] is not fully defined`. Testy mogą być
zielone przy leżącym `/docs`, więc przy dokładaniu endpointów trzeba o tym
pamiętać. Zweryfikowane na `fastapi 0.136.3` / `pydantic 2.12.5`;
`requirements.txt` nie ma górnego ograniczenia wersji.

**Wniosek.**

- **Architektoniczny (YAGNI):** dopóki interfejs modelu jest wąski i stabilny,
  a w pipeline nie ma type-checkera, narzut utrzymania formalnych protokołów
  przewyższa realne korzyści.
- **Wdrożeniowy:** kluczowym zyskiem jest odcięcie ciężkiego runtime'u ML od
  warstwy webowej.
- **Warunek rewizji:** decyzja przestaje się bronić w momencie dodania `mypy`
  do CI. Wtedy `Protocol` zaczyna realnie pilnować atrap i warto go wprowadzić.
