# Predikcija kategorije proizvoda na osnovu naslova

Model mašinskog učenja koji na osnovu naziva proizvoda (Product Title) automatski
predviđa odgovarajuću kategoriju proizvoda (npr. Mobile Phones, TVs, Fridges...).

## Kontekst projekta

Zadatak simulira realan poslovni problem online trgovine koja svakodnevno unosi
hiljade novih proizvoda. Ručna kategorizacija je spora i podložna greškama, pa je
cilj bio razviti ML model koji automatski predlaže kategoriju na osnovu naziva
proizvoda.


## Tok rada (workflow)

1. **Istraživačka analiza podataka (EDA)** – pregled strukture podataka, provera
   nedostajućih vrednosti i analiza distribucije kategorija.
2. **Čišćenje i standardizacija podataka** – uklanjanje redova bez naslova/kategorije,
   standardizacija teksta i mapiranje dupliranih naziva kategorija na jedinstven oblik
   (npr. `fridge` → `Fridges`, `CPU` → `CPUs`).
3. **Inženjering karakteristika** – izvedene nove karakteristike iz naslova:
   `title_length`, `word_count`, `has_digit`, `longest_word_length`.
4. **Poređenje algoritama** – testirani Logistic Regression, Naive Bayes,
   Random Forest i Support Vector Machine. Najbolje rezultate postigao je
   **Support Vector Machine (LinearSVC)**.
5. **Treniranje i čuvanje finalnog modela** – finalni model treniran na celom
   skupu podataka i sačuvan u `model/category_model.pkl` pomoću `joblib`.

Detaljna analiza, kod i vizualizacije nalaze se u
`product_category_analysis.ipynb`.

## Kako pokrenuti projekat

### 1. Instalacija zavisnosti

```bash
python -m pip install pandas scikit-learn joblib
```

### 2. Treniranje modela

```bash
python train_model.py
```

Ova skripta učitava `data/products.csv`, čisti podatke, generiše karakteristike,
trenira model i čuva ga u `model/category_model.pkl`.

### 3. Testiranje modela

```bash
python predict_category.py
```

Skripta učitava sačuvan model i omogućava interaktivno testiranje – korisnik
unosi naziv proizvoda, a model vraća predviđenu kategoriju. Za izlazak, ukucaj `exit`.


## Rezultati modela

Model je evaluiran pomoću accuracy, precision, recall i F1 metrike, kao i
matrice zabune (detalji u notebook-u). Support Vector Machine je dao najbolji
balans preciznosti i F1 skora u odnosu na ostale testirane algoritme.

## Poznato ograničenje modela

Model ponekad meša kategorije **Fridges** i **Fridge Freezers** su predviđeni kao `Fridges` 
umesto `Fridge Freezers`). Ovo je očekivano jer su te dve kategorije semantički veoma
slične, a naslovi retko eksplicitno navode da li je uređaj samo frižider ili
kombinovani frižider-zamrzivač. Ovo ograničenje bi se moglo dodatno rešiti sa
dodatnim karakteristikama izvedenim iz šifri modela proizvoda ili sa više
podataka za te kategorije.
