import pandas as pd
import joblib


MODEL_PATH = "model/category_model.pkl"


def engineer_features_for_input(title: str) -> pd.DataFrame:
    title_clean = title.lower().strip()

    title_length = len(title_clean)
    word_count = len(title_clean.split())
    has_digit = int(any(ch.isdigit() for ch in title_clean))
    words = title_clean.split()
    longest_word_length = max((len(w) for w in words), default=0)

    return pd.DataFrame([{
        "Product Title": title_clean,
        "title_length": title_length,
        "word_count": word_count,
        "has_digit": has_digit,
        "longest_word_length": longest_word_length,
    }])


def main():
    print("Ucitavanje modela...")
    model = joblib.load(MODEL_PATH)
    print("Model uspesno ucitan!")
    print("Unesi naziv proizvoda da dobijes predvidjenu kategoriju.")
    print("Otkucaj 'exit' za izlaz.\n")

    while True:
        title = input("Naziv proizvoda: ")
        if title.lower() == "exit":
            print("Izlazak iz programa...")
            break

        user_input = engineer_features_for_input(title)
        prediction = model.predict(user_input)[0]

        print(f"Predvidjena kategorija: {prediction}\n" + "-" * 40)


if __name__ == "__main__":
    main()
