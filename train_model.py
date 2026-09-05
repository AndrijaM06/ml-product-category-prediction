import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = "data/products.csv"
MODEL_PATH = "model/category_model.pkl"


def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df):
    df = df.copy()

    # Uklanjamo redove bez naslova ili bez kategorije
    df = df.dropna(subset=["Product Title", "Category Label"])

    # Standardizujemo naslov
    df["Product Title"] = df["Product Title"].astype(str).str.lower().str.strip()

    # Standardizujemo kategoriju
    df["Category Label"] = df["Category Label"].astype(str).str.strip()

    category_map = {
        "fridge": "Fridges",
        "fridges": "Fridges",
        "fridge freezers": "Fridge Freezers",
        "cpu": "CPUs",
        "cpus": "CPUs",
        "mobile phone": "Mobile Phones",
        "mobile phones": "Mobile Phones",
        "tv": "TVs",
        "tvs": "TVs",
        "digital camera": "Digital Cameras",
        "digital cameras": "Digital Cameras",
        "microwave": "Microwaves",
        "microwaves": "Microwaves",
        "dishwasher": "Dishwashers",
        "dishwashers": "Dishwashers",
        "washing machine": "Washing Machines",
        "washing machines": "Washing Machines",
        "freezer": "Freezers",
        "freezers": "Freezers",
    }
    df["Category Label"] = (
        df["Category Label"].str.lower().map(category_map).fillna(df["Category Label"])
    )
    df["Category Label"] = df["Category Label"].astype("category")

    # Uklanjamo duplikate naslova
    df = df.drop_duplicates(subset=["Product Title"])

    return df


def engineer_features(df):
    df = df.copy()
    df["title_length"] = df["Product Title"].str.len()
    df["word_count"] = df["Product Title"].str.split().str.len()
    df["has_digit"] = df["Product Title"].str.contains(r"\d", regex=True).astype(int)
    df["longest_word_length"] = (
        df["Product Title"].str.split().apply(lambda words: max((len(w) for w in words), default=0))
    )
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("title", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000), "Product Title"),
            ("numeric", MinMaxScaler(), ["title_length", "word_count", "has_digit", "longest_word_length"]),
        ]
    )


def main():
    print("1. Ucitavanje podataka...")
    df = load_data(DATA_PATH)
    print(f"   Broj redova pre ciscenja: {len(df)}")

    print("2. Ciscenje i standardizacija podataka...")
    df = clean_data(df)
    print(f"   Broj redova posle ciscenja: {len(df)}")

    print("3. Inzenjering karakteristika...")
    df = engineer_features(df)

    feature_columns = ["Product Title", "title_length", "word_count", "has_digit", "longest_word_length"]
    X = df[feature_columns]
    y = df["Category Label"]

    print("4. Kreiranje pipeline-a...")
    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LinearSVC()),
    ])

    print("5. Treniranje finalnog modela na celom skupu podataka...")
    pipeline.fit(X, y)

    print("6. Cuvanje modela...")
    joblib.dump(pipeline, MODEL_PATH)
    print(f"   Model sacuvan u '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
