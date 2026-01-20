# Script pentru antrenarea modelului de clasificare produse
# Pe baza titlului produsului se prezice categoria

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# Cale catre dataset
DATA_PATH = "data/products.csv"

# Cale catre modelul final
MODEL_PATH = "models/product_category_model.pkl"


def load_and_clean_data(path):
    # Incarcam datele
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Standardizam etichetele
    LABEL_FIX = {
        "fridge": "Fridges",
        "CPU": "CPUs",
        "Mobile Phone": "Mobile Phones",
    }

    # Curatare titlu produs
    df["Product Title"] = (
        df["Product Title"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    df = df[df["Product Title"].str.len() > 0]

    # Curatare label
    df["Category Label"] = df["Category Label"].astype("string").str.strip()
    df["Category Label"] = df["Category Label"].replace(LABEL_FIX)
    df = df[df["Category Label"].notna()]

    return df


def main():
    df = load_and_clean_data(DATA_PATH)

    X = df["Product Title"]
    y = df["Category Label"]

    # Split stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Definim cele doua modele testate
    model_lr = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
       ("clf", LogisticRegression(
           max_iter=2000, 
           class_weight="balanced", 
           random_state=42
           ))

    ])

    model_svc = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ("clf", LinearSVC(
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Antrenare
    model_lr.fit(X_train, y_train)
    model_svc.fit(X_train, y_train)

    # Evaluare
    pred_lr = model_lr.predict(X_test)
    pred_svc = model_svc.predict(X_test)

    acc_lr = accuracy_score(y_test, pred_lr)
    acc_svc = accuracy_score(y_test, pred_svc)

    print("Logistic Regression accuracy:", acc_lr)
    print("LinearSVC accuracy:", acc_svc)

    # Alegem modelul final
    if acc_svc >= acc_lr:
        final_model = model_svc
        print("\nModel ales: LinearSVC")
    else:
        final_model = model_lr
        print("\nModel ales: Logistic Regression")

    # Salvam modelul final
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    print("\nModel salvat in:", MODEL_PATH)
    print("\nClassification report final:\n")
    print(classification_report(y_test, final_model.predict(X_test)))


if __name__ == "__main__":
    main()
