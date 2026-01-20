# Script pentru testare interactiva a modelului antrenat
# Utilizatorul introduce titlul produsului si primeste categoria prezisa

import joblib

MODEL_PATH = "models/product_category_model.pkl"


def main():
    model = joblib.load(MODEL_PATH)

    print("Introdu titlul produsului (ENTER pentru iesire):")

    while True:
        title = input("\nTitlu produs: ").strip()

        if title == "":
            print("Iesire.")
            break

        prediction = model.predict([title.lower()])[0]
        print("Categoria prezisa:", prediction)


if __name__ == "__main__":
    main()
