# product_category_classifier
Product Category Classifier – Machine Learning Project

Acest proiect prezinta dezvoltarea unui sistem de clasificare automata a produselor pe categorii, folosind tehnici de machine learning si procesare a limbajului natural. Scopul principal al aplicatiei este de a automatiza procesul de incadrare a produselor intr-o platforma de e-commerce, pe baza titlului produsului introdus de utilizator.

In mediul real al comertului online, zilnic sunt adaugate un numar foarte mare de produse noi. Clasificarea manuala a acestora consuma timp, este predispusa la erori si incetineste publicarea produselor pe platforma. Prin introducerea unui sistem automat de clasificare, echipa operationala este eliberata de munca manuala repetitiva, iar produsele pot fi incadrate corect si rapid, imbunatatind atat fluxurile interne, cat si experienta utilizatorilor finali.

Setul de date utilizat in acest proiect contine peste 30.000 de produse si include informatii precum titlul produsului, categoria asociata si alte date comerciale. Pentru a simula un scenariu real de utilizare, modelul dezvoltat se bazeaza exclusiv pe titlul produsului, presupunand ca acesta este disponibil imediat in momentul listarii.

Proiectul este structurat astfel incat sa reflecte un flux complet de dezvoltare a unei solutii de machine learning. In prima etapa, datele au fost analizate si curatate pentru a asigura consistenta si calitatea informatiilor. Titlurile produselor au fost standardizate, valorile lipsa au fost eliminate, iar etichetele de categorie au fost uniformizate pentru a evita ambiguitatile.

Ulterior, a fost construit un model de baza folosind TF-IDF pentru vectorizarea textului si Logistic Regression pentru clasificare. Acest model a servit drept punct de referinta pentru evaluarea performantelor ulterioare. In continuare, a fost comparat cu un model bazat pe TF-IDF si LinearSVC, care s-a dovedit a avea o performanta superioara pentru acest tip de date textuale.

Pentru a imbunatati intelegerea comportamentului modelului, au fost rulate si experimente suplimentare. Acestea au inclus utilizarea CountVectorizer in loc de TF-IDF, precum si adaugarea unor caracteristici numerice simple extrase din titlul produsului, cum ar fi lungimea titlului, numarul de cuvinte, prezenta cifrelor si detectarea unor branduri cunoscute. Desi varianta cu feature engineering a obtinut cea mai buna acuratete, imbunatatirea fata de modelul TF-IDF si LinearSVC a fost relativ mica, iar complexitatea suplimentara nu a justificat schimbarea solutiei finale.

Modelul ales pentru utilizare este bazat pe TF-IDF si LinearSVC, oferind un echilibru bun intre performanta, simplitate si mentenanta. Acesta este salvat ca un pipeline complet in fisierul product_category_model.pkl, permitand reutilizarea si integrarea facila intr-un sistem mai larg.

Pentru a facilita utilizarea si reantrenarea modelului, proiectul include doua scripturi Python. Scriptul train_model.py permite antrenarea modelului pe setul de date si salvarea acestuia, in timp ce scriptul predict_category.py ofera o interfata simpla pentru testarea interactiva a predictiilor, prin introducerea unui titlu de produs direct din terminal. Exemple de rulare si testare sunt documentate prin capturi de ecran disponibile in folderul screenshots.

Evaluarea performantei a fost realizata folosind metrici standard de clasificare, precum accuracy, precision, recall si F1-score, precum si prin analiza matricii de confuzie. Rezultatele obtinute demonstreaza ca modelul generalizeaza bine pe date nevazute, cu confuzii minore intre categorii apropiate semantic, un comportament asteptat pentru acest tip de problema.

In concluzie, acest proiect demonstreaza capacitatea de a aborda o problema reala de business folosind machine learning, de la analiza datelor si experimentare, pana la livrarea unei solutii functionale, documentate si usor de extins. Solutia poate fi adaptata in viitor prin adaugarea de date noi, optimizarea caracteristicilor sau integrarea intr-un serviciu API pentru utilizare la scara larga.