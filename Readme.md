# AyurHealth – Disease Prediction and Remedies (Flask)

## 1) What this project is
AyurHealth is a small Flask web app that:
- **Predicts a probable disease** from 4 user-entered symptoms using ML
- **Suggests Ayurvedic remedies** for a selected condition from a curated CSV
- **Provides a simple, responsive UI** (Bootstrap + custom CSS)

## 2) How it works (high level)
- The **Symptoms** page posts four symptoms to the backend.
- The backend **trains multiple classifiers** on `Training.csv` (RandomForest, SVC,
  LogisticRegression, GaussianNB), evaluates them on `Testing.csv` and
  selects the model with the best accuracy for prediction.
- The **predicted disease** is displayed and also stored in a local SQLite DB.
- The **Suggester** page looks up `Remedies.csv` and displays the matching row.

## 3) Run locally
**Prerequisites:** Python 3.10+ recommended

### a. Create/activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # macOS/Linux
```

### b. Install dependencies
```bash
pip install -r requirements.txt
```

### c. Start the app
```bash
python AyurHealth.py
```

### d. Open in browser
Navigate to: `http://localhost:3000`

## 4) App navigation
- **Home**      → Overview and quick links
- **Symptoms**  → Enter four symptoms (pretty labels), get ML prediction
- **Suggester** → Select a condition and view remedies from CSV
- **About**     → About page

## 5) Important files and folders

### Core Application Files
- **`AyurHealth.py`**
  - Flask application entry point and routes:
    - `/`           → Home
    - `/symptoms`   → Symptoms page (GET)
    - `/suggester`  → Suggester page (GET)
    - `/about`      → About page (GET)
    - `/s1` (POST)  → Handles symptoms form, predicts disease and renders result
    - `/s2` (POST)  → Handles suggester form and renders remedies
  - The server runs on host `'localhost'`, port `3000` (`debug=True`)

- **`Symptom.py`**
  - ML workflow for disease prediction:
    - Loads `Training.csv` / `Testing.csv`
    - Encodes prognosis labels to integers
    - Trains candidate models: **RandomForest**, **SVC**, **LogisticRegression**, **GaussianNB**
    - Evaluates on `Testing.csv`, selects the best by accuracy
    - Builds a feature vector from the four symptoms and predicts with the best model
    - Logs model accuracies and chosen model
    - Saves each prediction into SQLite (`database.db` → table `SymptomPrediction`)

- **`Suggester.py`**
  - Simple CSV lookup for remedies. Finds the row in `Remedies.csv` whose first cell
    matches the chosen condition and returns the formatted content.

### Templates (Jinja2)
- **`templates/index.html`**      → Home (navbar includes Symptoms and Suggester)
- **`templates/symptoms.html`**   → Symptoms form + prediction result
- **`templates/suggester.html`**  → Remedies lookup + result
- **`templates/about.html`**      → Static info page

### Static Assets
- **`static/css/`**
  - `project.css`         → Custom styles and responsive rules
  - `bootstrap.min.css`   → Bootstrap framework
- **`static/js/`**
  - `slide.js`            → Home page image slideshow (only runs on Home)
  - `symptoms.js`         → Hides previous prediction when user edits inputs
  - `suggester.js`        → Page-specific JS for suggester (if any hooks)
- **`static/images/`**    → App images and icons

### Data Files
- **`Training.csv`**  → Training data with symptom columns and a prognosis column
- **`Testing.csv`**   → Test set used for accuracy selection
- **`Remedies.csv`**  → Conditions and corresponding remedy text

### Configuration & Database
- **`database.db`**
  - Local SQLite DB created at runtime; stores symptom inputs and predictions
  - Ignored by Git via `.gitignore`

- **`requirements.txt`**
  - Minimal dependencies to run the app (Flask, pandas, scikit-learn)
  - If your platform needs it, install numpy/scipy as well (pip will usually resolve)

- **`start.sh`**
  - Simple one-line runner used for Glitch/hosting scenarios

- **`.gitignore`**
  - Excludes caches, virtualenvs, databases, build outputs and editor files

## 6) UI details / helpful behaviors
   - Symptoms inputs show human-friendly labels (e.g., “Back Pain”) but the backend
     receives machine codes (e.g., back_pain) so the model works consistently.
   - After submitting Symptoms/Suggester, the page anchors to the result box.
   - When the user focuses/types again in Symptoms inputs, the previous result box
     is hidden to avoid confusion.

## 7) Notes on data and accuracy
- Accuracy is selected on a small `Testing.csv` (43 rows), so different models can tie. The project logs per-model accuracies and the selected model each request.
- For stronger validation, consider k-fold cross-validation on `Training.csv` and using `Testing.csv` strictly for a final holdout check.

