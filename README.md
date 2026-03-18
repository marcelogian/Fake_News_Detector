# Fake_News_Detector
An NLP + ML pipeline that classifies news articles as **FAKE** or **REAL**, deployed as a Flask REST API with an interactable GUI that shows the percentage of prediction, suggested content to be predicted and keywords that make the content either real or fake.

## ML Pipeline

### Step 1 — Text Preprocessing (NLTK)
- Lowercase normalisation
- URL & HTML tag removal
- Punctuation & digit stripping
- Tokenisation (`nltk.word_tokenize`)
- Stopword removal (`nltk.corpus.stopwords`)
- Lemmatisation (`WordNetLemmatizer`)

### Step 2 — Feature Extraction
| Method | Details |
|---|---|
| **TF-IDF (word)** | 50k features, unigrams + bigrams, sublinear TF |
| **TF-IDF (char)** | 30k features, char n-grams (3–5), sublinear TF |
| **Bag of Words** | 40k features, unigrams + bigrams |

### Step 3 — Models Trained
| Model | Notes |
|---|---|
| Logistic Regression | L2, balanced class weights |
| Naive Bayes | Multinomial, α=0.1 |
| Random Forest | 200 trees, balanced weights |
| Linear SVC | Fast, often best for text |
| Gradient Boosting | 150 estimators |

All models are evaluated on **accuracy, precision, recall, F1** and the best F1 pipeline is saved.

### Step 4 — Best Model Selection
5-fold stratified cross-validation confirms the winner. The winning `sklearn.Pipeline` (vectoriser + classifier) is serialised with `pickle`.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Drop your Kaggle CSVs in data/
#    Then update pipeline.py → load_kaggle_data() call in train()

# 3. Train (runs automatically on first API call too)
python -m ml.pipeline

# 4. Start the API
python app.py
# or with gunicorn:
gunicorn -w 4 -b 0.0.0.0:5000 app:app