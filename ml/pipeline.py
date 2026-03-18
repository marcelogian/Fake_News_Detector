"""
Fake News Detector — Full ML/NLP Pipeline
"""

import re
import string
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


class TextPreprocessor:
    def __init__(self, use_lemmatizer: bool = True, use_stemmer: bool = False):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer() if use_lemmatizer else None
        self.stemmer = PorterStemmer() if use_stemmer else None

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
        text = re.sub(r"\d+", "", text)
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = re.findall(r"\b[a-z]+\b", text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        elif self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    def fit_transform(self, series: pd.Series) -> pd.Series:
        return series.apply(self.clean)


def build_vectorizers():
    return {
        "tfidf_word": TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        "tfidf_char": TfidfVectorizer(max_features=30_000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3),
        "bow": CountVectorizer(max_features=40_000, ngram_range=(1, 2), min_df=2),
    }


def build_models():
    return {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced"),
        "naive_bayes": MultinomialNB(alpha=0.1),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, class_weight="balanced"),
        "linear_svc": LinearSVC(C=0.5, max_iter=2000, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5),
    }


def evaluate(model, X_test, y_test, label: str) -> dict:
    y_pred = model.predict(X_test)
    results = {
        "model": label,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    print(f"\n{'='*55}\n  {label.upper()}\n{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    return results


def generate_synthetic_data(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    fake_templates = [
        "BREAKING: {celebrity} secretly {verb} in {place} – government cover-up exposed!",
        "SHOCKING: Scientists HIDE the truth about {topic} – leaked documents REVEAL all!",
        "URGENT: {politician} ADMITS to {crime} – mainstream media REFUSES to report!",
        "BOMBSHELL: {celebrity} arrested for {crime} – you won't believe what happened next!",
        "WHISTLEBLOWER: {agency} has been LYING about {topic} for decades!",
    ]
    real_templates = [
        "Federal Reserve raises interest rates by {n} basis points amid inflation concerns.",
        "New study published in {journal} shows {topic} linked to improved health outcomes.",
        "{country} signs trade agreement with {country2} following months of negotiations.",
        "Tech company reports {n}% revenue increase in Q{q} earnings call.",
        "Health officials confirm {n} new cases of {disease} in {region}.",
    ]
    fillers = {
        "celebrity": ["a major celebrity", "a famous actor", "a tech billionaire"],
        "verb": ["fled", "conspired", "hid billions"],
        "place": ["Antarctica", "a bunker", "the Vatican"],
        "topic": ["vaccines", "climate change", "5G networks"],
        "politician": ["a senator", "a world leader", "a cabinet member"],
        "crime": ["fraud", "espionage", "money laundering"],
        "country": ["Russia", "China", "the US"],
        "agency": ["the FDA", "the CDC", "NASA"],
        "journal": ["Nature", "The Lancet", "NEJM"],
        "country2": ["Canada", "Germany", "Japan"],
        "disease": ["influenza", "RSV", "norovirus"],
        "region": ["the Northeast", "the Southwest", "the Midwest"],
        "n": [str(x) for x in range(2, 99)],
        "q": ["1", "2", "3", "4"],
    }

    def fill(template):
        for k, vs in fillers.items():
            while "{" + k + "}" in template:
                template = template.replace("{" + k + "}", rng.choice(vs), 1)
        return template

    fake_texts = [fill(rng.choice(fake_templates)) + " " + " ".join(rng.choice(["claim","exposed","hidden","truth","secret","mainstream","elite","cover","lies","shocking"]) for _ in range(rng.integers(8, 20))) for _ in range(n // 2)]
    real_texts = [fill(rng.choice(real_templates)) + " " + " ".join(rng.choice(["according","reported","announced","confirmed","study","officials","percent","analysis","data","research","published"]) for _ in range(rng.integers(8, 20))) for _ in range(n // 2)]

    df = pd.DataFrame({
        "text": fake_texts + real_texts,
        "label": ["FAKE"] * (n // 2) + ["REAL"] * (n // 2),
    }).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_kaggle_data(fake_path: str, real_path: str) -> pd.DataFrame:
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)
    fake_df["label"] = "FAKE"
    real_df["label"] = "REAL"
    df = pd.concat([fake_df, real_df], ignore_index=True)
    df["text"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")).str.strip()
    return df[["text", "label"]]


def train(df: pd.DataFrame | None = None):
    print("\n🔍  FAKE NEWS DETECTOR — Training Pipeline")
    print("=" * 55)

    if df is None:
        print("⚠️  No dataset provided — using synthetic data for demo.")
        df = generate_synthetic_data(n=4000)

    print(f"📦  Dataset: {len(df)} rows | Label distribution:")
    print(df["label"].value_counts().to_string())

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    print("\n🧹  Preprocessing text …")
    preprocessor = TextPreprocessor(use_lemmatizer=True)
    df["clean_text"] = preprocessor.fit_transform(df["text"])

    X, y = df["clean_text"], df["label_enc"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    vectorizers = build_vectorizers()
    models = build_models()
    best_f1, best_pipeline, best_name, all_results = 0.0, None, "", []

    print("\n🏋️  Training models …")
    for vec_name, vec in vectorizers.items():
        for model_name, clf in models.items():
            if model_name == "naive_bayes" and vec_name == "tfidf_char":
                continue
            label = f"{model_name} [{vec_name}]"
            pipe = Pipeline([("vec", vec), ("clf", clf)])
            try:
                pipe.fit(X_train, y_train)
                res = evaluate(pipe, X_test, y_test, label)
                all_results.append(res)
                if res["f1"] > best_f1:
                    best_f1, best_pipeline, best_name = res["f1"], pipe, label
            except Exception as e:
                print(f"  ✗ {label} failed: {e}")

    results_df = pd.DataFrame(all_results).sort_values("f1", ascending=False)
    print("\n\n📊  MODEL COMPARISON\n" + "=" * 75)
    print(results_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\n\n🥇  Best model: {best_name}  (F1={best_f1:.4f})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_pipeline, X, y, cv=cv, scoring="f1_weighted")
    print(f"   5-fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Do NOT pickle TextPreprocessor — instantiate fresh in FakeNewsPredictor
    # to avoid pickle module-path errors when loaded from app.py
    artefacts = {
        "pipeline": best_pipeline,
        "label_encoder": le,
        "results": results_df,
        "best_model_name": best_name,
    }
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(artefacts, f)

    print(f"\n✅  Saved to {MODELS_DIR / 'best_model.pkl'}")
    return artefacts


class FakeNewsPredictor:
    """Load trained pipeline and make predictions."""

    def __init__(self, model_path: str | None = None):
        path = Path(model_path) if model_path else MODELS_DIR / "best_model.pkl"
        with open(path, "rb") as f:
            arts = pickle.load(f)
        self.pipeline = arts["pipeline"]
        self.le = arts["label_encoder"]
        # Fresh instance — never pickled, so no module-path issues
        self.preprocessor = TextPreprocessor(use_lemmatizer=True)

    def predict(self, text: str) -> dict:
        cleaned = self.preprocessor.clean(text)
        pred_enc = self.pipeline.predict([cleaned])[0]
        label = self.le.inverse_transform([pred_enc])[0]

        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            proba = self.pipeline.predict_proba([cleaned])[0]
            confidence = float(max(proba))
        elif hasattr(clf, "decision_function"):
            df_val = self.pipeline.decision_function([cleaned])[0]
            confidence = float(1 / (1 + np.exp(-abs(df_val))))
        else:
            confidence = 1.0

        return {
            "label": label,
            "is_fake": label == "FAKE",
            "confidence": round(confidence * 100, 2),
            "cleaned_preview": cleaned[:200],
        }


if __name__ == "__main__":
    train()