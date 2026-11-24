
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


# ============================================================
# 1. VERİ SETİNİ YÜKLEME
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "short_focused_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

# Eğitim ve doğrulama verilerini ayır.
train_df = df[df["split"] == "train"]
dev_df = df[df["split"] == "dev"]

X_train = train_df["sentence"]
y_train = train_df["lang"]

X_dev = dev_df["sentence"]
y_dev = dev_df["lang"]

print("Train kayıt sayısı:", len(X_train))
print("Dev kayıt sayısı:", len(X_dev))


# ============================================================
# 2. WORD N-GRAM MODELİ
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            min_df=2,
            max_features=30000
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=300)
    )
])

print("\nWord modeli eğitiliyor...")

model.fit(X_train, y_train)

y_pred = model.predict(X_dev)

print("\nWORD MODEL - DEV SONUÇLARI")

print(
    classification_report(
        y_dev,
        y_pred,
        digits=4
    )
)


# ============================================================
# 3. CHARACTER N-GRAM MODELİ
# ============================================================

char_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            max_features=50000
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=300)
    )
])

print("\nCharacter modeli eğitiliyor...")

char_model.fit(X_train, y_train)

char_pred = char_model.predict(X_dev)

print("\nCHARACTER MODEL - DEV SONUÇLARI")

print(
    classification_report(
        y_dev,
        char_pred,
        digits=4
    )
)


# ============================================================
# 4. DEV MODEL KARŞILAŞTIRMASI
# ============================================================

word_macro_f1 = f1_score(
    y_dev,
    y_pred,
    average="macro"
)

char_macro_f1 = f1_score(
    y_dev,
    char_pred,
    average="macro"
)

print("\nDEV MODEL KARŞILAŞTIRMASI")

print("Word Macro F1:", round(word_macro_f1, 4))
print("Char Macro F1:", round(char_macro_f1, 4))


# ============================================================
# 5. METİN UZUNLUĞUNA GÖRE DEV KARŞILAŞTIRMASI
# ============================================================

results_df = dev_df[["sentence", "lang"]].copy()

# Her metnin kelime sayısını hesapla.
results_df["word_count"] = (
    results_df["sentence"].str.split().str.len()
)

# Metinleri uzunluk gruplarına ayır.
results_df["length_group"] = pd.cut(
    results_df["word_count"],
    bins=[0, 3, 7, 15, float("inf")],
    labels=["1-3", "4-7", "8-15", "16+"]
)

comparison_df = results_df.copy()

# Modellerin doğru tahmin durumları
comparison_df["word_correct"] = (
    y_pred == y_dev.to_numpy()
)

comparison_df["char_correct"] = (
    char_pred == y_dev.to_numpy()
)

grouped = comparison_df.groupby(
    "length_group",
    observed=True
)

comparison_table = grouped[
    ["word_correct", "char_correct"]
].mean() * 100

comparison_table["count"] = grouped.size()

comparison_table = comparison_table.rename(
    columns={
        "word_correct": "Word Accuracy (%)",
        "char_correct": "Char Accuracy (%)"
    }
)

print("\nMETİN UZUNLUĞUNA GÖRE DEV KARŞILAŞTIRMASI")
print(comparison_table.round(2))


# ============================================================
# 6. HATA ANALİZİ
# ============================================================

error_comparison = dev_df[
    ["sentence", "lang"]
].copy()

error_comparison["word_pred"] = y_pred
error_comparison["char_pred"] = char_pred

# Her modelin yanlış tahmin ettiği kayıtlar
word_wrong = (
    error_comparison["word_pred"] != error_comparison["lang"]
)

char_wrong = (
    error_comparison["char_pred"] != error_comparison["lang"]
)

# Word yanlış, Char doğru
fixed_errors = error_comparison[
    word_wrong & ~char_wrong
]

# Word doğru, Char yanlış
new_errors = error_comparison[
    ~word_wrong & char_wrong
]

# İki model de yanlış
common_errors = error_comparison[
    word_wrong & char_wrong
]

print("\nKARAKTER MODELİNİN DÜZELTTİĞİ HATALAR")
print(fixed_errors.to_string(index=False))

print("\nKARAKTER MODELİNİN YAPTIĞI YENİ HATALAR")
print(new_errors.to_string(index=False))

print("\nİKİ MODELİN ORTAK HATALARI")
print(common_errors.to_string(index=False))

print("\nHATA ÖZETİ")
print("Düzeltilen:", len(fixed_errors))
print("Yeni:", len(new_errors))
print("Ortak:", len(common_errors))


# ============================================================
# 7. KISA METİNLERDE DEV DEĞERLENDİRMESİ
# ============================================================

short_df = error_comparison[
    error_comparison["sentence"]
    .str.split()
    .str.len()
    .between(1, 3)
].copy()

print("\nKISA DEV METİNLERİNİN DİL DAĞILIMI")
print(short_df["lang"].value_counts().sort_index())

print("\nWORD MODEL - KISA DEV SONUÇLARI")

print(
    classification_report(
        short_df["lang"],
        short_df["word_pred"],
        digits=4,
        zero_division=0
    )
)

print("\nCHAR MODEL - KISA DEV SONUÇLARI")

print(
    classification_report(
        short_df["lang"],
        short_df["char_pred"],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 8. NİHAİ TEST DEĞERLENDİRMESİ
# ============================================================

test_df = df[df["split"] == "test"].copy()

X_test = test_df["sentence"]
y_test = test_df["lang"]

assert len(test_df) == 4000

# Eğitilmiş modellerle Test tahminleri
word_test_pred = model.predict(X_test)
char_test_pred = char_model.predict(X_test)

print("\nWORD MODEL - TEST SONUÇLARI")

print(
    classification_report(
        y_test,
        word_test_pred,
        digits=4
    )
)

print("\nCHAR MODEL - TEST SONUÇLARI")

print(
    classification_report(
        y_test,
        char_test_pred,
        digits=4
    )
)

print("\nNİHAİ TEST KARŞILAŞTIRMASI")

print(
    "Word Macro F1:",
    round(f1_score(y_test, word_test_pred, average="macro"), 4)
)

print(
    "Char Macro F1:",
    round(f1_score(y_test, char_test_pred, average="macro"), 4)
)


# ============================================================
# 9. KISA METİNLERDE TEST DEĞERLENDİRMESİ
# ============================================================

# Test kümesindeki 1-3 kelimelik metinleri seç.
short_mask = (
    X_test.str.split().str.len().between(1, 3)
)

y_short = y_test[short_mask]

word_short_pred = word_test_pred[short_mask.to_numpy()]
char_short_pred = char_test_pred[short_mask.to_numpy()]

print("\nKISA TEST METNİ SAYISI:", len(y_short))

print("\nKISA TEST METİNLERİNİN DİL DAĞILIMI")
print(y_short.value_counts().sort_index())

print("\nWORD MODEL - KISA TEST SONUÇLARI")

print(
    classification_report(
        y_short,
        word_short_pred,
        digits=4,
        zero_division=0
    )
)

print("\nCHAR MODEL - KISA TEST SONUÇLARI")

print(
    classification_report(
        y_short,
        char_short_pred,
        digits=4,
        zero_division=0
    )
)
import joblib

# Modellerin saklanacağı klasör
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Kaydedilecek modelin dosya yolu
MODEL_PATH = MODEL_DIR / "language_detector_char.joblib"

# Seçtiğimiz karakter modelini kaydet
joblib.dump(char_model, MODEL_PATH)

print("\nModel başarıyla kaydedildi!")
print("Model yolu:", MODEL_PATH)