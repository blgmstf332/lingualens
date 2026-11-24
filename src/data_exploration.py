
from pathlib import Path

import pandas as pd

# Projemizin ana klasörünü buluyoruz.
BASE_DIR = Path(__file__).resolve().parent.parent

# Ham veri setinin yolu
DATA_PATH = BASE_DIR / "data" / "raw" / "mcv_text_lid" / "mcv_text_lid.tsv"

# Projemizde kullanacağımız diller
TARGET_LANGUAGES = ["tr", "en", "de", "fr"]

# Her dil ve veri bölümü için örnek sayıları
language_counts = {}

# Büyük TSV dosyasını parçalar halinde okuyoruz.
for chunk in pd.read_csv(
    DATA_PATH,
    sep="\t",
    usecols=["sentence", "lang", "split"],
    chunksize=50000,
):
    # Yalnızca seçtiğimiz dört dile ait kayıtlar
    filtered = chunk[chunk["lang"].isin(TARGET_LANGUAGES)]

    # Her dilin train/test/dev sayılarını hesaplıyoruz.
    counts = filtered.groupby(["lang", "split"]).size()

    for key, count in counts.items():
        language_counts[key] = language_counts.get(key, 0) + count

# Sonuçları tablo olarak gösteriyoruz.
result = pd.Series(language_counts).unstack(fill_value=0)

print("\nDil ve veri bölümü dağılımı:")
print(result)

#------------------------------
# METİN UZUNLUĞU ANALİZİ

print("\nMetin uzunluğu analizi başlatıldı...")

length_counts = {}

for chunk in pd.read_csv(
    DATA_PATH,
    sep="\t",
    usecols=["sentence", "lang", "split"],
    chunksize=50000,
):

    # Yalnızca projemizin dört dilini seçiyoruz.
    filtered = chunk[
        chunk["lang"].isin(TARGET_LANGUAGES)
    ].copy()

    # Boş metinleri bu analizden çıkarıyoruz.
    filtered = filtered.dropna(subset=["sentence"])

    # Metinlerdeki boşlukla ayrılmış parça sayısını buluyoruz.
    filtered["word_count"] = (
        filtered["sentence"].str.split().str.len()
    )

    # Sıfır kelimelik metinleri çıkarıyoruz.
    filtered = filtered[filtered["word_count"] > 0]

    # Metin uzunluklarını kategorilere ayırıyoruz.
    filtered["length_group"] = pd.cut(
        filtered["word_count"],
        bins=[0, 3, 7, 15, float("inf")],
        labels=["1-3", "4-7", "8-15", "16+"]
    )

    # Dil, veri bölümü ve uzunluk grubuna göre sayıyoruz.
    counts = filtered.groupby(
        ["lang", "split", "length_group"],
        observed=True
    ).size()

    # Her parçanın sonuçlarını topluyoruz.
    for key, count in counts.items():
        length_counts[key] = length_counts.get(key, 0) + count

# Sonuçları tabloya dönüştürüyoruz.
length_result = pd.Series(
    length_counts,
    name="count"
).rename_axis(
    ["lang", "split", "length_group"]
).unstack(fill_value=0)

print("\nDil ve veri bölümlerine göre metin uzunlukları:")
print(length_result)
