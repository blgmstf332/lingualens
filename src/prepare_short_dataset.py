
from pathlib import Path
import random
import unicodedata

import pandas as pd


# Projemizin ana klasörü
BASE_DIR = Path(__file__).resolve().parent.parent

# Orijinal büyük veri setimiz
RAW_DATA_PATH = (
    BASE_DIR / "data" / "raw"
    / "mcv_text_lid" / "mcv_text_lid.tsv"
)

# Önceden hazırladığımız 40.000 kayıtlık veri seti
SAMPLED_DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "sampled_dataset.csv"
)

# Yeni oluşturacağımız veri setinin konumu
OUTPUT_PATH = (
    BASE_DIR / "data" / "processed"
    / "short_focused_dataset.csv"
)

# Kullanacağımız diller
TARGET_LANGUAGES = ("tr", "en", "de", "fr")

# Eğitim için uzunluk grubu başına örnek sayısı
TRAIN_QUOTAS = {
    "1-3": 2000,
    "4-7": 3000,
    "8-15": 3000,
}

CHUNK_SIZE = 50000
SEED = 42

print("Ham veri mevcut mu?", RAW_DATA_PATH.is_file())
print("Örnek veri mevcut mu?", SAMPLED_DATA_PATH.is_file())
print("Hedef dosya:", OUTPUT_PATH)
def normalize_text(text):
    """Tekrar kontrolü için metni standartlaştırır."""

    text = unicodedata.normalize("NFC", str(text))
    text = text.casefold()
    text = " ".join(text.split())

    return text
def get_length_group(sentence):
    """Metnin kelime sayısına göre uzunluk grubunu döndürür."""

    word_count = len(sentence.split())

    if 1 <= word_count <= 3:
        return "1-3"

    elif 4 <= word_count <= 7:
        return "4-7"

    elif 8 <= word_count <= 15:
        return "8-15"

    return None

def reservoir_update(reservoir, record, seen, limit, rng):
    """
    Bir kayıt için reservoir sampling adımını uygular.

    reservoir: Seçilen kayıtların listesi
    record: Yeni gelen kayıt
    seen: Şimdiye kadar görülen uygun kayıt sayısı
    limit: Saklanacak en fazla kayıt sayısı
    rng: Rastgele sayı üreticisi
    """

    if len(reservoir) < limit:
        reservoir.append(record)

    else:
        index = rng.randrange(seen)

        if index < limit:
            reservoir[index] = record

# Önceden oluşturduğumuz veri setini okuyoruz.
sampled_df = pd.read_csv(SAMPLED_DATA_PATH)

# Sadece Dev ve Test kayıtlarını seçiyoruz.
evaluation_df = sampled_df[
    sampled_df["split"].isin(["dev", "test"])
].copy()

# Tekrar kontrolü için normalize edilmiş metinler
protected_texts = set(
    evaluation_df["sentence"].apply(normalize_text)
)

print("\nDeğerlendirme veri boyutu:", evaluation_df.shape)
print("Korunan benzersiz metin:", len(protected_texts))

rng = random.Random(SEED)
CANDIDATE_FACTOR = 1.20

candidate_limits = {
    group: int(quota * CANDIDATE_FACTOR)
    for group, quota in TRAIN_QUOTAS.items()
}

print("Aday havuz kapasiteleri:", candidate_limits)

# Her dil ve uzunluk grubu için ayrı havuz
reservoirs = {}

# Her havuza gelen uygun kayıtların sayacı
seen_counts = {}

for lang in TARGET_LANGUAGES:
    for length_group in TRAIN_QUOTAS:
        key = (lang, length_group)

        reservoirs[key] = []
        seen_counts[key] = 0

print("\nHam veri taraması başladı...")

total_rows = 0

for chunk in pd.read_csv(
    RAW_DATA_PATH,
    sep="\t",
    usecols=["id", "sentence", "lang", "split"],
    chunksize=CHUNK_SIZE
):
    total_rows += len(chunk)

    # Yalnızca eğitim kayıtlarını ve seçtiğimiz dilleri al.
    chunk = chunk[
        (chunk["split"] == "train")
        & (chunk["lang"].isin(TARGET_LANGUAGES))
    ]

    # Eksik veya boş metinleri ele.
    chunk = chunk.dropna(subset=["sentence"])
    chunk = chunk[
        chunk["sentence"].astype(str).str.strip().ne("")
    ]

    for record in chunk.itertuples(index=False):
        sentence = str(record.sentence)
        lang = record.lang

        length_group = get_length_group(sentence)

        # Hedef uzunluk aralıklarının dışında.
        if length_group is None:
            continue

        # Dev/Test ile çakışan metinleri dışla.
        if normalize_text(sentence) in protected_texts:
            continue

        key = (lang, length_group)

        # Bu havuza gelen uygun kayıt sayısı.
        seen_counts[key] += 1

        reservoir_update(
            reservoir=reservoirs[key],
            record={
                "id": record.id,
                "sentence": sentence,
                "lang": lang,
                "split": "train"
            },
            seen=seen_counts[key],
            limit=candidate_limits[length_group],
            rng=rng
        )

    if total_rows % 1_000_000 < CHUNK_SIZE:
        print(f"Taranan satır: {total_rows:,}")

print("\nTarama tamamlandı.")
print("\nAday havuz sonuçları:")

for key, reservoir in reservoirs.items():
    print(
        f"{key}: "
        f"Uygun kayıt={seen_counts[key]:,}, "
        f"Seçilen aday={len(reservoir):,}"
    )
selected_records = []
selected_texts = set(protected_texts)

for lang in TARGET_LANGUAGES:
    for length_group, quota in TRAIN_QUOTAS.items():

        key = (lang, length_group)
        candidates = reservoirs[key].copy()

        # Rastgele seçim sırası
        rng.shuffle(candidates)

        selected_count = 0

        for record in candidates:
            normalized = normalize_text(record["sentence"])

            # Dev/Test veya daha önce seçilen Train ile aynıysa atla.
            if normalized in selected_texts:
                continue

            selected_records.append(record)
            selected_texts.add(normalized)

            selected_count += 1

            if selected_count == quota:
                break

        print(
            f"{lang} | {length_group} | "
            f"Hedef: {quota} | Seçilen: {selected_count}"
        )

        if selected_count < quota:
            raise ValueError(
                f"{key} grubunda yeterli benzersiz aday yok! "
                f"Hedef={quota}, Seçilen={selected_count}"
            )

# Seçilen eğitim kayıtlarını DataFrame'e dönüştür.
train_df = pd.DataFrame(selected_records)

# Eğitim ve değerlendirme verilerini birleştir.
final_df = pd.concat(
    [train_df, evaluation_df],
    ignore_index=True
)

# Son kontroller
assert len(train_df) == 32000
assert len(final_df) == 40000

assert final_df["id"].is_unique

assert (
    final_df["sentence"]
    .apply(normalize_text)
    .is_unique
)

# Hedef klasörü oluştur ve CSV'yi kaydet.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

final_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)

print("\nYeni veri seti başarıyla kaydedildi!")
print("Toplam kayıt:", len(final_df))

print("\nSplit dağılımı:")
print(pd.crosstab(final_df["lang"], final_df["split"]))
