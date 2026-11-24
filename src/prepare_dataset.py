
from pathlib import Path
from collections import defaultdict
import random

import pandas as pd


# Proje yolları
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "raw"
    / "mcv_text_lid" / "mcv_text_lid.tsv"
)

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed"
    / "sampled_dataset.csv"
)

# Örnekleme ayarları
TARGET_LANGUAGES = ("tr", "en", "de", "fr")

SAMPLE_SIZES = {
    "train": 8000,
    "dev": 1000,
    "test": 1000,
}

SEED = 42
CHUNK_SIZE = 50000


def sample_dataset():
    """TSV dosyasından dengeli bir altküme oluşturur."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Veri seti bulunamadı: {DATA_PATH}"
        )

    rng = random.Random(SEED)

    # Her (dil, split) grubu için seçilen kayıtlar
    reservoirs = defaultdict(list)

    # Her grupta karşılaşılan kayıt sayısı
    seen_counts = defaultdict(int)

    print("Veri örnekleme başlatıldı...", flush=True)

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            sep="\t",
            usecols=["id", "sentence", "lang", "split"],
            chunksize=CHUNK_SIZE,
            dtype={
                "id": "string",
                "sentence": "string",
                "lang": "string",
                "split": "string",
            },
        ),
        start=1,
    ):

        # Yalnızca seçilen dört dili alıyoruz.
        chunk = chunk[
            chunk["lang"].isin(TARGET_LANGUAGES)
            & chunk["split"].isin(SAMPLE_SIZES)
        ].copy()

        # Eksik ve boş metinleri çıkarıyoruz.
        chunk = chunk.dropna(
            subset=["sentence", "lang", "split"]
        )

        chunk["sentence"] = chunk["sentence"].str.strip()

        chunk = chunk[chunk["sentence"] != ""]

        # Kayıtları tek tek reservoir sampling'e gönderiyoruz.
        for record in chunk.itertuples(index=False):

            record_id, sentence, lang, split = record

            key = (lang, split)
            limit = SAMPLE_SIZES[split]

            seen_counts[key] += 1
            seen = seen_counts[key]

            row = {
                "id": record_id,
                "sentence": sentence,
                "lang": lang,
                "split": split,
            }

            if len(reservoirs[key]) < limit:
                reservoirs[key].append(row)
            else:
                # Daha önce görülen kayıtlar arasında
                # eşit seçilme şansı sağlıyoruz.
                index = rng.randrange(seen)

                if index < limit:
                    reservoirs[key][index] = row

        if chunk_number % 20 == 0:
            print(
                f"İşlenen TSV parçası: {chunk_number}",
                flush=True,
            )

    # Yeterli veri olup olmadığını kontrol ediyoruz.
    for lang in TARGET_LANGUAGES:
        for split, required in SAMPLE_SIZES.items():
            actual = len(reservoirs[(lang, split)])

            if actual < required:
                raise ValueError(
                    f"{lang}/{split}: "
                    f"{required} kayıt gerekli, {actual} bulundu."
                )

    # Seçilen kayıtları bir DataFrame'e dönüştürüyoruz.
    rows = []

    for lang in TARGET_LANGUAGES:
        for split in SAMPLE_SIZES:
            rows.extend(reservoirs[(lang, split)])

    sampled_df = pd.DataFrame(rows)

    # Tekrarlanabilir biçimde karıştırıyoruz.
    sampled_df = sampled_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sampled_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nÖrnekleme tamamlandı.")
    print("Dosya:", OUTPUT_PATH)

    print("\nDil ve veri bölümü dağılımı:")
    print(
        pd.crosstab(
            sampled_df["lang"],
            sampled_df["split"],
        )
    )

    return sampled_df


if __name__ == "__main__":
    sample_dataset()
