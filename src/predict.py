
from pathlib import Path
import joblib


# 1. Kaydedilen modelin konumu
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "language_detector_char.joblib"
)

# 2. Dil kodlarını Türkçe isimlere dönüştür
LANGUAGE_NAMES = {
    "tr": "Türkçe",
    "en": "İngilizce",
    "de": "Almanca",
    "fr": "Fransızca"
}

# 3. Model dosyası mevcut mu?
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model bulunamadı: {MODEL_PATH}"
    )

# 4. Eğitilmiş modeli yükle
model = joblib.load(MODEL_PATH)

print("Dil tespit modeli yüklendi.")
print("Desteklenen diller: Türkçe, İngilizce, Almanca, Fransızca")

# 5. Kullanıcıdan metin al
while True:
    text = input("\nMetin giriniz (çıkış için q): ").strip()

    if text.lower() == "q":
        print("Program sonlandırıldı.")
        break

    if not text:
        print("Lütfen boş olmayan bir metin giriniz.")
        continue

    # 6. Dil tahmini yap
    predicted_code = model.predict([text])[0]

    language = LANGUAGE_NAMES.get(
        predicted_code,
        predicted_code
    )

    print(f"Tahmin edilen dil: {language} ({predicted_code})")
