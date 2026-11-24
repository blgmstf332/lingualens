
from pathlib import Path

import joblib
import streamlit as st


# Proje ve model yolları
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "language_detector_char.joblib"
)

LANGUAGE_NAMES = {
    "tr": "Türkçe",
    "en": "İngilizce",
    "de": "Almanca",
    "fr": "Fransızca"
}


# Kaydedilmiş modeli yükle
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Sayfa ayarları
st.set_page_config(
    page_title="Kısa Metinlerde Dil Tespiti",
    page_icon="💬"
)

st.title("💬 Kısa Metinlerde Dil Tespiti")

st.write(
    "Türkçe, İngilizce, Almanca ve Fransızca "
    "metinlerin dilini makine öğrenmesiyle tespit edin."
)

# Model dosyasının varlığını kontrol et
if not MODEL_PATH.is_file():
    st.error("Model dosyası bulunamadı.")
    st.stop()

model = load_model()

# Kullanıcıdan metin al
text = st.text_area(
    "Analiz edilecek metni giriniz:",
    placeholder="Örnek: Bonjour, comment allez-vous?",
    height=120
)

# Tahmin butonu
if st.button("Dili Tespit Et", type="primary"):

    if not text.strip():
        st.warning("Lütfen bir metin giriniz.")

    else:
        predicted_code = model.predict([text.strip()])[0]

        language = LANGUAGE_NAMES.get(
            predicted_code,
            predicted_code
        )

        st.success(
            f"Tahmin edilen dil: {language} ({predicted_code})"
        )

st.info(
    "Bu sistem yalnızca Türkçe, İngilizce, "
    "Almanca ve Fransızca dillerini destekler."
)
