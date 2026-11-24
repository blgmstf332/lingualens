# 💬 LinguaLens - Kısa Metinlerde Dil Tespiti (Language Detection)

## Proje Hakkında

Bu proje, Doğal Dil İşleme dersi kapsamında geliştirilmiştir.

Amaç, birkaç kelimeden oluşan kısa metinlerin dilini makine öğrenmesi yöntemleriyle tahmin etmektir. Proje kapsamında Türkçe, İngilizce, Almanca ve Fransızca dilleri desteklenmektedir.

## Proje Kapsamı

- Türkçe, İngilizce, Almanca ve Fransızca için dil tespiti
- Karakter ve sözcük n-gramlarının karşılaştırılması
- TF-IDF ve Logistic Regression ile metin sınıflandırma
- Dil başına Precision, Recall ve F1-score değerlendirmesi
- Kısa metinlerde performans ve hata analizi
- Streamlit tabanlı web arayüzü

## Kullanılan Teknolojiler

- Python 3.12
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Joblib
- Matplotlib
- Git ve GitHub

## Veri Seti

Mozilla Common Voice Text Language Identification veri seti kullanılmıştır.

Reservoir Sampling yöntemiyle dört dil için toplam **40.000 kayıtlık** bir veri seti hazırlanmıştır.

- Eğitim (Train): 32.000 kayıt
- Doğrulama (Dev): 4.000 kayıt
- Test: 4.000 kayıt

**Veri kaynağı:** https://mozilladatacollective.com/datasets/cmj8ddapc02c8mb07l6wyr882

## Model Sonuçları

| Model | Test Macro F1 |
|---|---:|
| Word N-gram + Logistic Regression | 0.9965 |
| Character N-gram + Logistic Regression | **0.9983** |

Karakter tabanlı model, 1–3 kelimelik test metinlerinde **0.8793 Macro F1** skoruna ulaşmıştır.

## Kurulum ve Çalıştırma

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

Veri setini indirdikten sonra hazırlamak ve modeli eğitmek için:

```bash
python src/prepare_dataset.py
python src/prepare_short_dataset.py
python src/train_baseline.py
```

Web uygulamasını başlatmak için:

```bash
streamlit run src/web_app.py
```

Uygulama yerel olarak `http://localhost:8501` adresinde çalışır.

**Not:** Veri seti ve eğitilmiş model GitHub'a yüklenmemektedir. Veri seti `data/raw/mcv_text_lid/mcv_text_lid.tsv` konumuna çıkarılarak model yeniden oluşturulabilir.

## Proje Durumu

Veri hazırlama, model eğitimi, karşılaştırmalı değerlendirme, hata analizi ve Streamlit web arayüzü tamamlanmıştır.