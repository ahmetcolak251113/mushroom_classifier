# Mushroom Classifier

Bu repo, mantarların özelliklerine bakarak **farklı sınıflar** şeklinde sınıflandırılmasını amaçlayan bir makine öğrenmesi projesidir. Tüm kodlar `src/` klasöründe, ham / işlenmiş veriler `data/` klasöründe, deney sonuçları ve çıktı dosyaları ise `results/` klasöründe tutulur.

---

##  Proje Amacı

- Mantar özelliklerinden (şapka rengi, şekli, lamel özellikleri vb.) yola çıkarak **otomatik sınıflandırma** yapmak  
- Farklı model ve hiperparametre kombinasyonlarını deneyerek **en iyi performansı veren modeli** bulmak  
- Veri bilimi / makine öğrenmesi sürecini uçtan uca (veri hazırlama → modelleme → değerlendirme → sonuç analizi) göstermeyi hedeflemek

---

## Proje Yapısı

Depoda şu anda üst düzeyde aşağıdaki klasör ve dosyalar bulunuyor:

```text
mushroom_classifier/
├── src/            # Asıl Python kodları (veri yükleme, model eğitimi, değerlendirme vb.)
├── data/           # Veri setleri (ham / işlenmiş)
├── results/        # Eğitim çıktılarını, metrikleri, grafik vb. sonuçları saklamak için
├── .idea/          # IDE (PyCharm vb.) proje ayarları
├── .gitignore      # Git tarafından izlenmeyen dosya/klasörler
└── requirements.txt# Python bağımlılıkları
