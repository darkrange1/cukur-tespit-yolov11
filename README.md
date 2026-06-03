# 🚧 Yol Hasarı ve Çukur Tespit Sistemi

Bu proje, yol yüzeyindeki çukur (pothole), çatlak (crack) ve rögar kapağı (manhole) konumlarını derin öğrenme yöntemleriyle otomatik olarak tespit eden bir web uygulamasıdır. Projede **YOLO11s** mimarisi kullanılmıştır.

## 🚀 Özellikler
- **Gerçek Zamanlı Nesne Tespiti**: Yüklenen yol fotoğraflarındaki hasarları anında algılar.
- **Etkileşimli Arayüz**: Streamlit tabanlı kolay ve kullanıcı dostu arayüz.
- **Yüksek Performans**: T4 GPU ortamında optimize edilmiş YOLO11s model ağırlıkları (`best.pt`).

## 🛠️ Kurulum

Uygulamayı yerel makinenizde çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

1. **Depoyu Klonlayın**:
   ```bash
   git clone <github-depo-adresi>
   cd Cukur_Tespit_Final
   ```

2. **Gerekli Kütüphaneleri Yükleyin**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Uygulamayı Çalıştırın**:
   ```bash
   streamlit run app.py
   ```

## 📂 Dosya Yapısı
- `app.py`: Streamlit web arayüzü kaynak kodu.
- `best.pt`: Eğitilmiş YOLO11s model ağırlık dosyası.
- `requirements.txt`: Gerekli Python kütüphaneleri listesi.
- `proje_raporu.pdf`: Detaylı akademik proje raporu (8 sayfa).
- `generate_report.py`: Raporu derleyen ve PDF çıktısı üreten Python betiği.
