# 🎛️ El Çizimi Mantık Devresi Analizi ve Dijital Çözümleme

Bu proje, kâğıt üzerine el ile çizilmiş analog mantık devrelerini (AND, OR, NOT, NAND, NOR, XOR, XNOR) bilgisayar görmesi (Computer Vision) ve nesne tespiti (Object Detection) yöntemleriyle tanıyarak dijital bir envantere dönüştürür.

## 🚀 Özellikler
- **YOLOv8 Nano Mimarisi:** CPU/GPU optimize, yüksek hızlı nesne tespiti.
- **Görüntü Ön İşleme (OpenCV):** Kareli kâğıt çizgilerini ve gürültüleri temizleyen Adaptive Thresholding algoritması.
- **Dinamik Web Arayüzü (Streamlit):** Kullanıcının resim yükleyebildiği ve anlık dijital envanter dökümü alabildiği dashboard.

## 📦 Kurulum ve Çalıştırma

1. Gerekli kütüphaneleri yükleyin:
pip install -r requirements.txt

2. Projeyi başlatın :
python -m streamlit run app.py
