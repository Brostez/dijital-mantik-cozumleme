import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import glob

# Sayfa ayarları
st.set_page_config(page_title="Dijital Çözümleme Final", layout="wide")
st.title("🎛️ Dijital Çözümleme: El Çizimi Mantık Devresi Analizi")
st.write("Yüklenen el çizimi devre şemalarındaki mantık kapılarını (AND, OR, NOT vb.) tespit eden uçtan uca sistem.")

@st.cache_resource
def load_model():
    # En son eğitilen model klasörünü otomatik bulma algoritması
    search_path = os.path.join("runs", "detect", "train*")
    folders = glob.glob(search_path)
    if not folders:
        raise FileNotFoundError("Eğitilmiş model klasörü bulunamadı.")
    
    # Klasörleri oluşturulma zamanına göre sıralayıp en güncel olanını (en son eğitimi) seçiyoruz
    latest_folder = max(folders, key=os.path.getmtime)
    model_path = os.path.join(latest_folder, "weights", "best.pt")
    
    # Eğer güncel klasörde weights yoksa varsayılan ilk klasöre bakıyoruz
    if not os.path.exists(model_path):
        model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
        
    return YOLO(model_path), model_path

# Modeli Yükleme Kontrolü
try:
    model, used_path = load_model()
    st.sidebar.success("Yapay zeka modeli başarıyla yüklendi!")
    st.sidebar.info(f"Aktif Klasör: {os.path.basename(os.path.dirname(os.path.dirname(used_path)))}")
except Exception as e:
    st.sidebar.error(f"Model yüklenirken hata oluştu. Eğitim devam ediyor olabilir. Detay: {e}")
    st.stop()

# Sol menü ayarları
st.sidebar.header("Model ve Görüntü Ayarları")
conf_threshold = st.sidebar.slider("Güven Skoru (Confidence)", min_value=0.01, max_value=1.0, value=0.40, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Görüntü Ön İşleme (Hocanın İsteği)")
# Arka plan temizleme filtresi için toggle
remove_background = st.sidebar.checkbox("Arka Planı Temizle (Kareli Çizgileri Sil)", value=True, help="Bu ayar kareli kağıttaki ızgara çizgilerini yok ederek arka planı dijital beyaza çevirir.")

# Kullanıcıdan dosya alma
uploaded_file = st.file_uploader("Lütfen test etmek için bir el çizimi devre fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Görüntü işleme aşaması
    if remove_background:
        # 1. Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 2. Gaussian Blur ile kağıt dokusunu ve gürültüyü yumuşat (YENİ EKLENDİ)
        # (5, 5) çekirdek boyutu, dokuyu pürüzsüzleştirir ama çizgileri korur.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Uyarlanabilir eşikleme (Adaptive Threshold) - PARAMETRELER GÜNCELLENDİ
        # blockSize'ı 21'den 31'e çıkardık, bu çizgilerin daha dolgun kalmasını sağlar.
        # C değerini 10'dan 15'e çıkardık, bu arka planı daha agresif beyaz yapar.
        processed_gray = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        
        # 4. Median Blur ile kalan ufak tefek noktaları temizleme
        # Değeri 3'ten 5'e çıkardık, pürüzsüzleştirmeyi artırdık.
        processed_gray = cv2.medianBlur(processed_gray, 5)
        
        # 5. YOLO'nun beklediği 3 kanallı (RGB) formata geri çevirme
        final_img_array = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
    else:
        # Hiçbir filtre uygulanmamış orijinal hal
        final_img_array = img_array

    # Ekranda yan yana iki sütun
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 İşlenecek Devre")
        st.image(final_img_array, use_container_width=True)

    if st.button("Devreyi Analiz Et ve Kapıları Çözümle"):
        with st.spinner("Yapay zeka mantık kapılarını tespit ediyor..."):
            
            # Görüntüyü OpenCV (BGR) formatına çevirme
            img_cv = cv2.cvtColor(final_img_array, cv2.COLOR_RGB2BGR)

            # Yapay zekanın tahmini yapması
            results = model(img_cv, conf=conf_threshold)
            
            # Bulunan sonuçların resim üzerine çizilmesi
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("🔍 Yapay Zeka Tespit Sonucu")
                st.image(res_rgb, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📊 Dijital Envanter Çıktısı")
            
            # Algılanan nesneleri çekme
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                detected_classes = []
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    detected_classes.append(class_name)
                
                # Benzersiz nesneleri ve adetlerini hesaplama
                unique_elements, counts = np.unique(detected_classes, return_counts=True)
                
                # Metinsel çıktı üretimi
                result_string = ", ".join([f"{count}x {element.upper()}" for element, count in zip(unique_elements, counts)])
                
                # Dinamik textbox
                st.text_input(label="Sistem Tarafından Algılanan Devre Elemanları (Dinamik Çıktı)", value=result_string, disabled=True)
                
            else:
                st.warning("Belirtilen güven skoru değerinde hiçbir mantık kapısı algılanamadı.")