import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Akıllı Yol Hasarı Analizi",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Dijital Çözümleme Dönem Projesi: Akıllı Yol Analizi")
st.markdown("Bu web uygulaması, yollardaki çukur (pothole), çatlak (crack) ve rögar kapağı (manhole) hasarlarını derin öğrenme (YOLO11s) kullanarak tespit eder.")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        return model, True
    except Exception as e:
        return None, False

model, model_loaded = load_model()

# Sidebar configuration
st.sidebar.header("🛠️ Ayarlar & Bilgiler")

if model_loaded:
    st.sidebar.success("YOLO11s Modeli Başarıyla Yüklendi.")
    # Show class mapping
    st.sidebar.markdown("### 🏷️ Algılanabilir Sınıflar")
    for cid, cname in model.names.items():
        st.sidebar.markdown(f"- **Sınıf {cid}:** {cname}")
else:
    st.sidebar.error("Model dosyası (`best.pt`) bulunamadı! Lütfen proje ana dizininde olduğundan emin olun.")

# Confidence slider to adjust sensitivity
conf_threshold = st.sidebar.slider(
    "🔍 Güven Eşiği (Confidence)", 
    min_value=0.05, 
    max_value=0.90, 
    value=0.15, 
    step=0.05,
    help="Düşük değerler daha fazla hasar bulur ama hata olasılığı artar. Yüksek değerler sadece kesin tespitleri gösterir."
)

# File uploader
uploaded_file = st.file_uploader("Analiz için bir yol fotoğrafı yükleyin...", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)
    
    # Create columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Yüklenen Fotoğraf")
        st.image(image, width="stretch")
        
    with col2:
        st.subheader("🔍 Analiz Sonucu")
        with st.spinner("Yapay zeka görseli analiz ediyor..."):
            # Inference with user selected confidence
            results = model(image, conf=conf_threshold)
            res_plotted = results[0].plot()
            st.image(res_plotted, width="stretch")
            
            # Extract detection details
            boxes = results[0].boxes
            num_detections = len(boxes)
            
            if num_detections > 0:
                st.success(f"Analiz Tamamlandı! Toplam {num_detections} adet nesne tespit edildi.")
                
                # Class counts breakdown
                classes = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                
                counts = {name: 0 for name in model.names.values()}
                for c in classes:
                    cname = model.names[c]
                    counts[cname] += 1
                
                st.markdown("### 📊 Tespit Detayları")
                for cname, count in counts.items():
                    if count > 0:
                        st.markdown(f"- **{cname}:** {count} adet")
            else:
                st.warning("Bu güven eşiği seviyesinde hiçbir hasar veya nesne tespit edilemedi. Lütfen sol menüdeki güven eşiğini (Confidence) düşürerek tekrar deneyin.")