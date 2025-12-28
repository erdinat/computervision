"""
Streamlit Web Arayüzü - Görüntü Sınıflandırıcı
"""

import streamlit as st
import torch
from PIL import Image
import os
import sys

# Proje kök dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import ImagePreprocessor
from src.model_loader import load_model, predict

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Görüntü Sınıflandırıcı",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist design
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    .uploadedImage {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    h1 {
        text-align: center;
        font-weight: 300;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Model ve preprocessor'ı cache'le
@st.cache_resource
def load_cached_model(model_path: str, num_classes: int):
    """Modeli cache'ler"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return load_model(model_path, num_classes, device), device

@st.cache_resource
def load_preprocessor():
    """Preprocessor'ı cache'ler"""
    return ImagePreprocessor()

def main():
    # Minimalist başlık
    st.markdown("<h1>Görüntü Sınıflandırıcı</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Yapay zeka ile görüntülerinizi sınıflandırın</p>', unsafe_allow_html=True)
    
    # Sidebar - Sadece gerekli ayarlar
    with st.sidebar:
        st.header("Ayarlar")
        
        model_path = st.text_input(
            "Model Yolu",
            value="models/best_model.pth",
            help="Eğitilmiş model dosyasının yolu"
        )
        
        num_classes = st.number_input(
            "Sınıf Sayısı",
            min_value=1,
            value=5,
            help="Modelin sınıflandıracağı sınıf sayısı"
        )
        
        class_names_input = st.text_area(
            "Sınıf İsimleri",
            value="fil, kelebek, kedi, koyun, örümcek",
            help="Virgülle ayırarak girin"
        )
        class_names = [name.strip() for name in class_names_input.split(',') if name.strip()]
        
        if len(class_names) != num_classes:
            st.warning(f"Sınıf sayısı ({num_classes}) ile isim sayısı ({len(class_names)}) eşleşmiyor")
    
    # Ana içerik - Merkezi ve odaklı
    max_width = 800
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Görüntü yükleme alanı
        uploaded_file = st.file_uploader(
            "",
            type=['png', 'jpg', 'jpeg'],
            help="PNG, JPG veya JPEG formatında görüntü yükleyin",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Görüntüyü göster
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="")
            
            # Tahmin butonu
            predict_button = st.button("Tahmin Et", type="primary", use_container_width=True)
        else:
            predict_button = False
            st.markdown("---")
            st.markdown("""
                <div style='text-align: center; padding: 3rem 1rem; color: #999;'>
                    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>Görüntü yükleyin</p>
                    <p style='font-size: 0.9rem;'>PNG, JPG veya JPEG formatında</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Sonuçlar
        if predict_button and uploaded_file is not None:
            try:
                if not os.path.exists(model_path):
                    st.error(f"Model dosyası bulunamadı: {model_path}")
                else:
                    with st.spinner("Tahmin yapılıyor..."):
                        model, device = load_cached_model(model_path, num_classes)
                        preprocessor = load_preprocessor()
                        image_tensor = preprocessor.preprocess_image(image, augment=False)
                        results = predict(
                            model=model,
                            image_tensor=image_tensor,
                            class_names=class_names[:num_classes],
                            device=device,
                            top_k=min(5, num_classes)
                        )
                    
                    st.markdown("---")
                    
                    # En yüksek tahmin - vurgulu
                    top_result = results[0]
                    st.markdown(f"""
                        <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 12px; margin: 1rem 0;'>
                            <p style='font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;'>Tahmin</p>
                            <h2 style='margin: 0.5rem 0; font-weight: 500;'>{top_result['class'].title()}</h2>
                            <p style='font-size: 1.2rem; color: #0066cc; font-weight: 600;'>{top_result['probability']*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Diğer tahminler - sade
                    if len(results) > 1:
                        st.markdown("### Diğer Olasılıklar")
                        for i, result in enumerate(results[1:], 2):
                            prob_percent = result['probability'] * 100
                            st.markdown(f"""
                                <div style='display: flex; justify-content: space-between; align-items: center; 
                                            padding: 0.75rem; margin: 0.5rem 0; 
                                            background: white; border-radius: 8px; border-left: 3px solid #0066cc;'>
                                    <span style='font-weight: 500;'>{result['class'].title()}</span>
                                    <span style='color: #666;'>{prob_percent:.1f}%</span>
                                </div>
                            """, unsafe_allow_html=True)
                            
            except Exception as e:
                st.error(f"Hata: {str(e)}")
                with st.expander("Detaylı hata bilgisi"):
                    st.exception(e)
    
    # Minimalist footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #999; font-size: 0.85rem; padding: 1rem;'>
            Görüntü Sınıflandırıcı
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
