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
    layout="wide"
)

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
    st.title("🖼️ Yapay Zeka Destekli Görüntü Sınıflandırıcı")
    st.markdown("---")
    
    # Sidebar - Model ayarları
    st.sidebar.header("⚙️ Model Ayarları")
    
    model_path = st.sidebar.text_input(
        "Model Yolu",
        value="models/best_model.pth",
        help="Eğitilmiş model dosyasının yolu"
    )
    
    num_classes = st.sidebar.number_input(
        "Sınıf Sayısı",
        min_value=1,
        value=10,
        help="Modelin sınıflandıracağı sınıf sayısı"
    )
    
    # Sınıf isimlerini yükle (örnek - kullanıcı kendi sınıflarını girebilir)
    st.sidebar.subheader("📋 Sınıf İsimleri")
    class_names_input = st.sidebar.text_area(
        "Sınıf isimlerini virgülle ayırarak girin",
        value="airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck",
        help="Örnek: cat, dog, bird"
    )
    class_names = [name.strip() for name in class_names_input.split(',')]
    
    if len(class_names) != num_classes:
        st.sidebar.warning(f"⚠️ Sınıf sayısı ({num_classes}) ile sınıf isimleri sayısı ({len(class_names)}) eşleşmiyor!")
    
    # Ana içerik alanı
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Görüntü Yükle")
        
        uploaded_file = st.file_uploader(
            "Bir görüntü seçin",
            type=['png', 'jpg', 'jpeg'],
            help="PNG, JPG veya JPEG formatında görüntü yükleyin"
        )
        
        if uploaded_file is not None:
            # Görüntüyü göster
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Yüklenen Görüntü", use_container_width=True)
            
            # Tahmin butonu
            predict_button = st.button("🔍 Tahmin Et", type="primary", use_container_width=True)
        else:
            predict_button = False
            st.info("👆 Lütfen bir görüntü yükleyin")
    
    with col2:
        st.subheader("📊 Sonuçlar")
        
        if predict_button and uploaded_file is not None:
            try:
                # Model yükleme
                if not os.path.exists(model_path):
                    st.error(f"❌ Model dosyası bulunamadı: {model_path}")
                    st.info("💡 Lütfen önce modeli eğitin veya doğru model yolunu girin.")
                else:
                    with st.spinner("🔄 Model yükleniyor ve tahmin yapılıyor..."):
                        model, device = load_cached_model(model_path, num_classes)
                        preprocessor = load_preprocessor()
                        
                        # Görüntüyü ön işleme tabi tut
                        image_tensor = preprocessor.preprocess_image(image, augment=False)
                        
                        # Tahmin yap
                        results = predict(
                            model=model,
                            image_tensor=image_tensor,
                            class_names=class_names[:num_classes],
                            device=device,
                            top_k=min(5, num_classes)
                        )
                    
                    # Sonuçları göster
                    st.success("✅ Tahmin tamamlandı!")
                    
                    # En yüksek tahmin
                    top_result = results[0]
                    st.metric(
                        label="En Olası Sınıf",
                        value=top_result['class'],
                        delta=f"{top_result['probability']*100:.2f}%"
                    )
                    
                    # Tüm tahminler
                    st.subheader("📈 Tüm Tahminler")
                    for i, result in enumerate(results, 1):
                        st.progress(
                            result['probability'],
                            text=f"{i}. {result['class']}: {result['probability']*100:.2f}%"
                        )
                    
                    # Bar chart için veri hazırla
                    import pandas as pd
                    df_results = pd.DataFrame(results)
                    st.bar_chart(
                        df_results.set_index('class')['probability'],
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"❌ Hata oluştu: {str(e)}")
                st.exception(e)
        else:
            st.info("👈 Sol taraftan bir görüntü yükleyip 'Tahmin Et' butonuna tıklayın")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Yapay Zeka Destekli Görüntü Sınıflandırıcı | PyTorch & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

