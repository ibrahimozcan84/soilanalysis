import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier
import plotly.express as px

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Toprak Sınıflandırma Sistemi", layout="wide")
st.title("🌱 Toprak Analizi ve Sınıflandırma Sistemi (TabNet)")
st.write("Veri dosyanızı yükleyin, TabNet modeli ile otomatik sınıflandırma yapın.")


# --- 1. Model ve Yardımcı Araçları Yükle ---
@st.cache_resource
def load_artifacts():
    # Eğittiğiniz dosyaların isimlerini buraya göre güncelleyin
    scaler = joblib.load("toprak_scaler.pkl")
    le = joblib.load("toprak_le.pkl")

    # TabNet modelini yükle (Klasör veya .zip adını yazın)
    model = TabNetClassifier()
    model.load_model("toprak_tabnet_modeli.zip")
    return scaler, le, model


try:
    scaler, le, model = load_artifacts()
    st.sidebar.success("Model ve Scaler başarıyla yüklendi.")
except Exception as e:
    st.sidebar.error(f"Dosyalar yüklenirken hata oluştu: {e}")
    st.stop()

# --- 2. Dosya Yükleme Paneli ---
st.sidebar.header("Veri Girişi")
uploaded_file = st.sidebar.file_uploader("CSV veya Excel dosyası seçin", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Dosya okuma
    if uploaded_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)

    st.subheader("📋 Yüklenen Veri Önizlemesi")
    st.dataframe(df_input.head(10))

    # Gerekli sütunların kontrolü
    feature_cols = ["pH", "EC", "CEC", "ESP", "SAR", "TDS", "PS"]
    missing_cols = [col for col in feature_cols if col not in df_input.columns]

    if missing_cols:
        st.error(f"Dosyada şu sütunlar eksik: {missing_cols}")
    else:
        if st.sidebar.button("Analizi Başlat", type="primary"):
            with st.spinner("TabNet tahmin yapıyor..."):
                # Veriyi hazırla
                X_raw = df_input[feature_cols].values.astype(np.float32)

                # Ölçeklendirme (Eğitimdeki scaler kullanılmalı)
                X_scaled = scaler.transform(X_raw)

                # Tahmin
                preds = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)  # Olasılıklar

                # Sayısal etiketleri orijinal isimlere çevir (I, D, ND vb.)
                decoded_preds = le.inverse_transform(preds)

                # Sonuçları ana tabloya ekle
                df_input['Tahmin_Sinif'] = decoded_preds
                df_input['Güven_Skoru'] = np.max(probs, axis=1)  # En yüksek olasılık değeri

                # --- 3. Sonuçların Gösterimi ---
                st.divider()
                st.subheader("🚀 Sınıflandırma Sonuçları")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.dataframe(df_input)
                    # CSV İndirme Butonu
                    csv = df_input.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("Sonuçları İndir (CSV)", csv, "Tahmin_Sonuclari.csv", "text/csv")

                with col2:
                    # Sınıf Dağılım Grafiği
                    fig = px.pie(df_input, names='Tahmin_Sinif', title="Sınıf Dağılımı", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)

                # --- 4. Özellik Önem Düzeyleri (Global) ---
                st.divider()
                st.subheader("🔍 TabNet Özellik Önem Düzeyleri")
                importances = model.feature_importances_
                imp_df = pd.DataFrame({'Özellik': feature_cols, 'Önem': importances}).sort_values(by='Önem',
                                                                                                  ascending=True)

                fig_imp = px.bar(imp_df, x='Önem', y='Özellik', orientation='h',
                                 title="Model Tahmin Yaparken Hangi Parametrelere Baktı?",
                                 color='Önem', color_continuous_scale='Viridis')
                st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.info("Lütfen sol menüden analiz edilecek toprak verilerini içeren bir dosya yükleyin.")
    st.warning("Not: Dosyanız şu sütunları içermelidir: pH, EC, CEC, ESP, SAR, TDS, PS")

# --- Alt Bilgi ---
st.sidebar.divider()
st.sidebar.caption("İbrahim Özcan - Akademik Çalışma Aracı")