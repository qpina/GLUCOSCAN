import streamlit as st
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import pandas as pd
import numpy as np

# -------------------------
# Configuración de la página
# -------------------------
st.set_page_config(page_title="GlucoScan", page_icon="🍏", layout="wide")

# -------------------------
# Cargar modelo y base de datos
# -------------------------
@st.cache_resource
def cargar_modelo():
    model = YOLO("best.pt")
    return model

@st.cache_data
def cargar_base_datos():
    df = pd.read_csv("BaseDeDatos.csv")
    df = df.dropna(how='all')
    df.columns = df.columns.str.strip()  # Quitar espacios
    return df

modelo = cargar_modelo()
base_datos = cargar_base_datos()

# -------------------------
# Tabs principales
# -------------------------
tab1, tab2 = st.tabs(["📸 Contador", "🎓 Sobre Nosotros"])

# -------------------------
# Tab 1: Contador
# -------------------------
with tab1:
    st.header("Calcula las raciones de insulina subiendo una imagen 📷")

    uploaded_file = st.file_uploader("📤 Sube una imagen con un dado visible:", type=["jpg", "jpeg", "png"])

    st.divider()

    ratio = st.number_input("💉 Ratio de insulina (g HC / unidad)", min_value=1.0, max_value=30.0, step=0.5, value=10.0)

    st.divider()

    procesar = st.button("🔎 Procesar imagen")

    if uploaded_file is not None and procesar:
        with st.spinner('🔄 Procesando imagen...'):

            with open("imagen_temporal.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                resultados = procesar_imagen("imagen_temporal.jpg", ratio_usuario=ratio)

                st.success("✅ Imagen procesada correctamente")
                st.divider()

                st.subheader("📊 Resultados por alimento:")
                col1, col2, col3 = st.columns(3)

                for idx, (fruta, datos) in enumerate(resultados.items()):
                    with [col1, col2, col3][idx % 3]:
                        st.metric(
                            label=f"🍏 {fruta}",
                            value=f"{datos['raciones']:.2f} u insulina",
                            delta=f"{datos['hc']:.1f}g HC / {datos['masa']:.1f}g peso",
                            help=f"Volumen: {datos['volumen']:.1f} cm³"
                        )

                st.divider()

                total = sum([d["raciones"] for d in resultados.values()])
                st.success(f"💉 Ración total recomendada: **{total:.2f} unidades de insulina**")

                st.divider()

                st.subheader("📸 Imagen procesada con detecciones:")
                imagen_resultado = Image.open("imagen_procesada.jpg")
                st.image(imagen_resultado, caption="Alimentos y dado detectados", use_container_width=True)

                with open("imagen_procesada.jpg", "rb") as file:
                    st.download_button(
                        label="📥 Descargar imagen procesada",
                        data=file,
                        file_name="resultado.jpg",
                        mime="image/jpeg"
                    )

            except Exception as e:
                st.error(f"❌ Error procesando la imagen: {e}")
                raise

# -------------------------
# Tab 2: Sobre Nosotros
# -------------------------
with tab2:
    st.header("Sobre el proyecto GlucoScan 🎓")

    col1, col2 = st.columns(2)

    with col1:
        st.image("foto_grupo.jpg", width=400)  # Asegúrate de subir esta imagen al repo

    with col2:
        st.markdown("""
        **GlucoScan** es un proyecto desarrollado por estudiantes de la Universidad [Nombre Universidad].

        Nuestro objetivo es facilitar el conteo de hidratos de carbono a personas con diabetes mediante visión artificial.

        - 👩‍🎓 Estudiantes: Quique, María, Juan, Ana
        - 📍 Universidad: [Nombre Universidad]
        - 📅 Año: 2024

        Gracias por confiar en nuestra app. ¡Seguiremos mejorándola cada día! 🚀
        """)
