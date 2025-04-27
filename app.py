import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------
# Configuración de la página
# -------------------------
st.set_page_config(page_title="GlucoScan", page_icon="🍏", layout="wide")

# -------------------------
# Cargar base de datos
# -------------------------
df = pd.read_csv("BaseDeDatos.csv")
df = df.dropna(how='all').set_index("Alimento")

# -------------------------
# Logo + título
# -------------------------
col1, col2 = st.columns([1, 5])
with col1:
    st.image("LOGOBUENO.png", width=100)  # Ajusta tamaño si quieres
with col2:
    st.markdown("# **GLUCOSCAN**")
    st.caption("Contador inteligente de hidratos de carbono a partir de imágenes 🍏🤖")

# -------------------------
# Función de procesar imagen
# -------------------------
def procesar_imagen(imagen_path, ratio_usuario=10):
    from ultralytics import YOLO
    model = YOLO("best.pt")

    results = model.predict(source=imagen_path, conf=0.4)

    masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    names = results[0].names

    dice_index = [i for i, cls in enumerate(classes) if names[cls] == "Dice"][0]
    dice_mask = masks[dice_index]
    dice_area_px = dice_mask.sum()
    cm2_per_pixel = (1.6**2) / dice_area_px

    volumenes = {}
    for i, cls in enumerate(classes):
        label = names[cls]
        if label != "Dice":
            area_cm2 = masks[i].sum() * cm2_per_pixel

            if label in ["Apple", "Orange", "Cherry", "Grapes"]:
                r = np.sqrt(area_cm2 / np.pi)
                volumen = (4/3) * np.pi * r**3
            elif label in ["Kiwi", "Mango"]:
                a = np.sqrt(area_cm2 / np.pi)
                b = c = a * 0.6
                volumen = (4/3) * np.pi * a * b * c
            elif label == "Pera":
                a = np.sqrt(area_cm2 / np.pi)
                b = c = a * 0.5
                volumen = (4/3) * np.pi * a * b * c
            elif label == "Banana":
                volumen = area_cm2 * 2
            elif label == "Strawberry":
                volumen = (1/3) * area_cm2 * 2
            else:
                volumen = None

            if volumen is not None:
                volumenes[label] = volumen

    info = {}
    for fruta, vol in volumenes.items():
        if fruta in df.index:
            densidad = float(str(df.loc[fruta]["Densidad (g/cm^3)"]).replace(",", "."))
            hc_100g = float(str(df.loc[fruta]["HC por 100 g"]).replace(",", "."))
            masa = densidad * vol
            hc = masa * hc_100g / 100
            raciones = hc / ratio_usuario
            info[fruta] = {
                "volumen": vol,
                "masa": masa,
                "hc": hc,
                "raciones": raciones
            }

    import cv2
    annotated_image = results[0].plot()
    cv2.imwrite("imagen_procesada.jpg", annotated_image)

    return info

# -------------------------
# Pestañas principales
# -------------------------
tab1, tab2 = st.tabs(["📸 Contador", "Sobre Nosotros"])

# -------------------------
# Tab 1: Contador
# -------------------------
with tab1:
    st.header("📷 Calcula las raciones de insulina subiendo una imagen")

    imagen = st.file_uploader("📤 Sube una imagen con un dado visible", type=["jpg", "jpeg", "png"])
    ratio = st.number_input("💉 Ratio de insulina (g HC / unidad)", value=10.0)

    if st.button("🔎 Procesar imagen") and imagen:
        with st.spinner("🔄 Procesando imagen..."):
            with open("imagen_temporal.jpg", "wb") as f:
                f.write(imagen.read())

            try:
                resultados = procesar_imagen("imagen_temporal.jpg", ratio_usuario=ratio)

                st.success("✅ Imagen procesada correctamente")
                st.divider()

                st.subheader("📊 Resultados por alimento:")
                for fruta, datos in resultados.items():
                    st.write(f"**{fruta}**: volumen = {datos['volumen']:.2f} cm³, "
                             f"masa = {datos['masa']:.2f} g, HC = {datos['hc']:.2f} g, "
                             f"raciones = {datos['raciones']:.2f} u insulina")

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
    st.header("Sobre el proyecto")
    col1, col2 = st.columns(2)

    with col1:
        st.image("foto_grupo.jpg", width=400)  # Subir una imagen de tu grupo
    with col2:
        st.markdown("""
        **GlucoScan** es un proyecto desarrollado por estudiantes de la Universidad Politécnica de Madrid.

        Nuestro objetivo es facilitar el conteo de hidratos de carbono a personas con diabetes mediante visión artificial.

        - 👩‍🎓 Estudiantes: Quique, María, Juan, Ana
        - 📍 Universidad: UPM
        - 📅 Año: 2024-2025

        Gracias por confiar en nuestra app. ¡Seguiremos mejorándola cada día! 🚀
        """)

