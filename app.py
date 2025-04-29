import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------
# Configuración de la página
# -------------------------
st.set_page_config(page_title="GlucoScan", layout="wide")

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
                if label not in volumenes:
                    volumenes[label] = []
                volumenes[label].append(volumen)

    info = {}
    for fruta, lista_vol in volumenes.items():
        for i, vol in enumerate(lista_vol):
            nombre = f"{fruta} {i+1}"  # Banana 1, Banana 2, etc.
            if fruta in df.index:
                densidad = float(str(df.loc[fruta]["Densidad (g/cm^3)"]).replace(",", "."))
                hc_100g = float(str(df.loc[fruta]["HC por 100 g"]).replace(",", "."))
                masa = densidad * vol
                hc = masa * hc_100g / 100
                raciones = hc / ratio_usuario
                info[nombre] = {
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
tab1, tab2, tab3 = st.tabs(["📸 Contador", "❓ Cómo usar la app", "👨🏽‍💻 Sobre Nosotros"])

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
# Tab 2: ¿Cómo usar la app?
# -------------------------
with tab2:
    st.header("¿Cómo usar la aplicación?")
    st.markdown("""
    Sigue estos pasos para usar GlucoScan de manera correcta:

    ### 1️⃣ Prepara la imagen
    - Situa los alimentos que quieres analizar en una superficie plana.
    - Coloca nuestro dado de referenica apoyado sobre la cara del logo.
                """)
                
    st.image("foto_preparacion.jpg", caption="Ejemplo de imagen correcta", use_container_width=True)

    st.markdown("""

    ### 2️⃣ Sube tu imagen
    - Dirígete a la pestaña 📸 **Contador**.
    - Usa el botón para subir una foto en formato `.jpg` o `.png`. 
    - Puedes subirla desde tu galería o hacer la foto desde la propia app.            
                

    ### 3️⃣ Ajusta tu ratio de insulina
    - Introduce tu ratio personal de insulina (por ejemplo: 10 g de HC por unidad de insulina).

    ### 4️⃣ Procesa la imagen
    - Pulsa el botón "🔎 Procesar imagen".
    - Espera unos segundos mientras la app detecta los alimentos y el dado.

    ### 5️⃣ Consulta los resultados
    - Verás el volumen, peso, hidratos de carbono y raciones necesarias para cada alimento detectado.
    - Puedes descargar la imagen anotada con las detecciones.
                
    )"""

    st.image("foto_resultados.jpg", caption="Ejemplo de resultados", use_container_width=True)

    st.markdown("""
    🎯 ¡Así de fácil puedes controlar tu alimentación de forma automática!
    """)


# -------------------------
# Tab 3: Sobre Nosotros
# -------------------------
with tab3:
    st.header("Sobre el proyecto")
    col1, col2 = st.columns(2)

    with col1:
        st.image("foto_grupo.jpg", use_container_width=True)  # Subir una imagen de tu grupo
    with col2:
        st.markdown("""
        **GlucoScan** es una aplicación médica desarrollada por un grupo de alumnos de Máster de l universidad Politécnica de Madrid para la asignatura 
        Ingenia Diseño en Bioingenierí - Medtech.
                    
        El objetivo de la aplicación es poder facilitar el control preciso y autónomo de la alimentación en personas 
                    con diabetes Tipo I, mediante el cálculo automatizado de la dosis de insulina a partir del reconocimiento de alimentos. 


        - 👩‍🎓 Estudiantes: Aitana Carrillo, Blanca Santón, Juan García, Jon Beristain, María Fernández-Cordeiro, Marina Durán, Enrique Pina y Sofía Vigara
        - 📅 Año: 2024-2025

        Gracias por probar nuestra app. ¡Esperemos que os sea util!
        """)

