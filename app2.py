import streamlit as st
from ultralytics import YOLO
import pandas as pd
import numpy as np
from PIL import Image
import time

# -------------------
# Configurar la página
# -------------------
st.set_page_config(
    page_title="GlucoScan 2.0 🍏",
    page_icon="🍏",
    layout="wide"
)

# Opcional: fondo personalizado
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1506784983877-45594efa4cbe");
background-size: cover;
background-position: center;
background-attachment: fixed;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# -------------------
# Cabecera
# -------------------
st.title("📷 Calculadora de Raciones de Insulina")
st.caption("Sube una imagen con alimentos y un dado de referencia para calcular hidratos y raciones.")

st.divider()

# -------------------
# Cargar modelo y base de datos
# -------------------
@st.cache_resource
def cargar_modelo():
    model = YOLO("best.pt")
    return model

@st.cache_data
def cargar_base_datos():
    df = pd.read_csv("BaseDeDatos.csv")
    df = df.dropna(how='all')
    df.columns = df.columns.str.strip()  # Limpia espacios en columnas
    return df

modelo = cargar_modelo()
base_datos = cargar_base_datos()

# -------------------
# Sidebar
# -------------------
with st.sidebar:
    st.header("⚙️ Opciones")
    ratio_usuario = st.number_input(
        "Ratio de insulina (g HC / unidad de insulina)", 
        min_value=1.0, 
        max_value=20.0, 
        value=10.0
    )
    st.markdown("---")
    st.write("Instrucciones:")
    st.write("- Foto superior del alimento.")
    st.write("- Dado de 1.4cm visible.")
    st.markdown("---")
    st.info("GlucoScan 2.0 funcionando en modo beta.")

# -------------------
# Subir imagen
# -------------------
uploaded_file = st.file_uploader("📸 Sube tu imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    imagen = Image.open(uploaded_file)

    # Mostrar imagen
    st.image(imagen, caption="Imagen subida", use_container_width=True)

    # Botón de procesar
    if st.button("🚀 Procesar imagen"):
        with st.spinner("Detectando alimentos y calculando hidratos..."):
            # Procesar predicción
            results = modelo.predict(imagen, conf=0.4)

            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            names = results[0].names

            # Buscar dado para escala
            dice_index = [i for i, cls in enumerate(classes) if names[cls] == "Dice"][0]
            dice_mask = masks[dice_index]
            dice_area_px = dice_mask.sum()
            cm2_per_pixel = (1.4**2) / dice_area_px

            # Cálculo de hidratos
            volumenes = {}
            for i, cls in enumerate(classes):
                label = names[cls]
                if label == "Dice":
                    continue

                area_px = masks[i].sum()
                area_cm2 = area_px * cm2_per_pixel

                # Estimación de volumen
                if label in ["Apple", "Orange", "Cherry", "Grapes", "Pear"]:
                    r = np.sqrt(area_cm2 / np.pi)
                    volumen = (4/3) * np.pi * r**3
                elif label in ["Kiwi", "Mango"]:
                    a = np.sqrt(area_cm2 / np.pi)
                    b = c = a * 0.6
                    volumen = (4/3) * np.pi * a * b * c
                elif label == "Banana":
                    h = 2
                    volumen = area_cm2 * h
                elif label == "Strawberry":
                    h = 2
                    volumen = (1/3) * area_cm2 * h
                else:
                    volumen = None

                if volumen is not None:
                    volumenes[label] = volumen

            info_nutricional = {}

            for fruta, volumen in volumenes.items():
                if fruta in base_datos.set_index("Alimento").index:
                    fila = base_datos.set_index("Alimento").loc[fruta]
                    densidad = fila["Densidad (g/cm^3)"]
                    hc_100g = fila["HC por 100 g"]

                    masa = densidad * volumen
                    hc = (masa * hc_100g) / 100
                    raciones = hc / ratio_usuario

                    info_nutricional[fruta] = {
                        "Volumen (cm3)": volumen,
                        "Masa (g)": masa,
                        "HC (g)": hc,
                        "Raciones": raciones
                    }

        st.success("✅ Análisis completo!")

        # Mostrar resultados
        st.subheader("📊 Resultados por alimento:")
        resultados_df = pd.DataFrame(info_nutricional).T
        st.dataframe(resultados_df)

        # Mostrar ración total
        racion_total = resultados_df["Raciones"].sum()
        st.subheader(f"📊 Ración total estimada: {racion_total:.2f} unidades de insulina")
