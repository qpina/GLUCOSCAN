import streamlit as st
import numpy as np
import pandas as pd
df = pd.read_csv("BaseDeDatos.csv")
df = df.dropna(how='all').set_index("Alimento")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from PIL import Image

# Logo + título
col1, col2 = st.columns([1, 5])
with col1:
    st.image("LOGOBUENO.png")  # Ajusta tamaño según el logo
with col2:
    st.markdown("# **GLUCOSCAN**")
    st.caption("Contador inteligente de hidratos de carbono a partir de imágenes 🍏🤖")



# -------- FUNCIÓN PRINCIPAL --------
def procesar_imagen(imagen_path, ratio_usuario=10):
    from ultralytics import YOLO
    model = YOLO("best.pt")  # CAMBIA esto a la ruta real en tu PC

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
                b = c = a * 0.5  # Más estrecho que Kiwi/Mango
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
      
    # Guardar imagen anotada con las detecciones
    import cv2
    annotated_image = results[0].plot()
    cv2.imwrite("imagen_procesada.jpg", annotated_image) 
        
    return info



# -------- INTERFAZ STREAMLIT --------
st.title("📷 Calcula las raciones de insulina subiendo una imagen")

imagen = st.file_uploader("Sube una imagen con un dado visible", type=["jpg", "jpeg", "png"])
ratio = st.number_input("Ratio de insulina (g HC / unidad)", value=10.0)

if st.button("Procesar imagen") and imagen:
    with open("imagen_temporal.jpg", "wb") as f:
        f.write(imagen.read())

    try:
        resultados = procesar_imagen("imagen_temporal.jpg", ratio_usuario=ratio)

        st.write("## Resultados por alimento:")
        for fruta, datos in resultados.items():
            st.write(f"**{fruta}**: volumen = {datos['volumen']:.2f} cm³, "
                     f"masa = {datos['masa']:.2f} g, HC = {datos['hc']:.2f} g, "
                     f"raciones = {datos['raciones']:.2f} u insulina")

        total = sum([d["raciones"] for d in resultados.values()])
        st.markdown(f"### 💉 Ración total recomendada: **{total:.2f} unidades de insulina**")
        
        from PIL import Image
        st.write("### 📸 Imagen procesada con detecciones:")
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
        st.error(f"Ocurrió un error al procesar la imagen: {e}")
        raise


