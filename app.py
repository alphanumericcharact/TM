import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

# Almacenar el modelo en caché para optimizar el rendimiento
@st.cache_resource
def cargar_modelo():
    return load_model('keras_model.h5')

model = cargar_modelo()

st.title("Reconocimiento de celulares, toma una foto")

with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar tu dispositivo.")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Leer la imagen y prepararla
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalizar la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Ejecutar la predicción
    prediction = model.predict(data)
    
    # Extraer las probabilidades para mayor claridad
    prob_celular = prediction[0][0]
    prob_celunot = prediction[0][1]

    # Lógica de mensajes basados en tus clases (0: Celular, 1: Celunot)
    if prob_celular > 0.5:
        # Muestra un mensaje de éxito
        st.success(f"Celular aprobado (Probabilidad: {prob_celular:.2%})")
        
    elif prob_celunot > 0.5:
        # Muestra un mensaje de advertencia
        st.warning(f"Muestra el celular (Probabilidad de que no sea celular: {prob_celunot:.2%})")
        
    else:
        # En caso de que ninguna clase supere el 50%
        st.info("No estoy muy seguro de lo que veo. Por favor, toma otra foto.")
