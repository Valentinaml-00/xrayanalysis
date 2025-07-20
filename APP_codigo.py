import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# ---------------------
# FUNCIONES DE FILTROS
# ---------------------
def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_median_filter(img, ksize=5):
    return cv2.medianBlur(img, ksize)

def apply_gaussian_filter(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def adjust_gamma(image, gamma_value=1.0):
    inv_gamma = 1.0 / gamma_value
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# ---------------------
# INTERFAZ STREAMLIT
# ---------------------
st.set_page_config(page_title="Análisis de Rayos X", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2e7bcf;'>🩻 Análisis de Imágenes de Rayos X</h1>", unsafe_allow_html=True)
st.markdown("Aplicación simple para mejorar imágenes médicas con filtros. Carga una imagen y prueba distintas mejoras visuales.")

uploaded_file = st.file_uploader("📤 Sube una imagen de rayos X", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Imagen original", use_column_width=True)

    with st.sidebar:
        st.header("🛠️ Filtros")
        filtro = st.selectbox("Selecciona un filtro", ["Ninguno", "Escala de grises", "Filtro mediana", "Filtro gaussiano", "Corrección gamma"])
        gamma_value = st.slider("Valor de gamma (si aplica)", 0.1, 3.0, 1.0, step=0.1)

    if filtro == "Escala de grises":
        result = convert_to_grayscale(image)
        result_display = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    elif filtro == "Filtro mediana":
        result = apply_median_filter(image)
        result_display = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    elif filtro == "Filtro gaussiano":
        result = apply_gaussian_filter(image)
        result_display = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    elif filtro == "Corrección gamma":
        result = adjust_gamma(image, gamma_value)
        result_display = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    else:
        result = image
        result_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with col2:
        st.image(result_display, caption="Imagen procesada", use_column_width=True)

    st.success("✅ Imagen procesada con éxito.")

st.markdown("""---""")
st.markdown("<p style='text-align: center; font-size: 14px;'>Creado por Valentina Miguel | GitHub: <a href='https://github.com/Valentinaml-00' target='_blank'>@Valentinaml-00</a></p>", unsafe_allow_html=True)
