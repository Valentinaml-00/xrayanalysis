import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io
from PIL import Image
import imageio.v3 as iio  # más estable que imageio.v2 para DICOM

# ----------------------------
# FUNCIONES DE PROCESAMIENTO
# ----------------------------

def convert_to_grayscale(uploaded_file):
    image_array = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def enhance_contrast(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image)

def detect_edges(image, threshold1=30, threshold2=100):
    return cv2.Canny(image, threshold1, threshold2)

def apply_gaussian_blur(image, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_median_filter(image, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)

def apply_pseudocolor(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)

def background_subtraction_segmentation(image):
    kernel_size = (21, 21)
    background_estim = cv2.GaussianBlur(image, kernel_size, sigmaX=20, sigmaY=20)
    image_wobackground = cv2.subtract(image, background_estim, dtype=cv2.CV_8U)
    _, seeds = cv2.threshold(image_wobackground, 16, 255, cv2.THRESH_BINARY)
    return seeds

def watershed_segmentation(image):
    img_blur = cv2.medianBlur(image, 3)
    _, initial_segmentation = cv2.threshold(img_blur, 85, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    opening = cv2.morphologyEx(initial_segmentation, cv2.MORPH_OPEN, kernel, iterations=2)
    true_background = cv2.dilate(opening, kernel, iterations=3)
    op_distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, true_foreground = cv2.threshold(op_distance, 0.5 * op_distance.max(), 255, 0)
    true_foreground = np.uint8(true_foreground)
    unknown = cv2.subtract(true_background, true_foreground)
    _, markers = cv2.connectedComponents(true_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    img_color[markers == -1] = [255, 0, 0]
    return img_color

def region_growing(img, seed, thresh=10):
    h, w = img.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    img_copy = img.copy()
    seed_point = (min(max(seed[1], 0), w-1), min(max(seed[0], 0), h-1))
    cv2.floodFill(img_copy, mask, seed_point, 255, (thresh,), (thresh,), cv2.FLOODFILL_FIXED_RANGE)
    return img_copy

def convert_to_downloadable(image_array):
    pil_img = Image.fromarray(image_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# ----------------------------
# INTERFAZ STREAMLIT
# ----------------------------

st.set_page_config(page_title="XRay-AIid Educativa", layout="centered")
st.title("🩻 XRay-AIid – Herramienta educativa para análisis de radiografías")

st.markdown("Sube una radiografía en blanco y negro (JPG, PNG, TIF o DICOM) para visualizar mejoras y segmentaciones.")

uploaded_file = st.file_uploader("📤 Sube tu imagen", type=["jpg", "jpeg", "png", "tif", "dcm"])

if uploaded_file:
    if uploaded_file.name.endswith('.dcm'):
        img = iio.imread(uploaded_file)
        if len(img.shape) > 2:
            img = img[..., 0]  # Si tiene varios canales, toma uno
        img = cv2.convertScaleAbs(img, alpha=(255.0 / img.max()))
    else:
        img = convert_to_grayscale(uploaded_file)

    st.subheader("1. Imagen original")
    st.image(img, clamp=True, use_column_width=True)

    st.subheader("2. Imagen con contraste mejorado")
    contrasted = enhance_contrast(img)
    st.image(contrasted, clamp=True, use_column_width=True)
    st.markdown("🔍 Aumenta el contraste local para destacar diferencias entre tejidos.")

    st.subheader("3. Detección de bordes (Canny)")
    low = st.slider("Umbral bajo (Canny)", 0, 100, 30)
    high = st.slider("Umbral alto (Canny)", 100, 300, 100)
    edges = detect_edges(contrasted, low, high)
    st.image(edges, clamp=True, use_column_width=True)
    st.markdown("🔍 Resalta contornos bruscos en la imagen (fracturas, bordes de órganos).")

    st.subheader("4. Filtro Gaussiano (reducción de ruido suave)")
    kernel_gauss = st.slider("Tamaño del kernel (Gaussiano)", 3, 21, 5, step=2)
    blurred = apply_gaussian_blur(img, kernel_gauss)
    st.image(blurred, clamp=True, use_column_width=True)
    st.markdown("🔍 Suaviza detalles finos para reducir ruido general.")

    st.subheader("5. Filtro de Mediana (reducción de ruido más fuerte)")
    kernel_median = st.slider("Tamaño del kernel (Mediana)", 3, 21, 5, step=2)
    median = apply_median_filter(img, kernel_median)
    st.image(median, clamp=True, use_column_width=True)
    st.markdown("🔍 Elimina ruido salpicado sin afectar bordes importantes.")

    st.subheader("6. Imagen con pseudocolor (falso color)")
    pseudo = apply_pseudocolor(contrasted)
    st.image(pseudo, use_column_width=True)
    st.markdown("🔍 Asigna colores a diferentes intensidades para resaltar zonas.")

    st.subheader("7. Segmentación por Substracción de Fondo")
    segmented_background = background_subtraction_segmentation(img.copy())
    st.image(segmented_background, clamp=True, use_column_width=True)
    st.markdown("🍚 Útil para separar objetos de un fondo no uniforme.")

    st.subheader("8. Segmentación por Watershed")
    segmented_watershed = watershed_segmentation(img.copy())
    st.image(segmented_watershed, use_column_width=True)
    st.markdown("💧 Separa objetos que se tocan entre sí.")

    st.subheader("9. Segmentación por Crecimiento de Regiones")
    st.markdown("Haz clic en la imagen para seleccionar un punto semilla (usamos sliders como alternativa).")
    seed_x = st.slider("Coordenada X de la semilla", 0, img.shape[1] - 1, int(img.shape[1] / 2))
    seed_y = st.slider("Coordenada Y de la semilla", 0, img.shape[0] - 1, int(img.shape[0] / 2))
    threshold_rg = st.slider("Umbral de crecimiento", 1, 100, 10)

    segmented_region_growing = region_growing(img.copy(), (seed_y, seed_x), threshold_rg)

    img_with_seed = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(img_with_seed, (seed_x, seed_y), 5, (0, 0, 255), -1)
    st.image(img_with_seed, use_column_width=True, caption="Imagen con punto semilla (en rojo)")

    st.image(segmented_region_growing, clamp=True, use_column_width=True)
    st.markdown("🌱 Expande una región a partir de un punto inicial.")

    st.download_button(
        label="💾 Descargar imagen contrastada",
        data=convert_to_downloadable(contrasted),
        file_name="radiografia_contraste.png",
        mime="image/png"
    )

    st.markdown("⚠️ Esta herramienta es **educativa** y no reemplaza un diagnóstico médico.")
