import cv2
import numpy as np

# 1. Cargar imagen en escala de grises
def load_image(path):
    """
    Carga una imagen en escala de grises desde el disco.

    Parámetros:
        path (str): Ruta de la imagen.

    Retorna:
        np.ndarray: Imagen en escala de grises.
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# 2. Aplicar filtro de mediana
def apply_median_filter(img, ksize=5):
    """
    Aplica un filtro de mediana para eliminar ruido.

    Parámetros:
        img (np.ndarray): Imagen de entrada.
        ksize (int): Tamaño del kernel (debe ser impar). Por defecto 5.

    Retorna:
        np.ndarray: Imagen suavizada.
    """
    return cv2.medianBlur(img, ksize)

# 3. Aplicar filtro gaussiano
def apply_gaussian_blur(img, ksize=(9, 9), sigma=0):
    """
    Aplica un filtro gaussiano para suavizar bordes suaves y reducir ruido.

    Parámetros:
        img (np.ndarray): Imagen de entrada.
        ksize (tuple): Tamaño del kernel (debe ser impar y positivo). Por defecto (9,9).
        sigma (float): Desviación estándar en X e Y. 0 significa que se calcula automáticamente.

    Retorna:
        np.ndarray: Imagen suavizada.
    """
    return cv2.GaussianBlur(img, ksize, sigma)

# 4. Aplicar corrección gamma
def apply_gamma_correction(img, gamma=1.5):
    """
    Aplica corrección gamma para ajustar el brillo y mejorar contraste local.

    Parámetros:
        img (np.ndarray): Imagen de entrada (normalizada entre 0 y 255).
        gamma (float): Valor de gamma. >1 oscurece sombras, <1 aclara.

    Retorna:
        np.ndarray: Imagen con corrección gamma.
    """
    return np.array(255 * (img / 255) ** (1 / gamma), dtype='uint8')

# 5. Aplicar umbralización adaptativa
def apply_adaptive_threshold(img, block_size=31, C=10):
    """
    Aplica umbralización adaptativa basada en la media local gaussiana.

    Parámetros:
        img (np.ndarray): Imagen de entrada.
        block_size (int): Tamaño del área local (debe ser impar).
        C (int): Valor constante que se resta de la media.

    Retorna:
        np.ndarray: Imagen binaria donde se resaltan detalles.
    """
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, C)

# 6. Detectar bordes con filtro laplaciano
def apply_laplacian_edge(img):
    """
    Aplica el operador Laplaciano para detectar cambios de intensidad bruscos.

    Parámetros:
        img (np.ndarray): Imagen binaria o en escala de grises.

    Retorna:
        np.ndarray: Imagen con bordes resaltados.
    """
    return cv2.Laplacian(img, cv2.CV_8U)

# 7. Filtrar contornos grandes
def filter_large_contours(img, min_area=25): # La area minima se establecio segun la resolucion de la imagen 256 x 256 px
    """
    Extrae contornos y filtra aquellos con área mayor a un umbral.

    Parámetros:
        img (np.ndarray): Imagen binaria con bordes.
        min_area (float): Área mínima para conservar un contorno.

    Retorna:
        list: Lista de contornos grandes.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# 8. Dibujar contornos sobre la imagen original
def draw_contours(original_gray_img, contours):
    """
    Dibuja contornos sobre una imagen original (convertida a BGR).

    Parámetros:
        original_gray_img (np.ndarray): Imagen base en escala de grises.
        contours (list): Lista de contornos a dibujar.

    Retorna:
        np.ndarray: Imagen con contornos resaltados (en rojo).
    """
    img_color = cv2.cvtColor(original_gray_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_color, contours, -1, (0, 0, 255), 1)
    return img_color

# Pipeline de procesamiento completo
def process_crack_detection_pipeline(image_path):
    """
    Ejecuta un pipeline completo para detectar grietas en una imagen:
    incluye suavizado, realce de bordes, umbralización adaptativa
    y detección de contornos significativos.

    Parámetros:
        image_path (str): Ruta del archivo de imagen.

    Retorna:
        tuple: Imagen final con grietas resaltadas, y lista de contornos detectados.
    """
    img = load_image(image_path)
    img_median = apply_median_filter(img)
    img_gauss = apply_gaussian_blur(img_median)
    gamma_corrected = apply_gamma_correction(img_gauss)
    thresh = apply_adaptive_threshold(gamma_corrected)
    laplacian = apply_laplacian_edge(thresh)
    big_cracks = filter_large_contours(laplacian)
    img_contours = draw_contours(img, big_cracks)

    return img_contours, big_cracks


