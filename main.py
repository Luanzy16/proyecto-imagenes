import cv2
import numpy as np
from pathlib import Path
from src.filtros import process_crack_detection_pipeline

# Carpeta de entrada y salida
carpeta_entrada = Path("data/img")
carpeta_salida = Path("data/output")
carpeta_salida.mkdir(exist_ok=True)

# Extensiones válidas
extensiones_validas = [".jpg", ".png", ".jpeg"]

# Procesamiento de imágenes
for ruta_imagen in carpeta_entrada.glob("*"):
    if ruta_imagen.suffix.lower() not in extensiones_validas:
        continue

    print(f"Procesando: {ruta_imagen.name}")

    try:
        # Desempaquetar tupla de salida
        img_resultado, contornos = process_crack_detection_pipeline(str(ruta_imagen))

        if isinstance(img_resultado, np.ndarray):
            salida_path = carpeta_salida / f"resultado_{ruta_imagen.stem}.png"
            cv2.imwrite(str(salida_path), img_resultado)
        else:
            print(f"⚠️ No se obtuvo una imagen válida para {ruta_imagen.name}")

    except Exception as e:
        print(f"❌ Error procesando {ruta_imagen.name}: {e}")

print("✅ Procesamiento completado.")
