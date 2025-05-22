import os
from src.filtros import process_crack_detection_pipeline
import cv2

# Rutas de entrada y salida
input_path = 'data/img/c.jpg'
output_dir = 'data/output'
output_path = os.path.join(output_dir, 'b_resultado.jpg')

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Ejecutar pipeline
output_img, cracks = process_crack_detection_pipeline(input_path)

# Guardar imagen con grietas resaltadas
cv2.imwrite(output_path, output_img)

print(f"Imagen procesada guardada en: {output_path}")
print(f"Grietas detectadas: {len(cracks)}")
