# Cloud Segmentation with U-Net

Proyecto de visión por computador enfocado en la segmentación de nubes sobre imágenes satelitales utilizando redes neuronales convolucionales (U-Net) en PyTorch.

## Objetivo
Entrenar un modelo capaz de recibir un parche de imagen satelital y generar una máscara binaria que identifique con precisión la cobertura nubosa, sirviendo como base técnica para el procesamiento de datos geoespaciales y meteorológicos con Deep Learning.

## Tecnologías
* **Lenguaje:** Python 3
* **Deep Learning:** PyTorch
* **Procesamiento y Visualización:** NumPy, Matplotlib, Pillow / OpenCV
* **Entorno de entrenamiento:** Google Colab (GPU)

## Estructura del proyecto
* `notebooks/`: Cuadernos de Colab con el flujo completo (carga de datos, entrenamiento y evaluación).
* `src/`: Módulos en Python con la arquitectura U-Net y el `Dataset` personalizado.
* `models/`: Pesos entrenados del modelo (ignorados en Git).

## Resultados
*(imagen comparativa: Imagen Satelital | Máscara Real | Predicción de la U-Net una vez entrenado el modelo).*
