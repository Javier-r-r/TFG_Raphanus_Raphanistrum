# **Análisis del Patrón de Venación Foliar en Raphanus raphanistrum L. bajo Estrés Salino**

Este repositorio contiene el código, datos y experimentos realizados para el Trabajo de Fin de Grado (TFG) titulado:

**“Variación de patrones de venación foliar en hojas de *Raphanus raphanistrum* L. sometidas a estrés salino”**

---

## **Descripción del Proyecto**
El objetivo principal de este trabajo es analizar cómo el estrés salino afecta la morfología y el patrón de venación foliar en hojas de *Raphanus raphanistrum* L. Para ello se emplean técnicas de procesamiento de imágenes y análisis computacional que permiten extraer métricas cuantitativas sobre la estructura vascular.

---

## **Estructura del Repositorio**
- **`database_segmentation.py`**: Script para gestionar la base de datos y segmentación de imágenes.
- **`mask_generator.py`**: Generación de máscaras para análisis de venación.
- **`model.py`**: Modelos utilizados para la segmentación y clasificación.
- **`experiments.sh` / `test_all_models.sh`**: Scripts para ejecutar experimentos y pruebas.
- **`ground_truth.xml`**: Datos de referencia para validación.
- **Carpetas**:
  - `petalos_iguales_224` y `petalos_iguales_mascara_224`: Imágenes y máscaras utilizadas en el análisis.
  - `segmentation-app`: Aplicación para segmentación.

---

## **Tecnologías Utilizadas**
- **Lenguajes**: Python (90.9%), Shell (9.1%)
- **Bibliotecas principales**:
  - OpenCV
  - NumPy
  - Matplotlib
  - Scikit-learn / TensorFlow

---

## **Objetivos Específicos**
1. Preprocesamiento y segmentación de imágenes foliares.
2. Extracción de métricas de venación (longitud, densidad, conectividad).
3. Comparación entre hojas sometidas a diferentes niveles de estrés salino.

---

## **Cómo Ejecutar**
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Javier-r-r/TFG_Raphanus_Raphanistrum.git
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar experimentos:
   ```bash
   bash experiments.sh
   ```

---

## **Licencia**
Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
