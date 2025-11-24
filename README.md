## Integrantes

- Cristian Castiblanco (Grupo Virtual: 10895)
- Ronald Monterroza (Grupo Virtual: 10895)
- Luisa Mancilla (Grupo Virtual: 10895)
- Emely Rueda (Grupo Virtual: 10919)

# Clasificación de Especies de Iris

Este proyecto implementa un proceso completo de análisis y modelado para clasificar flores del dataset Iris mediante técnicas de aprendizaje automático. Incluye exploración de datos, visualización, preprocesamiento, entrenamiento de un modelo y evaluación de resultados.

## Variables

El dataset utiliza cuatro variables numéricas como características y una variable objetivo:

1. **SepalLengthCm** (float): Longitud del sépalo en centímetros.
2. **SepalWidthCm** (float): Anchura del sépalo en centímetros.
3. **PetalLengthCm** (float): Longitud del pétalo en centímetros.
4. **PetalWidthCm** (float): Anchura del pétalo en centímetros.
5. **Species** (str): Nombre de la especie de la flor.  
   Categorías:
   - Iris-setosa  
   - Iris-versicolor  
   - Iris-virginica  

En el preprocesamiento, esta variable se convierte a formato numérico mediante codificación.

## Exploración Inicial

Durante la fase de exploración se realizaron las siguientes acciones:

- Visualización de las primeras filas del dataset.
- Inspección de tipos de datos y estructura.
- Estadísticas descriptivas para todas las variables numéricas.
- Verificación de valores nulos.
- Verificación de filas duplicadas.
- Distribución de frecuencia de cada especie.

Esta etapa permite comprender la calidad del dato antes de iniciar el modelado.

## Visualización

Se generaron histogramas para las cuatro variables numéricas con el fin de:

- Identificar la distribución de cada atributo.
- Observar posibles diferencias entre especies.
- Detectar valores atípicos o patrones relevantes.

Las gráficas se organizan en una matriz de 2x2 para una visualización comparativa.

## Preprocesamiento

El preprocesamiento incluye:

- Eliminación de la columna **Id**, debido a que no aporta información al modelo.
- Codificación de la variable **Species** usando `LabelEncoder`.
- Separación de variables independientes (X) y variable objetivo (y).
- División del conjunto de datos en entrenamiento y prueba (80% / 20%).
- Escalado de características mediante `StandardScaler` para normalizar los valores.

Este proceso asegura que los datos estén limpios y correctamente preparados antes del entrenamiento.

## Modelado

El modelo utilizado es un **RandomForestClassifier** con los siguientes parámetros:

- 200 árboles de decisión.
- `random_state=42` para reproducibilidad.
- Entrenamiento utilizando los datos escalados.

El modelo aprende a distinguir las especies a partir de las medidas de sépalo y pétalo.

## Métricas

Se evaluó el rendimiento del modelo utilizando:

- **Accuracy**
- **Precisión**
- **Recall**
- **F1 Score**

Estas métricas permiten medir el desempeño general del modelo y su capacidad de clasificación.

## Matriz de Confusión

Se generó una matriz de confusión para analizar:

- Aciertos por categoría.
- Errores cometidos en cada clase.
- Relación entre predicciones reales y predichas.

Esta visualización facilita identificar posibles confusiones entre especies similares.

## Importancia de Variables

El modelo proporciona la importancia de cada característica. Se incluye:

- Gráfico de barras comparando la relevancia de cada variable.
- Listado de valores numéricos ordenados según su peso en el modelo.

Este análisis permite interpretar qué variables contribuyen más al proceso de clasificación.

## Uso

Este proyecto puede utilizarse para actividades de:

- Entrenamiento de modelos de clasificación.
- Prácticas de preprocesamiento y análisis exploratorio.
- Evaluación de algoritmos de aprendizaje automático.
- Interpretación de modelos basados en árboles.


