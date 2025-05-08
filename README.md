# Análisis Avanzado de Datos y Sentimientos en E-Commerce Brasileño

Este proyecto analiza exhaustivamente datos de un e-commerce brasileño (Olist) para comprender el comportamiento de los clientes, tendencias de ventas y niveles de satisfacción basados en reseñas. El análisis incluye exploración de datos, análisis exploratorio (EDA) y un modelo de análisis de sentimientos para clasificar las reseñas de los clientes.

## Instalación y Requisitos Previos

### Datasets

Antes de comenzar, es necesario descargar los conjuntos de datos desde Kaggle:
1. Visitar: [Brazilian E-Commerce Dataset](https://www.kaggle.com/code/annastasy/brazilian-e-commerce-eda-nlp-ml/input)
2. Descargar todos los archivos CSV
3. Colocarlos en la carpeta `DATA` del proyecto

### Dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn transformers torch nltk spacy geopandas plotly
```

## 1. Exploración y Preparación de Datos

El primer paso del proyecto consiste en la exploración de cada archivo CSV proporcionado por Olist y la creación de un dataset unificado:

- Análisis de estructura de cada CSV (dimensiones, tipos de datos, valores nulos)
- Limpieza de datos (manejo de valores faltantes y duplicados)
- Selección de columnas relevantes para el análisis
- Normalización de columnas (traducción de categorías, estandarización de fechas)
- Creación de nuevas variables derivadas para enriquecer el análisis
- Unificación de todos los datasets en un único dataframe master para análisis posterior

### Datasets Explorados:
- Pedidos (`orders`)
- Clientes (`customers`)
- Productos (`products`)
- Vendedores (`sellers`)
- Categorías de productos (`product_category_name_translation`)
- Reseñas (`reviews`)
- Pagos (`payments`)
- Ítems de pedido (`order_items`)
- Geolocalización (`geolocation`)

## 2. Análisis Exploratorio de Datos (EDA)

El análisis exploratorio revela patrones, tendencias y relaciones importantes en los datos:

### Segmentación de Clientes
- Análisis de frecuencia de compra y valor de ticket
- Identificación de clientes de compra única vs recurrentes
- Distribución de gasto por segmento

### Análisis de Categorías
- Categorías más populares
- Relación entre categorías y patrones de compra
- Segmentación de categorías por frecuencia y precio

### Análisis Geográfico
- Distribución de compras por estado y región de Brasil
- Categorías más vendidas por estado
- Análisis regional de la actividad comercial

### Análisis de Pagos
- Medios de pago predominantes
- Relación entre el número de cuotas y el gasto promedio
- Patrones de financiación de compras

### Análisis de Satisfacción
- Distribución de calificaciones en reseñas
- Categorías con mejores y peores calificaciones
- Relación entre tiempo de entrega y satisfacción

### Análisis Logístico
- Tiempos de procesamiento, preparación y envío
- Análisis de retrasos en la entrega
- Impacto de la logística en la satisfacción del cliente

### Análisis Temporal
- Tendencias de compras a lo largo del año
- Análisis específico de eventos como Black Friday
- Patrones estacionales de ventas

# 3. Análisis de Sentimientos en Reseñas

Esta sección implementa un análisis exhaustivo de los comentarios y reseñas de los clientes, utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático para extraer insights valiosos sobre la percepción de los usuarios.

## 3.1 Análisis Léxico y Nube de Palabras

### Análisis Comparativo de Reseñas Positivas y Negativas

Se realizó un análisis léxico detallado para identificar patrones lingüísticos distintivos entre reseñas positivas y negativas:

#### Distribución de Frecuencias
- **Reseñas positivas**: Alta concentración en léxico asociado a cumplimiento de expectativas, predominio de adjetivos valorativos positivos y referencias a temporalidad favorable.
- **Reseñas negativas**: Concentración en léxico de incumplimiento, problemas logísticos, procesos de reclamación y defectos específicos de producto.

#### Términos de Alta Frecuencia Compartidos

| Término | Frec. Positivas | Frec. Negativas | Ratio P/N | Contexto diferencial |
|---------|-----------------|-----------------|-----------|---------------------|
| produto | 10,456 | 6,127 | 1.71 | Asociado a calidad vs. problemas |
| prazo | 6,885 | 896 | 7.68 | "dentro do prazo" vs. "fora do prazo" |
| entrega | 4,538 | 1,393 | 3.26 | Satisfacción vs. problemas logísticos |
| chegou | 3,946 | 1,167 | 3.38 | Confirmación positiva vs. contexto de demora |
| bom | 3,922 | 207 | 18.95 | Valoración directa vs. uso contextual |
| qualidade | 2,111 | 418 | 5.05 | Atributo positivo vs. "falta de qualidade" |

#### Léxico Exclusivo de Reseñas Positivas
Términos distintivos con frecuencia >500 incluyen: "recomendo" (3,707), "bem" (2,672), "entregue" (2,151), "ótimo" (1,730), "excelente" (1,729), reflejando alta satisfacción y superación de expectativas.

#### Léxico Exclusivo de Reseñas Negativas
Términos distintivos con frecuencia >200 incluyen: "recebi" (3,253), "comprei" (1,863), "veio" (1,535), "nao" (935), "quero" (494), reflejando problemas en el ciclo de compra y deficiencias en comunicación post-venta.

### Visualización mediante Nube de Palabras

Se generaron nubes de palabras para visualizar gráficamente la distribución de términos más frecuentes en:
- Reseñas positivas (calificaciones 4-5)
- Reseñas negativas (calificaciones 1-2)
- Reseñas neutras (calificación 3)

Esta visualización permite identificar rápidamente las palabras clave que caracterizan cada tipo de experiencia del cliente.

## 3.2 Implementación de Modelo de Análisis de Sentimientos

### Modelo Pre-entrenado de HuggingFace

Se implementó un pipeline de análisis de sentimientos utilizando un modelo BERT específico para portugués:

```python
from transformers import pipeline

# Carga del modelo pre-entrenado para portugués
sentiment_analyzer = pipeline("sentiment-analysis", 
                             model="lipaoMai/BERT-sentiment-analysis-portuguese",
                             tokenizer="lipaoMai/BERT-sentiment-analysis-portuguese")
```

### Procesamiento y Clasificación de Reseñas

Se procesaron 40,977 reseñas a través del modelo para obtener una clasificación binaria (positivo/negativo):

```python
# Aplicando el modelo a las reseñas
results = []
for review in tqdm(df_reviews['review_text'].tolist()):
    try:
        result = sentiment_analyzer(review)
        results.append(result[0])
    except:
        results.append({'label': 'ERROR', 'score': 0.0})
        
# Creando columnas con los resultados
df_reviews['sentiment_label'] = [r['label'] for r in results]
df_reviews['sentiment_score'] = [r['score'] for r in results]
```

### Matriz de Confusión y Evaluación del Modelo

**Matriz de confusión con 40,977 reseñas:**

|                | **Predicho Negativo** | **Predicho Positivo** |
|----------------|--------------------|--------------------|
| **Real Negativo** | 9,194              | 1,696              |
| **Real Positivo** | 3,571              | 26,516             |

**Métricas de evaluación:**
- **Precisión global**: 87.4%
- **Precisión para clase positiva**: 94.0%
- **Recall para clase positiva**: 88.1%
- **Precisión para clase negativa**: 72.0%
- **Recall para clase negativa**: 84.4%

**Análisis de resultados:**
- El modelo muestra un buen rendimiento general, con alta precisión en la clasificación de reseñas positivas.
- La precisión en la clasificación de reseñas negativas es menor, indicando que algunos comentarios negativos se están clasificando incorrectamente como positivos.
- La estructura del modelo BERT para portugués logra capturar eficazmente matices lingüísticos específicos del idioma.

## 3.3 Modelos Predictivos Adicionales

Se implementaron tres modelos de aprendizaje automático para comparar su rendimiento en la clasificación de sentimientos:

### Implementación de Modelos

```python
# Preparación de datos
X = df_features[feature_columns]
y = df_features['review_score']

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelos implementados
models = {
    'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42),
    'Regresión Logística' (balanceado):LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
    'Árbol de Decisión (balanceado)': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Entrenamiento y evaluación
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Modelo: {name}")
    print(classification_report(y_test, y_pred))
```

### Análisis de Resultados por Modelo

#### Tendencias Generales

- **Random Forest** demostró el mejor rendimiento general, obteniendo los AUC más altos en todas las clases. Su capacidad para manejar interacciones complejas en los datos le otorga ventaja sobre modelos más simples.
  
- **Regresión Logística** mostró un desempeño estable pero no sobresaliente, con valores de AUC moderados en todas las clases.
  
- **Árbol de Decisión balanceado** presentó el peor desempeño, especialmente en clases como la 1 y la 2, indicando que la estrategia de balanceo puede estar reduciendo su capacidad predictiva.

#### Rendimiento en Clases Críticas (1 y 5)

- **Clase 1 (muy negativa):** 
  - Random Forest (AUC = 0.77) y Árbol de Decisión estándar (AUC = 0.74) destacaron como las mejores opciones
  - Árbol de Decisión balanceado (AUC = 0.58) mostró un mal ajuste
  - Estos resultados sugieren que modelos más flexibles pueden captar mejor las características de reseñas muy negativas

- **Clase 5 (muy positiva):** 
  - Random Forest (AUC = 0.64) y Árbol de Decisión (AUC = 0.63) lideraron por un margen pequeño
  - Todos los modelos tuvieron un rendimiento moderado
  - La clase 5 parece ser más fácilmente diferenciable, pero aún podría beneficiarse de un modelo más optimizado

### Importancia de Características

Se analizó la importancia de las diferentes características en los modelos:

```python
# Para Random Forest
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualización
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances.head(15))
plt.title('Top 15 características más importantes (Random Forest)')
plt.tight_layout()
plt.show()
```

Las características más importantes incluyeron:
1. Tiempo de entrega (delivery_delay)
2. Precio del producto (price)
3. Tiempo de procesamiento del pedido (order_processing_time)
4. Categoría del producto

## 3.4 Conclusiones y Recomendaciones

### Conclusiones Técnicas

- **Random Forest es la mejor opción global**, especialmente para la clase 1 (reseñas muy negativas), que es la más difícil de clasificar correctamente.
  
- **Se debe evitar el Árbol de Decisión balanceado**, ya que su desempeño es inferior en todas las clases a pesar del intento de equilibrar el entrenamiento.
  
- **La Regresión Logística es una alternativa aceptable** si se busca simplicidad, pero no la mejor opción en términos de desempeño puro.

- **El modelo pre-entrenado BERT para portugués** tiene un rendimiento notable (87.4% de precisión global), pero podría mejorarse específicamente para la detección de reseñas negativas.

### Recomendaciones para el Negocio

Basado en el análisis léxico y los modelos predictivos:

1. **Implementar sistema predictivo de detección temprana de insatisfacción**:
   - Monitorear menciones de términos negativos de alta frecuencia en comunicaciones iniciales
   - Utilizar el modelo de Random Forest para clasificar proactivamente reseñas potencialmente negativas

2. **Optimizar procesos logísticos**:
   - El cumplimiento de plazos aparece como factor crítico de satisfacción
   - Priorizar visibilidad y transparencia en estado de envíos
   - Reducir los tiempos de entrega, identificado como la característica más importante en la predicción

3. **Reforzar comunicación post-venta**:
   - Establecer protocolos de respuesta rápida ante términos clave de insatisfacción
   - Implementar seguimiento proactivo en casos de alta probabilidad de insatisfacción

4. **Revisión de políticas de devolución y compensación**:
   - Simplificar procesos de devolución y reembolso
   - Considerar compensaciones estratégicas en casos críticos

### Próximos Pasos

Para mejorar aún más los resultados, se recomienda:
- Ajustar hiperparámetros en Random Forest para optimizar su rendimiento
- Explorar técnicas de ingeniería de características para fortalecer la diferenciación entre clases
- Implementar un sistema de monitoreo continuo de sentimientos que alerte sobre cambios en la percepción de los clientes
- Integrar los resultados del análisis de sentimientos con datos operativos para identificar áreas específicas de mejora

## Conclusiones y Recomendaciones

### Principales Hallazgos
- Alta proporción de clientes de compra única, indicando baja fidelización
- El segmento de ticket medio (75-350 BRL) concentra la mayor parte del gasto y volumen de pedidos
- Los clientes de alto valor tienen poca recurrencia pero generan ingresos relevantes
- Existe una relación inversa entre frecuencia de compra y valor del ticket
- El tiempo de entrega impacta significativamente en la satisfacción del cliente

### Recomendaciones Estratégicas
- Implementar estrategias de retención para reducir la proporción de compras únicas
- Optimizar precios y promociones en el rango de 75-350 BRL
- Desarrollar programas de fidelización específicos para clientes de alto ticket
- Mejorar la logística de entrega, especialmente en las regiones con mayores retrasos
- Enfocar esfuerzos de mejora en las categorías con peores calificaciones según el análisis de sentimientos

## Autor

[BarbaraGamarra]