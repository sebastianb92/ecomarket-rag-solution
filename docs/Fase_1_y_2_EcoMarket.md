<img src="https://upload.wikimedia.org/wikipedia/commons/6/68/Logo_universidad_icesi.svg" width="220">

# EcoMarket AI Support — Sistema RAG
## Informe de Diseño y Arquitectura

**Maestría en Inteligencia Artificial · IA Generativa · Taller Práctico #2**  
**Integrantes:** Johan Sebastian Bonilla · Edwin Gómez

---

# Fase 1: Selección de Componentes Clave del Sistema RAG

Antes de comenzar la implementación, es fundamental tomar decisiones de arquitectura informadas. En esta fase se documenta la selección y justificación de los dos componentes centrales del pipeline RAG: el modelo de embeddings y la base de datos vectorial.

---

## 1.1 Modelo de Embeddings

El modelo de embeddings transforma fragmentos de texto en vectores numéricos de alta dimensión, de forma que documentos semánticamente similares queden cercanos en el espacio vectorial. La elección de este componente impacta directamente la calidad de la recuperación.

### Modelo seleccionado: `intfloat/multilingual-e5-large`

Para EcoMarket se seleccionó el modelo **multilingual-e5-large** de Hugging Face (código abierto, disponible sin costo). A continuación se justifica esta decisión frente a las alternativas.

| Criterio | `multilingual-e5-large` ✅ | `text-embedding-3-small` (OpenAI) | `paraphrase-multilingual-mpnet-base-v2` (SBERT) |
|---|---|---|---|
| **Idioma español** | Excelente — entrenado explícitamente para recuperación semántica en +100 idiomas | Bueno — soporta español pero optimizado para inglés | Bueno — multilingüe pero menor precisión en recuperación |
| **Dimensiones del vector** | 1024 — alta capacidad semántica | 1536 | 768 — menor capacidad |
| **Costo** | Gratuito (open-source, corre en GPU/CPU local) | Pago por token (~$0.02 / 1M tokens) | Gratuito (open-source) |
| **Precisión en recuperación** | Estado del arte en benchmarks MTEB para español | Alta, pero dependiente de API externa | Moderada — menos preciso en consultas informacionales |
| **Dependencia externa** | Ninguna — corre localmente (Colab / servidor) | Requiere API key y conexión a internet | Ninguna — corre localmente |
| **Uso en el proyecto** | ✅ Implementado | — | — |

### Justificación

La elección de `multilingual-e5-large` responde a tres factores determinantes para el caso de EcoMarket:

- **Dominio bilingüe:** Los documentos están en español (política de devoluciones, FAQ) y las consultas de usuarios también. Un modelo entrenado explícitamente para recuperación semántica en español garantiza mejor calidad que modelos anglocéntricos.

- **Costo cero:** Al tratarse de un proyecto académico desplegado en Google Colab, el uso de un modelo open-source elimina la dependencia de APIs externas de pago y permite iterar sin restricciones.

- **Precisión:** Los vectores de 1024 dimensiones, normalizados (`normalize_embeddings=True`), permiten calcular similitud coseno de forma eficiente y con mayor resolución semántica que modelos de 384 o 768 dimensiones.

```python
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda"},   # GPU en Colab
    encode_kwargs={"normalize_embeddings": True}
)
```

---

## 1.2 Base de Datos Vectorial

La base de datos vectorial almacena los embeddings generados y desde donde se recuperan los fragmentos más relevantes mediante búsqueda por similitud. La selección debe equilibrar facilidad de uso, costo y escalabilidad para las necesidades actuales y futuras de EcoMarket.

### Comparación de opciones

| Criterio | ChromaDB ✅ | FAISS | Pinecone | Weaviate |
|---|---|---|---|---|
| **Tipo** | Librería local / embebida | Librería local (Facebook AI) | Servicio cloud gestionado | Motor vectorial open-source |
| **Persistencia** | ✅ Nativa en disco (`persist_directory`) | ⚠️ Requiere serialización manual | ✅ Nativa en la nube | ✅ Nativa |
| **Costo** | Gratuito | Gratuito | Pago (free tier muy limitado) | Gratuito self-hosted; pago en cloud |
| **Escalabilidad** | Media — adecuada hasta millones de vectores | Alta — usado en producción a gran escala | Muy alta — diseñado para escala empresarial | Alta — soporte para billones de objetos |
| **Integración LangChain** | ✅ `langchain-chroma` nativo | ✅ `langchain-community` | ✅ `langchain-pinecone` | ✅ `langchain-weaviate` |
| **Filtros por metadatos** | ✅ Incluidos nativamente | ⚠️ Sin soporte nativo | ✅ Filtros avanzados | ✅ Búsqueda híbrida (BM25 + vectores) |
| **Facilidad de uso** | ⭐⭐⭐⭐⭐ setup en 3 líneas | ⭐⭐⭐ requiere más configuración | ⭐⭐⭐⭐ requiere cuenta y API key | ⭐⭐⭐ configuración más compleja |
| **Uso en el proyecto** | ✅ Implementado | — | — | — |

### Justificación de ChromaDB para EcoMarket

1. **Persistencia sin configuración adicional:** ChromaDB guarda automáticamente el índice en `./chroma_langchain_db`, eliminando la necesidad de serializar y deserializar manualmente el índice entre sesiones (limitación de FAISS).

2. **Soporte nativo de metadatos:** Permite filtrar resultados por fuente del documento (Excel, PDF, JSON), lo que facilita depuración y control sobre qué fragmentos se recuperan.

3. **Integración directa con LangChain:** El paquete `langchain-chroma` ofrece una interfaz unificada que simplifica la creación del vector store, la inserción de documentos y la configuración del retriever.

4. **Escalabilidad suficiente para el caso de uso:** El volumen de datos de EcoMarket (pedidos, política de devoluciones, FAQ) es manejable localmente. Pinecone y Weaviate añadirían complejidad operativa innecesaria para este alcance.

5. **Costo cero:** Apropiado para un entorno académico y de prototipado. Si el proyecto escala a producción, ChromaDB puede reemplazarse por Pinecone o Weaviate con cambios mínimos en el código gracias a la abstracción de LangChain.

```python
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
```

---

# Fase 2: Creación de la Base de Conocimiento

El desempeño de un sistema RAG depende directamente de la calidad y cobertura de su base de conocimiento. En esta fase se describe la identificación de documentos, la estrategia de segmentación (chunking) y el proceso de indexación implementado para EcoMarket.

---

## 2.1 Identificación de Documentos

Para cubrir los dos flujos de atención al cliente (estado de pedidos y devoluciones), se seleccionaron tres tipos de documentos complementarios:

| Tipo de documento | Archivo | Loader utilizado | Contenido y relevancia |
|---|---|---|---|
| **Hoja de cálculo (Excel)** | `pedidos_ecomarket.xlsx` | `DataFrameLoader` | Registro estructurado de pedidos con estado, tracking, fechas estimadas y observaciones. Fuente principal para responder consultas de estado de pedido. |
| **Documento PDF** | `POLÍTICA DE DEVOLUCIONES.pdf` | `PyPDFLoader` | Política oficial de devoluciones: condiciones de elegibilidad, plazos, categorías excluidas y proceso de reembolso. Fuente principal para evaluar solicitudes de devolución. |
| **Archivo JSON** | `FAQ.json` | `JSONLoader` (jq_schema) | Preguntas frecuentes estructuradas por categoría. Complementa los dos flujos con respuestas estandarizadas a consultas comunes sobre procesos y políticas. |

### Decisión de diseño: fuentes heterogéneas

La combinación de tres formatos distintos responde a una realidad frecuente en entornos empresariales: la información no está centralizada en un solo sistema. Los pedidos viven en un ERP exportado a Excel, las políticas existen como documentos PDF para uso legal, y las FAQ se mantienen en JSON para integrarse con CMS. El sistema RAG abstrae estos formatos mediante loaders especializados, unificándolos en una colección vectorial coherente.

```python
# Cada fila del Excel se convierte en un string con todos sus campos
df["contenido"] = df.astype(str).agg(" | ".join, axis=1)
loader = DataFrameLoader(df, page_content_column="contenido")
# Resultado: "ECO-12345 | En tránsito | Ana García | Botella..."

# El JSON se extrae estructuradamente con jq
loader = JSONLoader(
    file_path="data/FAQ.json",
    jq_schema='.faq[] | "Categoría: \(.categoria)\nPregunta: \(.pregunta)\nRespuesta: \(.respuesta)"',
    text_content=True
)
```

---

## 2.2 Estrategia de Segmentación (Chunking)

La segmentación consiste en dividir los documentos en fragmentos más pequeños llamados *chunks*. Esta etapa es crítica: chunks demasiado grandes incluyen información irrelevante y reducen la precisión del retriever; chunks demasiado pequeños pierden contexto y producen respuestas incompletas.

### Estrategias de segmentación disponibles

| Estrategia | Descripción | Ventajas | Desventajas |
|---|---|---|---|
| **Tamaño fijo** | Divide por número exacto de caracteres o tokens, sin considerar estructura del texto. | Simple, predecible, sin dependencias. | Puede cortar a mitad de oraciones, perdiendo coherencia semántica. |
| **Por párrafos** | Divide en los saltos de línea o separadores naturales del documento. | Respeta la estructura; chunks coherentes. | Párrafos de longitud muy variable generan chunks inconsistentes. |
| **Recursiva** ✅ | Intenta dividir por separadores en orden de preferencia: párrafo → oración → palabra → carácter. Solo baja al siguiente nivel si el chunk sigue siendo muy grande. | Equilibrio entre coherencia semántica y control de tamaño. Funciona bien con documentos heterogéneos. | Más compleja de configurar que tamaño fijo. |
| **Semántica** | Agrupa oraciones por similitud semántica usando embeddings. | Máxima coherencia temática dentro de cada chunk. | Costosa computacionalmente; requiere embeddings solo para chunking. |

### Justificación de `RecursiveCharacterTextSplitter`

Para EcoMarket se utilizó `RecursiveCharacterTextSplitter` de LangChain con los siguientes parámetros:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Máximo de caracteres por chunk
    chunk_overlap=200,   # Solapamiento entre chunks consecutivos
    add_start_index=True # Trazabilidad: índice en el documento original
)
```

La elección de estos parámetros responde a tres justificaciones:

- **`chunk_size=1000`:** Suficiente para incluir una entrada completa del Excel (número de pedido + estado + observaciones) o un párrafo de política de devoluciones con su contexto completo, sin exceder la ventana de contexto del LLM.

- **`chunk_overlap=200`:** El solapamiento del 20% garantiza que la información que cae en el borde entre dos chunks no se pierda. Por ejemplo, si la condición de devolución de un producto se menciona a caballo entre dos fragmentos, ambos chunks la contendrán parcialmente.

- **Estrategia recursiva frente a tamaño fijo:** Los tres documentos de EcoMarket tienen estructuras muy distintas (tabular, prosa legal, JSON transformado). La estrategia recursiva se adapta a cada caso respetando primero los separadores naturales del texto antes de recurrir al corte por caracteres.

---

## 2.3 Proceso de Indexación

La indexación es el proceso de convertir los chunks en vectores numéricos y almacenarlos en ChromaDB de forma que puedan recuperarse eficientemente mediante búsqueda por similitud semántica.

### Pasos del pipeline

| Paso | Operación | Componente responsable |
|---|---|---|
| **1. Carga** | Lectura de los documentos fuente (Excel, PDF, JSON) y conversión a objetos `Document` de LangChain. | `DataFrameLoader`, `PyPDFLoader`, `JSONLoader` |
| **2. Unificación** | Concatenación de los tres listados en una única colección. | Python (lista) |
| **3. Fragmentación** | División recursiva en chunks de 1000 chars con overlap de 200. | `RecursiveCharacterTextSplitter` |
| **4. Limpieza** | Eliminación de metadatos con tipos incompatibles que ChromaDB no puede serializar. | `filter_complex_metadata` |
| **5. Vectorización** | Conversión de cada chunk a un vector de 1024 dimensiones. | `HuggingFaceEmbeddings` (`multilingual-e5-large`) |
| **6. Almacenamiento** | Inserción de los vectores + texto + metadatos en ChromaDB. Se genera un ID único por chunk. | `Chroma.add_documents()` |
| **7. Persistencia** | ChromaDB escribe el índice en disco para reutilización sin re-indexar. | ChromaDB (automático) |

```python
# 1-2. Cargar y unificar
docs = docs_excel + docs_pdf + docs_json

# 3. Fragmentar
all_splits = text_splitter.split_documents(docs)

# 4. Limpiar metadatos complejos
all_splits = filter_complex_metadata(all_splits)

# 5-6-7. Vectorizar y almacenar (ChromaDB llama al modelo de embeddings internamente)
document_ids = vector_store.add_documents(documents=all_splits)
```

### Flujo de datos por fuente

| Fuente | Loader | Chunks generados | Contenido clave para RAG |
|---|---|---|---|
| `pedidos_ecomarket.xlsx` | `DataFrameLoader` | 1 chunk por fila de pedido | Estado, tracking, fecha estimada, cliente |
| `POLÍTICA DE DEVOLUCIONES.pdf` | `PyPDFLoader` | N chunks por página/sección | Condiciones de elegibilidad, plazos, exclusiones |
| `FAQ.json` | `JSONLoader` + jq | 1 chunk por entrada FAQ | Preguntas frecuentes con categoría y respuesta |
| *(todos unidos)* | `RecursiveCharacterTextSplitter` | Varios cientos de sub-documentos | Colección vectorial unificada en ChromaDB |

---

## Síntesis de decisiones de arquitectura

| Componente | Decisión | Razón principal |
|---|---|---|
| **Modelo de embeddings** | `intfloat/multilingual-e5-large` | Open-source, multilingüe, 1024 dims, estado del arte en español |
| **Vector store** | ChromaDB | Persistencia nativa, filtros por metadatos, integración LangChain, costo cero |
| **Fuentes de datos** | Excel + PDF + JSON (3 tipos) | Cobertura completa de los flujos: pedidos + política + FAQ |
| **Estrategia de chunking** | `RecursiveCharacterTextSplitter` | Se adapta a documentos heterogéneos respetando estructura natural |
| **Parámetros de chunking** | `size=1000`, `overlap=200` | Equilibrio entre contexto completo y precisión de recuperación |
| **Limpieza de metadatos** | `filter_complex_metadata` | Evita errores de serialización en ChromaDB con datos tabulares |