# EcoMarket AI Support — Taller Práctico #2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebastianb92/ecomarket-rag-solution/blob/main/notebooks/EcoMarket_AI_RAG_Solution.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![LLM](https://img.shields.io/badge/LLM-LLaMA_3.3_70B-orange)
 
**Maestría en Inteligencia Artificial — IA Generativa | Taller Práctico #2**
 
**Integrantes:** Johan Sebastian Bonilla · Edwin Gómez

## Descripción

Sistema de atención al cliente con IA Generativa basado en **Retrieval-Augmented Generation (RAG)** para EcoMarket, un e-commerce de productos ecológicos.
 
El chatbot **EcoBot** resuelve dos tipos de consultas:
1. **Estado de pedidos** — tracking, retrasos, cancelaciones, pagos pendientes
2. **Devoluciones** — evaluación de elegibilidad según política, casos aprobados, rechazados y fuera de plazo
A diferencia de un LLM puro, EcoBot no genera respuestas desde su conocimiento preentrenado. Primero recupera fragmentos relevantes desde una base de conocimiento vectorial (pedidos reales + política de devoluciones + FAQ), y luego usa ese contexto para generar respuestas precisas, coherentes y alineadas con la información real del negocio.


---

##  Stack Tecnológico
 
| Componente | Tecnología |
|---|---|
| Framework RAG | LangChain (`langchain`, `langchain-groq`, `langchain-community`, `langchain-chroma`, `langchain-classic`) |
| LLM | `llama-3.3-70b-versatile` vía GROQ API — `temperature=0.3` |
| Embeddings | `intfloat/multilingual-e5-large` (HuggingFace, 1024 dims, normalized) |
| Vector Store | ChromaDB (persistido en `./chroma_langchain_db`) |
| Loaders | `DataFrameLoader` (Excel), `PyPDFLoader` (PDF), `JSONLoader` (FAQ) |
| Entorno | Google Colab / local con detección automática |
| Gestión de paquetes | `uv` |
 
---

## Estructura del Repositorio

```
ecomarket-rag-solution/
│
├── data/
│   ├── pedidos_ecomarket.xlsx        # Base de datos de pedidos
│   ├── POLÍTICA DE DEVOLUCIONES.pdf  # Política oficial de devoluciones
│   └── FAQ.json                      # Preguntas frecuentes estructuradas
│
├── docs/
│   └──
│
├── notebooks/
│   └── EcoMarket_AI_RAG_Solution.ipynb   # Notebook principal
│ 
├── requirements.txt                  # Dependencias con versiones fijadas
└── README.md

```



---
## Uso

Abre y ejecuta el notebook `notebooks/EcoMarket_AI_RAG_Solution.ipynb` en orden. Las secciones son:
 
1. **Configuración del entorno** — detección Colab/local, descarga de datos, instalación de dependencias
2. **Implementación del pipeline RAG** — inicialización del LLM, embeddings y vector store
3. **Indexación de datos** — carga, chunking e inserción en ChromaDB
4. **Recuperación y generación** — retriever, prompt template y cadena `RetrievalQA`
5. **Ejercicio 1** — Consultas de estado de pedido
6. **Ejercicio 2** — Solicitudes de devolución


---

###  Casos de uso incluidos

 
### Ejercicio 1 — Estado de pedido
 
| Número de pedido | Escenario |
|-----------------|-----------|
| `ECO-12345` | Pedido en tránsito |
| `ECO-12346` | Pedido retrasado |
| `ECO-12347` | Pedido entregado |
| `ECO-12349` | Pedido cancelado |
| `ECO-12371` | Pedido pendiente de pago |
| `ECO-99999` | Número inexistente (prueba anti-alucinación) |
 
### Ejercicio 2 — Devoluciones
 
| Producto | Motivo | Resultado esperado |
|----------|--------|--------------------|
| Botella de acero inoxidable 750ml | Diseño no gustó |  Aprobada |
| Jabón orgánico artesanal | Olor no gustó |  Rechazada |
| Mix de frutos secos orgánicos | Cambio de opinión |  Rechazada |
| Cepillo de dientes de bambú | Paquete llegó dañado |  Aprobada |
| Set de cubiertos de bambú | Le regalaron otro |  Fuera de plazo |



---

## Autores

* Johan Sebastian Bonilla

* Edwin Gómez

