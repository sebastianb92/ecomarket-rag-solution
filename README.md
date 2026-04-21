# EcoMarket AI Support — Taller Práctico #1

Sistema de atención al cliente basado en **IA Generativa** para EcoMarket, una empresa de e-commerce sostenible.

## Descripción

Este repositorio implementa la solución propuesta en el Taller Práctico #1, demostrando cómo la ingeniería de prompts puede transformar la atención al cliente automatizando el 80% de consultas repetitivas mediante un modelo LLM open-source con contexto RAG simulado.

**Modelo utilizado:** `llama-3.3-70b-versatile` via [Groq API](https://groq.com)

---

## Estructura del Repositorio

```
ecomarket-solution/
├── README.md
├── requirements.txt
├── docs/
│   ├── fase1_selección_de_modelo.md
│   └── fase2_fortalezas_limitaciones_riesgos.md
│
├── prompts/
│   ├── order_status_prompt.txt
│   ├── return_policy_prompt.txt
│   └── system_prompt.txt
│
├── data/
│   ├── orders_database.txt
│   └── policies.txt
│
├── notebooks/
│   └── EcoMarket_AI_Solution.ipynb

```



---
## Uso

La solución se ejecuta principalmente a través de un notebook interactivo que integra los prompts y las fuentes de datos de EcoMarket.

---
##  Uso

###  Ejecutar en Google Colab

Puedes ejecutar la solución directamente en Google Colab sin necesidad de instalar dependencias en tu entorno local:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebastianb92/ecomarket-solution/blob/main/notebooks/EcoMarket_AI_Solution.ipynb)


---

### Pasos de ejecución

Una vez abierto el notebook:

1. Ejecuta las celdas en orden.
2. El sistema realizará automáticamente:
   - Carga de prompts desde `/prompts`
   - Carga de datos desde `/data`
   - Inicialización del modelo LLM
   - Ejecución de los casos de uso

---

###  Casos de uso incluidos

El notebook simula los principales flujos del sistema de atención al cliente.

####  Consulta de estado de pedido

Escenarios disponibles:
- Pedido válido
- Pedido retrasado
- Pedido no encontrado

**Archivos utilizados:**
- `prompts/order_status_prompt.txt`
- `data/orders_database.txt`

---

####  Solicitud de devolución

Escenarios disponibles:
- Devolución válida
- Producto no elegible (políticas de higiene)
- Devolución fuera de plazo

**Archivos utilizados:**
- `prompts/return_policy_prompt.txt`
- `data/policies.txt`

---

###  Arquitectura en ejecución

El notebook integra los siguientes componentes del proyecto:

- **Prompts (`/prompts`)**  
  Definen la lógica de interacción y comportamiento del modelo.

- **Datos (`/data`)**  
  Simulan las fuentes reales del negocio (pedidos y políticas).

- **Notebook (`/notebooks`)**  
  Orquesta el flujo completo tipo RAG (carga → contexto → generación → respuesta).

---

###  Notas

- No se requieren scripts adicionales: toda la lógica está contenida en el notebook.
- Al abrir desde Colab, se ejecuta una copia independiente del notebook (no modifica el repositorio original).
- La solución puede escalar fácilmente a producción reemplazando archivos locales por APIs o bases de datos reales.
---

## Casos de prueba incluidos

| # | Escenario | Tipo | Resultado esperado |
|---|-----------|------|--------------------|
| 1 | Pedido ECO-12345 en tránsito | Estado | Info de entrega + tracking |
| 2 | Pedido ECO-12346 retrasado | Estado | Disculpa + nueva fecha |
| 3 | Pedido ECO-12347 entregado | Estado | Confirmación de entrega |
| 4 | Pedido ECO-12349 cancelado | Estado | Info del reembolso |
| 5 | Pedido ECO-12351 pendiente | Estado | Pendiente de pago |
| 6 | Número de pedido inexistente | Estado | Redirección a soporte |
| 7 | Devolución de producto de hogar (OK) | Devolución | Instrucciones del proceso |
| 8 | Devolución de jabón abierto (NO) | Devolución | Negativa empática + alternativas |
| 9 | Devolución fuera de plazo 30 días | Devolución | Negativa + escalado a humano |
| 10 | Devolución por daño en envío | Devolución | Aprobación + compensación |
| 11 | Producto perecedero (alimento) | Devolución | Negativa empática + política |

---

##  Arquitectura de la solución

```
Consulta del cliente
        │
        ▼
┌───────────────────┐
│  Clasificador     │ ──► simple (80%) ──► RAG + LLM ──► Respuesta automática
│  de intención     │
└───────────────────┘ ──► compleja (20%) ──► Asistente de agente humano
```


---

## Autores

* Johan Sebastian Bonilla

* Edwin Gómez

