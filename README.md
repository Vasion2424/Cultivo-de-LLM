# Cultivo-de-LLM

# sovereign_growing_llm

> Un Transformer causal a nivel de byte que **crece, poda, conserva y repara** su propia arquitectura durante el entrenamiento. Es un proyecto de investigación: demuestra que un mecanismo de *ciclo de vida estructural* funciona.

---

### Qué es

Un **Transformer causal a nivel de byte** (vocabulario = 256 bytes UTF-8) cuyo bloque feed-forward (`SovereignFFN`) tiene **capacidad dinámica**. En lugar de una arquitectura fija, el modelo administra sus propias neuronas y conexiones con cuatro operaciones:

| Operación | Qué hace |
|---|---|
| **CREATE** | Crea neuronas y rutas nuevas cuando hay presión de capacidad |
| **CONSERVE** | Refuerza (mantiene) lo que se usa de forma estable |
| **MUTATE** | Bonifica el uso persistente de rutas |
| **PRUNE** | Poda probabilísticamente lo inactivo |

El sistema detecta **estancamiento** (meseta de validación) y responde **creciendo** su arquitectura: anchura, profundidad, atención o contexto. Después valida si el crecimiento ayudó o no, y **conserva siempre el mejor checkpoint**.

---

### Qué NO es

- **No es un razonador.** Pérdida final ~2 bits/byte; el texto generado es incoherente.
- **No es un LLM útil.** Escala minúscula (d_model 32→64, contexto 256→576).
- **Sí es** una prueba de que el *mecanismo* de crecimiento/poda/reparación **funciona y está documentado con métricas**.

Está a años-luz del DeepSeek/Qwen actuales, pero es exactamente el paso correcto **antes** de escalar a un equipo con más recursos.

---

## Arquitectura


MiniLM (Transformer causal, nivel de byte)
├── token: Embedding(256, d_model)
├── pos:   Embedding(context, d_model)
├── blocks: N × Block
│   ├── ln1 + MultiheadAttention(d_model, heads)
│   └── ln2 + SovereignFFN(d_model, capacity)   ← aquí vive el ciclo de vida
└── ln_final + head: Linear(d_model, 256)


Configuración inicial (`sovereign_growing.py`):

| Parámetro | Valor |
|---|---|
| d_model | 32 |
| heads | 2 |
| context | 256 |
| vocab_size | 256 (bytes UTF-8) |
| learning_rate | 0.0002 |
| batch_size | 2 |
| prune_probability | 0.25 |
| active_threshold | 0.85 |
| used_threshold | 0.70 |

---

## 📈 Crecimiento documentado (métricas reales)

### Etapa `etapaCFULL` (dataset `data2.txt`)

| Dimensión | Inicio | Final |
|---|---|---|
| Capacidad FFN | 32 | **64** |
| Profundidad (bloques) | 1 | **3** |
| d_model | 32 | **64** |
| Cabezas de atención | 2 | **4** |
| Contexto | 256 | **384** |

Pérdida de validación (holdout 5%): **5.696 → 2.492** (mejor: 2.492, macro_score 0.4375).

---

**Secuencia de eventos de crecimiento:**

| Paso | Tipo | Cambio |
|---|---|---|
| 4200 | width | capacidad 32 → 48 |
| 6200 | depth | profundidad 1 → 2 |
| 7800 | attention | d_model 32 → 48, heads 2 → 3 |
| 9400 | context | contexto 256 → 384 |
| 11000 | width | capacidad 48 → 64 |
| 12600 | depth | profundidad 2 → 3 |

### Etapa `etapa4_V3` (dataset `data3.txt` + replay de `data2.txt`)

| Dimensión | Final |
|---|---|
| Capacidad FFN | 64 |
| Profundidad | 3 |
| d_model | 64 |
| Cabezas | 4 |
| Contexto | **576** |

Pérdida (data3 holdout): **1.895 → 1.811** (mejor macro_score 0.9697, logrado en paso 21600).

---

## 🔧 Reparación ante daños

El proyecto incluye pruebas de **daño y reparación** (`mini_sovereign_lm_damage.py`, `resultadosdaños.json`): la red se somete a daño estructural y se verifica su capacidad de reconstruirse desde el estado conservado.

**Próximo paso natural:** escalar el mismo mecanismo a un equipo con más CPU/RAM/GPU y vocabulario mayor (subword en lugar de byte), y validar si el crecimiento mantiene la mejora o reproduce la regresión observada.

---

### Reproducibilidad 

sovereign_growing.py --dataset datos\(aqui datos para entrenar) --steps (numero de ciclos) --autosave-interval (numero de ciclos para el guardado) --out checkpoints\(nombre del modelo final).pt   

---

### Ejemplos prácticos con replay  

sovereign_growing.py --dataset datos\data2.txt --steps 100000 --autosave-interval 20000 --out checkpoints\etapaCFULL.pt  
sovereign_growing_autoscale.py --dataset datos\data3.txt --replay-dataset datos\data2.txt --checkpoint checkpoints\etapa2.pt --steps 1000 --autosave-interval 100 --eval-dataset datos\data1_val.txt --eval-dataset datos\data2_val.txt --eval-dataset datos\data3_val.txt --out checkpoints\etapa3.pt
sovereign_growing.py --chat --checkpoint checkpoints\etapa2.pt


