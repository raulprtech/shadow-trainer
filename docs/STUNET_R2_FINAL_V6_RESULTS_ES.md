# STU-Net r2: evaluación volumétrica completa y límites de inferencia

Fecha: 21 de septiembre de 2026. Estado técnico: **completa y auditada**.
Estado científico: **réplica exploratoria en la misma cohorte final usada en r1**;
no es una evaluación independiente nueva ni validación clínica.

## Protocolo y procedencia

Se evaluaron B0 (Stage20 congelado), A (pérdida anterior) y B (pérdida
jerárquica) en el mismo orden de seis volúmenes KiTS23. Los checkpoints
se seleccionaron previamente en desarrollo; esta evaluación no los modifica
ni elige un ganador. El plan congelado es
`research/jobs/stunet-r2-final-frozen-plan-v6.json` (SHA-256
`87f16237f0b3eec4297d034fca0b49cf61670f438e641a13e73247764692acf7`).
Se usaron ventanas 128³, stride 64, promedio de probabilidades, `sync`
y procesos GPU independientes por brazo. El protocolo v6 declara antes de
la evaluación que HD95 se omite si el volumen excede 20 millones de vóxeles;
los seis casos superan ese umbral. Dice, precisión y sensibilidad se calcularon.

La sesión `research/workspace/stunet-r2-final-v6-r1` terminó en
1,243.2 segundos (20.7 minutos). Los tres resúmenes tienen 6/6 casos
en el orden congelado. Una auditoría posterior releyó los doce NIfTI,
los tres checkpoints y el código congelado; confirmó los hashes de los
resúmenes y **18/18 predicciones con recibos válidos**. El recibo sanitizado,
con alias F1–F6, está en
`research/evidence/stunet-r2-final-v6-r1/paired-audit.json`.
Los identificadores originales y rutas de datos permanecen sólo en los
artefactos internos de la sesión.

## Resultado de segmentación

Dice tumoral individual; F1–F6 conservan el orden congelado, sin selección
por resultado:

| Caso | B0 | A | B |
|---|---:|---:|---:|
| F1 | 0.000104 | 0.055592 | 0.031965 |
| F2 | 0 | 0.002117 | 0.000318 |
| F3 | 0 | 0 | 0 |
| F4 | 0.009211 | 0.028389 | 0.027161 |
| F5 | 0.002834 | 0.121110 | 0.162091 |
| F6 | 0 | 0.003184 | 0.001906 |
| **Media** | **0.002025** | **0.035065** | **0.037240** |
| **Mediana** | **0.000052** | **0.015787** | **0.014533** |

La comparación principal predefinida, **B−A**, tiene diferencia media
+0.002175, mediana −0.001253: B mejora en 1/6 pacientes, empeora en 4/6 y
empata en 1/6. Su pequeña ventaja media depende principalmente de F5.
**No hay evidencia consistente de que la pérdida jerárquica B supere a A.**

Frente a B0, la diferencia media tumoral es +0.033041 para A y +0.035215
para B; ambos mejoran en 5/6 casos y empatan en uno. Esta señal descriptiva
cumple numéricamente el umbral orientativo de +0.03 para B frente a B0,
pero los Dice absolutos siguen siendo muy bajos: un caso tiene Dice cero
en los tres brazos y varios apenas superan cero. No es una calidad
de segmentación utilizable clínicamente.

El Dice renal de clase 1 medio fue 0.481606 (B0), 0.571480 (A) y
0.571495 (B). El Dice del conjunto riñón+tumor+quiste fue 0.418674,
0.516563 y 0.521392, respectivamente. Estas medias esconden retrocesos
individuales: para riñón clase 1, A y B empeoran en 3/6 frente a B0;
para el conjunto renal, B empeora en 2/6. Cuatro pacientes tienen quiste
en la referencia; los tres brazos obtuvieron Dice de quiste cero.
No hubo tumores de referencia vacíos. Precisión, sensibilidad, conteos y
denominadores por alias constan en el recibo auditado.

HD95 figura como `skipped_memory_budget` en los 18 resultados; no se
imputa. Esta omisión limita cualquier análisis de calidad de bordes.

## Resultado de sistemas

La RTX 3050 Ti física de 4 GiB completó las 18 inferencias volumétricas.
El pico PyTorch reservado fue 808 MiB en cada brazo. Por telemetría de
proceso: swap máximo 168 MiB (<192 MiB), RSS máximo 3.861 GiB
(<4.5 GiB), RAM disponible mínima 3.074 GiB (>1.5 GiB) y espacio físico
libre mínimo 28.136 GiB (>20 GiB). Al cierre del supervisor, los artefactos
de la sesión ocupaban 14,121,954 bytes; la caché NIfTI decodificada terminó
vacía. No se
observó OOM ni violación de presupuesto en esta tanda.

Este resultado demuestra factibilidad de **evaluación volumétrica**
reproducible bajo las guardas locales. No demuestra por sí solo entrenamiento
STU-Net completo, aceleración universal ni equivalencia de `prefetch`.
`sync` permanece predeterminado y el par largo Stage38 sigue pendiente.
Los intentos incompletos v2–v5 se conservan como diagnóstico, no como
baselines de tiempo comparables.

## Uso responsable y siguiente gate

Para sinodales: presentar juntos el éxito de ejecución en 4 GiB, los
valores individuales, los cuatro quistes fallidos y la limitación de
reutilización de cohorte. La tesis puede usarlo como piloto de sistemas y
segmentación, no como validación de una herramienta clínica.

Para Circuito14: la afirmación defendible es que Shadow Trainer ejecutó y
auditó esta carga 3D real dentro de límites de memoria/disco en la PC.
La comparación A/B **no** justifica afirmar superioridad algorítmica de B.

Para un artículo de aprendizaje se necesitan un conjunto realmente nuevo
y bloqueado, réplicas/semillas, baselines más fuertes y mejoras absolutas
de segmentación. Las próximas decisiones de entrenamiento deben tomarse
con desarrollo; no usar esta cohorte final reutilizada para retocar
hiperparámetros o seleccionar checkpoints. Una línea de sistemas requerirá
repeticiones y comparaciones pareadas del mismo protocolo, además de cerrar
el gate largo `sync`/`prefetch` antes de atribuir beneficios de `auto`.
