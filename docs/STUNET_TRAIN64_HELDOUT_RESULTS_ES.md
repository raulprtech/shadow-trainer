# STU-Net-S, 64 casos: evaluación adicional KiTS23 B0/A

**Estado (22-09-2026):** comparación técnica completa sobre seis casos de
validación local no presentes en los schedules/cohortes previos inspeccionados.
No es una validación clínica ni una puntuación oficial KiTS23.

## Pregunta y protocolo congelado

¿El checkpoint A posterior a 1,024 actualizaciones con los 64 casos del split
local de entrenamiento mejora la segmentación de riñón y tumor frente al
checkpoint Stage20 B0? A fue elegido por la regla de cuatro casos de desarrollo
**antes** de abrir esta nueva cohorte. Los seis casos se eligieron por hash con
semilla `20260922` entre los del CSV de validación ausentes en las cohortes y
schedules locales auditados. El orden, los doce pares de archivos y sus tamaños
quedaron en `selection.json`; staging comprobó integridad de NIfTI. Una
búsqueda local no prueba ausencia de uso en notebooks borrados, otras máquinas
o ajustes no registrados.

Ambos brazos usaron el mismo evaluador físico STU-Net-S, ventanas 128³,
stride 64, agregación de probabilidades por slabs, mismas imágenes/máscaras
y protocolo de métricas. A se reanudó tras una parada previa por swap;
reutilizó **una** predicción ya sellada, verificada por hash y recibo, sin
repetir la inferencia. HD95 se omitió explícitamente (`surface_voxel_limit=0`)
para respetar memoria. El auditor posterior revalidó hashes de CSV, manifest,
checkpoint base, checkpoint candidato, recibo de recuperación, los 12 insumos,
las 12 predicciones y sus 12 recibos de procedencia.

## Resultados por caso

Dice de clase renal = etiqueta riñón sola; no incluye tumor/quiste. La
agrupación `riñón + masas` se informa aparte. Identificadores P1–P6 siguen
el orden congelado; los IDs de origen y predicciones están en la sesión local.

| Caso | Riñón B0 | Riñón A | Δ renal | Tumor B0 | Tumor A | Δ tumoral |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 0.2742 | 0.5081 | +0.2339 | 0.0012 | 0.2552 | +0.2540 |
| P2 | 0.0519 | 0.4409 | +0.3890 | 0.0000 | 0.0988 | +0.0988 |
| P3 | 0.0976 | 0.4902 | +0.3926 | 0.0030 | 0.2262 | +0.2231 |
| P4 | 0.4390 | 0.7704 | +0.3313 | 0.0000 | 0.0000 | 0.0000 |
| P5 | 0.2092 | 0.4977 | +0.2884 | 0.0000 | 0.0120 | +0.0120 |
| P6 | 0.7677 | 0.8553 | +0.0875 | 0.0000 | 0.5272 | +0.5272 |

| Métrica media de seis pares | B0 | A | Δ pareado | Casos mejoran |
|---|---:|---:|---:|---:|
| Dice riñón | 0.3066 | 0.5937 | +0.2871 | 6/6 |
| Dice tumor | 0.0007 | 0.1866 | +0.1859 | 5/6; uno sin cambio |
| Dice riñón + masas | 0.2739 | 0.6135 | +0.3396 | 6/6 |

El Dice de quiste fue cero en los tres pares donde estuvo definido; los otros
tres no permiten comparación pareada de esa clase. La media tumoral tiene
variación importante: al quitar un caso por vez, la media de los cinco deltas
restantes varía de +0.1176 a +0.2230. El rango análogo renal es +0.2660 a
+0.3270. Estos análisis son descriptivos con n=6, no intervalos de validez
externa ni pruebas confirmatorias.

## Ejecución física y guardas

Se generaron seis láminas PNG y un PDF de seis páginas en
`research/workspace/stunet-train64-heldout-r1/figures/`. Cada figura usa el
corte con mayor área tumoral **según la referencia**, y el mismo recorte
determinado por esa referencia para CT, etiqueta, B0 y A. El
`figures/manifest.json` registra índice de corte y coordenadas; los cortes no
se eligieron por la calidad de A. Son ejemplos visuales de un corte por
volumen, no sustituyen las métricas de volumen completo.

La evaluación corrió en la RTX 3050 Ti de 4,096 MiB. Cada brazo registró
730 MiB de memoria CUDA reservada máxima por PyTorch durante inferencia; no
equivale a uso total de VRAM del equipo. B0 terminó 6/6 en 702 s. La
invocación reanudada de A terminó 6/6 en 633 s, incluida la verificación del
primer caso sellado; **no** es una comparación de velocidad porque A reutilizó
esa predicción y las condiciones de caché no son pareadas.

En la invocación reanudada de A, 317 muestras de recursos registraron máximo
3 MiB de swap, mínimo 3,023 MiB de RAM disponible, RSS máximo 4,032 MiB y
mínimo 22.48 GiB libres en C:. La caché final de seis casos ocupa ~1.2 GiB,
por debajo del límite de 1.5 GiB; la sesión completa ocupa ~1.2 GiB, por
debajo de 5 GiB. El intento anterior detenido por swap (225 MiB) sigue en el
log y no se elimina ni se mezcla con la telemetría del reintento exitoso.

## Interpretación y gate siguiente

La señal de mejora de segmentación en esta pequeña cohorte local es
consistente para riñón y positiva en cinco de seis tumores, mucho más clara
que la señal de desarrollo. Aun así, A no segmentó un tumor y no segmentó
quistes; los Dice absolutos tumorales siguen modestos. No se ha demostrado
generalización clínica, superioridad frente a otros segmentadores, ni mejora
de pronóstico frente a ResNet: esa última es **otra tarea**.

Para el módulo visual de Clinical-Core, el gate downstream era construir un
adaptador explícito del checkpoint KiTS de **cuatro clases** a un artefacto
visual y compararlo en un protocolo pronóstico pareado. El extractor STU-Net
operativo de Clinical-Core procede de TotalSegmentator y tiene 105 clases;
no se sustituye sin validar contrato, geometría y representación. Para un
artículo de aprendizaje
harían falta más cohortes/semillas y baselines de segmentación. Para un
artículo de sistemas, esta tanda aporta evidencia física de staging, guardas,
reanudación y procedencia, no equivalencia de prefetch largo.

**Gate downstream ejecutado el 22-09-2026:** Clinical-Core construyó el
artefacto KiTS de 512D en 75 CT TCIA usando la ROI renal histórica y un
contrato separado. Su comparación pronóstica interna quedó por debajo del
STU-Net/TotalSegmentator de 105 clases y de ResNet18 2.5D; el checkpoint KiTS
no se promueve como encoder pronóstico. Véase
[`kits_stunet_transfer_75_results.md`](/home/raulprtech/clinical_core/docs/kits_stunet_transfer_75_results.md).
Esto no cambia la evidencia de segmentación KiTS ni la de sistemas de Shadow
Trainer; tampoco aísla el efecto del fine-tuning frente al Stage20 B0.

Evidencia primaria:
`research/workspace/stunet-train64-heldout-r1/selection.json`,
`evaluation/evaluation/{B0,A}/summary.json`, `paired-audit.json`,
`eval_A_resume.json`, `eval_{B0,A}.resources.jsonl` y
`research/workspace/stunet-train64-a-v1/recovery-audit.json`. Figuras:
`research/workspace/stunet-train64-heldout-r1/figures/manifest.json` y
`stunet-b0-a-six-cases.pdf` en la misma carpeta.
