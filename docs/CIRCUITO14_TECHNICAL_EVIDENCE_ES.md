# Anexo técnico de evidencia — Shadow Trainer

## Afirmación central

Shadow Trainer es un runtime local que admite o rechaza trabajos de entrenamiento
antes de reservar la GPU, mueve casos completos mediante una caché acotada y
produce checkpoints y evidencia auditables. El MVP ha ejecutado CNN 3D en una
RTX 3050 Ti física de 4 GiB. Esto demuestra viabilidad de ingeniería; no prueba
todavía aceleración general, equivalencia universal ni utilidad clínica.

## Evidencia histórica válida del MVP compacto

Este bloque conserva el corte original; no es una nueva corrida ni el conteo
vigente de tests. Las afirmaciones externas deben cotejarse con el
[ledger canónico](../research/CLAIMS_LEDGER.md).

| Evidencia | Resultado | Uso autorizado |
| --- | --- | --- |
| P1 — Tiny3D físico | 12/12 pasos, 2.316 s, 22 MiB de VRAM reservada, caché 4096/4096 bytes. | Demo física y completitud de artefactos. |
| P2 — NIfTI 3D | 2/2 casos, 4.816 s, pérdidas finitas, 64 MiB de VRAM reservada. | Viabilidad de entrenamiento por parches con datos 3D reales. |
| P3 — ResNet18 2.5D | 4 pasos limitados, pérdidas finitas, 258 MiB de VRAM reservada. | Viabilidad del adaptador y de una segunda familia de carga. |
| P4 — ResNet50 2.5D | 2 pasos limitados, pérdidas finitas, 488 MiB de VRAM reservada. | Viabilidad del adaptador; no comparación de rendimiento. |
| Pruebas automatizadas | 30 pruebas después de incorporar el recolector científico. | Contratos de configuración, staging, admisión, reanudación y reporte. |

Los valores de P1 y P2 se regeneran desde artefactos JSON/JSONL y quedan
asociados a un SHA-256. KiTS23 se usa como carga técnica pública, no como
producto, cohorte clínica ni prueba de desempeño diagnóstico.

## Nueva evidencia volumétrica física (21 de septiembre)

Una evaluación STU-Net completa ejecutó B0/A/B en seis volúmenes reales cada
uno: 18/18 predicciones con hashes y recibos verificados, en 20.7 minutos.
En la RTX 3050 Ti de 4 GiB, el pico PyTorch reservado fue 808 MiB; swap
máximo 168 MiB frente al límite 192 MiB y disco libre mínimo 28.1 GiB frente
al piso 20 GiB. La caché decodificada se liberó entre casos. Véase el
[informe científico y de recursos](STUNET_R2_FINAL_V6_RESULTS_ES.md) y su
[recibo auditado](../research/evidence/stunet-r2-final-v6-r1/paired-audit.json).

Esta prueba sustenta **inferencia/evaluación volumétrica acotada y auditable**,
no un nuevo entrenamiento de 18 casos, aceleración de prefetch o calidad
clínica. La cohorte ya se había usado en r1; el Dice tumoral absoluto sigue
bajo y B no supera consistentemente a A. Mantener esas limitaciones visibles
en cualquier material de Circuito14.

## Entrenamiento físico de mayor escala (22 de septiembre)

Dos sesiones STU-Net-S con Stage20 inicial y staging `sync` completaron
1,536 actualizaciones cada una: 64 casos × 3 épocas y 96 casos × 2 épocas.
El brazo de 64 confirmó 192 fronteras de caso y pérdidas finitas; su sesión
ocupó 2.5 GiB, con mínimo 25.847 GiB libres en C:, RSS máximo del árbol
2.863 GiB y swap máximo 123.41 MiB. El comparador de 96 también terminó
1,536/1,536 pasos con caché de hasta 2 GiB. El contraste científico es
pos hoc y no aísla sólo el tamaño de datos. Véase el
[informe comparativo](STUNET_FIXED_BUDGET_COMPARISON_ES.md).

Esta evidencia fortalece la demostración de **ejecución, staging, guardas,
checkpointing y auditoría** de una carga 3D real bajo 4 GiB de VRAM y disco
local acotado. No autoriza una afirmación comercial de mejor Dice, de
aceleración ni de validación clínica. Una evaluación adicional de agrupaciones
KiTS en desarrollo se detuvo limpiamente al cruzar la guarda de 192 MiB de
swap; tras recuperar margen, reanudó desde el caso sellado y completó 4/4
pacientes por brazo con recibos auditados. El paro inicial y el reintento
exitoso se conservan por separado; sus tiempos no se comparan.

## Resultado negativo conservado

Una primera ejecución NIfTI falló porque el camino CUDA de la pérdida elegida
no ofrecía backward determinista. El fallo, su resumen y reporte se conservaron.
La corrección sustituyó la operación por `log_softmax + gather`, manteniendo la
semántica de entropía cruzada y permitiendo completar una ejecución nueva. La
ejecución fallida no aparece en figuras numéricas.

## Gate de prefetch

Stage38 r2 completó 108/144 pasos síncronos y una reanudación durable, pero fue
detenido por la guarda de disco al cruzar el piso de 20 GiB. No hubo brazo
prefetch, por lo que no existe comparación pareada. El producto mantiene
`sync` como opción estable y marca `prefetch` como experimental.

La matriz posterior `pair-matrix-physical-r1` sí obtuvo 24/24 pares exactos en
fixtures pequeños y condiciones frías/calientes. No sustituye Stage38 largo.
Su resultado temporal favorable en NIfTI3D frío sólo puede citarse junto a la
tabla completa de ocho celdas y sus límites; no demuestra aceleración universal.

## Diferenciación defendible

- coordina GPU, RAM, swap, disco, caché y origen remoto bajo un contrato único;
- rechaza trabajos inseguros antes de transferir datos o ocupar la GPU;
- valida tamaño y SHA-256 antes de promover descargas atómicas;
- conserva modelo, optimizador, RNG y posición del dataset;
- genera un expediente autocontenido de cada ejecución;
- impide convertir resultados inválidos o diagnósticos en cifras comerciales.

La diferenciación no consiste en afirmar que cada componente es nuevo. La
hipótesis de producto es que integrar estas garantías en un runtime pequeño,
local y verificable reduce el costo de operar entrenamiento AI/edge en equipos
que no pueden mantener simultáneamente dataset y estado de entrenamiento.

## Estado y próximos gates

1. Antes de una campaña autorizada, revalidar recursos físicos y piso de 20 GiB en C:; completar un par largo STU-Net exacto sigue pendiente.
2. ResNet18/50 2.5D ya figuran como adaptadores proxy y en la matriz física; extender evidencia, no presentarlos como implementación pendiente.
3. Conservar la matriz fría/caliente ya obtenida y preparar un baseline PyTorch/MONAI comparable, aún pendiente.
4. Completar cinco entrevistas industriales y registrar evidencia sin inventar tracción.
5. Congelar cifras, pitch y video después de la revisión de IP.

El paquete actual permite demostrar el producto de forma honesta aun si el
prefetch no supera el gate: en ese caso, la seguridad síncrona sigue siendo el
resultado estable y la divergencia se presenta como hallazgo experimental.
