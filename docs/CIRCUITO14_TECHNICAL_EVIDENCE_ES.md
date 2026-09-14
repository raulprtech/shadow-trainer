# Anexo técnico de evidencia — Shadow Trainer

## Afirmación central

Shadow Trainer es un runtime local que admite o rechaza trabajos de entrenamiento
antes de reservar la GPU, mueve casos completos mediante una caché acotada y
produce checkpoints y evidencia auditables. El MVP ha ejecutado CNN 3D en una
RTX 3050 Ti física de 4 GiB. Esto demuestra viabilidad de ingeniería; no prueba
todavía aceleración general, equivalencia universal ni utilidad clínica.

## Evidencia válida

| Evidencia | Resultado | Uso autorizado |
| --- | --- | --- |
| P1 — Tiny3D físico | 12/12 pasos, 2.316 s, 22 MiB de VRAM reservada, caché 4096/4096 bytes. | Demo física y completitud de artefactos. |
| P2 — NIfTI 3D | 2/2 casos, 4.816 s, pérdidas finitas, 64 MiB de VRAM reservada. | Viabilidad de entrenamiento por parches con datos 3D reales. |
| P3 — ResNet18 2.5D | 4 pasos limitados, pérdidas finitas, 258 MiB de VRAM reservada. | Viabilidad del adaptador y de una segunda familia de carga. |
| P4 — ResNet50 2.5D | 2 pasos limitados, pérdidas finitas, 488 MiB de VRAM reservada. | Viabilidad del adaptador; no comparación de rendimiento. |
| Pruebas automatizadas | 29 pruebas después de incorporar el recolector científico. | Contratos de configuración, staging, admisión, reanudación y reporte. |

Los valores de P1 y P2 se regeneran desde artefactos JSON/JSONL y quedan
asociados a un SHA-256. KiTS23 se usa como carga técnica pública, no como
producto, cohorte clínica ni prueba de desempeño diagnóstico.

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

1. Recuperar al menos 30 GiB físicos en C: y completar Stage38 r3.
2. Añadir ResNet18/50 y una carga 2.5D mediante la API pública.
3. Ejecutar comparaciones frías/calientes y un baseline PyTorch/MONAI.
4. Completar cinco entrevistas industriales y registrar evidencia sin inventar tracción.
5. Congelar cifras, pitch y video después de la revisión de IP.

El paquete actual permite demostrar el producto de forma honesta aun si el
prefetch no supera el gate: en ese caso, la seguridad síncrona sigue siendo el
resultado estable y la divergencia se presenta como hallazgo experimental.
