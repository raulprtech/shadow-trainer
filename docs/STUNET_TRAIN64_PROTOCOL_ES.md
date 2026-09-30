# STU-Net: entrenamiento de desarrollo con 64 casos KiTS23

## Estado (22-09-2026)

La sesión `research/workspace/stunet-train64-a-v1` entrenó el brazo A con la
pérdida histórica de Stage20. El supervisor `research/run_stunet_train64.py`
congela dos épocas, ocho parches por caso y 1,024 actualizaciones previstas.
Toma **los 64 casos del CSV local de entrenamiento**, no todos los casos del
dataset KiTS23. Los cuatro casos Stage34 de desarrollo sirven sólo para
selección/evaluación interna. La cohorte reutilizada de seis casos no participa
en esta tanda. El plan guarda hashes del manifest, CSV, schedule y checkpoint
base, además de la lista congelada de casos.

Staging `sync`, caché de entrenamiento de 2 GiB y guardas de 20 GiB libres en
C:, RAM, swap y RSS. El supervisor usa un lock para impedir otro trabajo GPU,
checkpoints en fronteras de caso y recibo de cierre. El recibo sólo cuenta
actualizaciones confirmadas en eventos `case_boundary`; pasos parciales no se
reclaman como recuperables.

```bash
cd /home/raulprtech/shadow-trainer
PYTHONPATH=src .venv/bin/python research/run_stunet_train64.py --preflight-only
PYTHONPATH=src .venv/bin/python research/run_stunet_train64.py --max-hours 6
```

La guarda de swap detuvo el proceso original durante la validación posterior
a la segunda época; el recibo original permanece `guard_stopped`. El
checkpoint se escribió antes del evento durable de frontera de caso 128 y
se abrió mediante el cargador restringido de PyTorch: `global_step=1024`,
`next_epoch=2`, `next_case_index=0`. El auditor independiente
`research/audit_stunet_train64_recovery.py` verificó 1,024 pérdidas finitas,
128 fronteras secuenciales, identidad del plan y del checkpoint, y los
recibos de predicción. Su resultado está en `recovery-audit.json`; no reescribe
ni oculta el incidente de swap.

La evaluación de desarrollo B0/A recuperada usó un proceso independiente por
brazo, la misma cohorte de cuatro casos y un evaluador de probabilidades por
slabs. HD95 se omitió explícitamente para reducir memoria; el criterio de
selección usa Dice. Resultado:

| Métrica media en desarrollo | B0 | A (1,024 pasos) | Delta |
|---|---:|---:|---:|
| Dice renal | 0.655816 | 0.669491 | +0.013675 |
| Dice tumoral | 0.025272 | 0.057741 | +0.032469 |

La media tumoral está dominada por un caso: uno mejora, uno empeora
ligeramente y dos permanecen en cero. A satisface la regla agregada de
desarrollo, pero no hay mejora consistente por paciente ni validación
independiente. La selección no autoriza una afirmación clínica.

## Lectura de resultados y siguiente gate

La cohorte adicional de seis casos del CSV de validación quedó congelada
mediante `research/freeze_stunet_train64_heldout.py` antes de abrir etiquetas o
predicciones. No aparece en schedules ni cohortes locales previas; una búsqueda
textual adicional sólo la encontró en inventarios de origen. Esto no demuestra
ausencia absoluta de uso externo. Su staging con rclone completó 6/6 pares
imagen-etiqueta bajo 1.5 GiB de caché. B0 completó 6/6 predicciones con recibos
de procedencia. A quedó inicialmente en 1/6 después de que el swap global
alcanzó 225 MiB frente al límite congelado de 192 MiB. Tras un reinicio de
WSL, el preflight pasó y A terminó los seis casos; verificó y reutilizó el
primero ya sellado. El auditor cerró los seis pares. Resultados y límites:
[`STUNET_TRAIN64_HELDOUT_RESULTS_ES.md`](STUNET_TRAIN64_HELDOUT_RESULTS_ES.md).
Los seis casos finales antiguos siguen clasificados como cohorte reutilizada
y no se mezclan con ésta.

`research/complete_stunet_train64_heldout.py` verifica los hashes congelados,
la cohorte y los recursos antes de reanudar A. Reutiliza la primera predicción
solo tras validar su recibo; al terminar ambos brazos, audita los seis pares,
sus hashes y sus diferencias. Se niega a publicar la auditoría si
un brazo sigue parcial o si cambió el protocolo. La interrupción original y
las muestras de recursos permanecen en la sesión.

```bash
cd /home/raulprtech/shadow-trainer
PYTHONPATH=research .venv/bin/python research/complete_stunet_train64_heldout.py --preflight-only
# Ejecutar solo cuando el preflight indique pass:
PYTHONPATH=research .venv/bin/python research/complete_stunet_train64_heldout.py
```

El primer preflight del 22-09-2026 devolvió `swap_limit`: aproximadamente
221 MiB permanecían ocupados aun sin evaluador activo. No se subió el límite
ni se reinició WSL automáticamente. Tras el reinicio hecho por el usuario, el
reintento terminó sin volver a descargar los seis casos; el incidente original
permanece registrado.

Después de esa comparación:

1. Los seis recibos por brazo, diferencias pareadas y presupuesto quedaron
   auditados. La señal de segmentación es positiva, pero piloto n=6.
2. Construir un extractor de embeddings
   KiTS de cuatro clases en Clinical-Core. El extractor visual actual usa otro
   checkpoint de STU-Net/TotalSegmentator de 105 clases y no acepta directamente
   este checkpoint. Congelar fuente, geometría y representación antes de una
   comparación pareada de pronóstico con ResNet.

Este experimento no demuestra rendimiento clínico, superioridad frente a
ResNet ni equivalencia del prefetch largo. `sync` sigue siendo el predeterminado.
