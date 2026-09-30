# Demo completa y recuperación CPU — aceptación acotada

Autorización del usuario: continuar con el siguiente ensayo de demo. Se usaron exclusivamente fixtures Tiny3D sintéticos locales y CPU; no se abrió ninguna campaña clínica ni se utilizó CUDA.

## Resultado

- Demo por CLI: 12/12 pasos, sync, reporte HTML y ocho artefactos esperados presentes. Tiempo externo del comando: 7.707 s en este ensayo; no es un benchmark comparativo ni una cifra GPU.
- Recuperación: segunda ejecución interrumpida deliberadamente al paso 2, reanudada hasta 12. Secuencia exacta de identificadores 1…12, sin repetidos.
- Estado de workload final idéntico mediante digest estructurado de tensores y valores: modelo, optimizador, RNG Python y PyTorch. Digest d9cec2c6c337335dfd85330ffd76fe25bc9cddc43739bb51ef8cfec11fa0da49. No se compararon bytes pickle ni metadatos de tiempo como equivalencia científica.
- En ambas ejecuciones, pérdidas y normas de gradiente finitas; cada evento staging registra ocupación ≤4096 B y presupuesto 4096 B. Picos GPU registrados: cero.
- Sesiones: 49,547 B continua y 50,042 B reanudada. Checkpoints originales de experimentos previos intactos.
- Límites: piso físico C:20 GiB, RAM mínima 1.5 GiB, swap máximo 256 MiB, reserva de artefactos 128 MiB. Preflight antes de cada ensayo, lock compartido no bloqueante, un hilo BLAS/OpenMP y límite externo 180 s. Swap al cierre: cero. No se afirma telemetría externa continua a partir de esas muestras.

## Artefactos

- research/workspace/demo-cpu-acceptance-r1/run/report.html
- research/workspace/demo-cpu-recovery-r1/run/report.html
- docs/coordination/demo-cpu-acceptance-2026-09-20.json: resultados de checks, tamaños y SHA-256 de artefactos de ambas ejecuciones.

Los reportes de ejecución son para revisión local y pueden contener rutas internas. Revisar/exportar una versión sanitizada antes de compartirlos externamente. El paquete público del puente de ronda3 sigue separado y describe otra prueba.

## Reproducción de la demo continua

Desde /home/raulprtech/shadow-trainer y sólo después del preflight, con una ruta que no exista:

    flock -n /tmp/lab-round3-tests.lock timeout 180s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /home/raulprtech/clinical_core/.venv/bin/python -m shadow_trainer.cli demo --cpu --output-dir /tmp/shadow-demo-cpu-nueva

Este comando sí ejecuta entrenamiento sintético breve; no confundir con --prepare-only. La recuperación se comprobó mediante la API run_job(config, stop_after_steps=2), seguida de resume_job(config.output_dir), con una configuración generada por prepare_demo(..., use_cuda=False). Se comparó el campo workload de los checkpoints locales recién creados mediante state_digest.

## Significado para Circuito14 y tesis

El recorrido corto local y la recuperación Tiny3D funcionan con los límites seguros nuevos. Sirve para mostrar mecánica del producto, no para afirmar que CPU sustituye la validación GPU o que mejoró la segmentación. sync sigue estable; prefetch experimental y gate largo STU-Net pendientes.

Siguiente trabajo sin campaña: preparar presentación/video y revisar paquete de postulación. Para ciencia sigue pendiente reconciliar configuración/procedencia de r1 y completar evaluación r2 bajo autorización propia. No se necesita esperar el artículo para preparar Circuito14.
