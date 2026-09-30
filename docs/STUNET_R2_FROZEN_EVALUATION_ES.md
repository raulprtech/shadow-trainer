# r2: protocolo congelado de continuación de evaluación

Estado inicial: **preparado sin autorización automática**. La sección final
registra las ejecuciones posteriores expresamente autorizadas y sus fallos.
Este documento congela una continuación, no constituye un preregistro anterior
al entrenamiento o a las evaluaciones históricas.

## Identidad y límites científicos

Se conservan los seis casos finales y su orden del manifest de r2. No existe
solapamiento con los 24 casos de entrenamiento o cuatro de desarrollo registrados
en esa campaña. Son los mismos seis casos finales que r1: no representan otra
cohorte ni seis pacientes nuevos. No se consultaron sus métricas en esta revisión.

La auditoría textual reproducible encontró nueve referencias de evaluación en r1,
siete de campaña/evaluación parcial en r2 y cuatro de inventario. No encontró
otro uso en estos dos repositorios. La bandera histórica `independent_evaluation=true` no es prueba suficiente de
independencia: el selector sólo auditaba `runs/*/schedule.json` y determinados
campos. No cubre necesariamente todos los notebooks, selección manual o usos
previos. Antes de afirmar validación independiente debe reconstruirse el historial
de uso. Si no puede establecerse, se describirá como evaluación exploratoria de
una cohorte separada de los splits registrados, potencialmente reutilizada.

No se buscarán nuevos pacientes para mejorar cifras ni se cambiarán parámetros
después de consultar resultados. La cohorte no se reduce ante fallos y los seis
casos se conservan para todos los brazos.

## Congelación material

Plan interno: `research/jobs/stunet-r2-final-frozen-plan.json`.
SHA-256 al cierre:
`b03dc250b6d9c3d1a2cc45aa5973d79dc2a15052e0e0ac6c762f168d459c8d24`.

- B0: Stage20, hash cotejado con desarrollo.
- A/B: checkpoints `best` de r2 seleccionados históricamente por desarrollo,
  época 2. No reentrenar ni reseleccionar usando test.
- Se registran hashes actuales de los tres checkpoints, manifest, configuración,
  evaluador y auxiliares. No sustituyen captura de todas las dependencias del entorno.
- Los doce NIfTI previstos existen y coinciden en tamaño con los metadatos
  congelados. Total: 889,607,223 bytes. No se leyeron sus contenidos para este plan;
  verificar MD5, adquirir SHA-256 y validar geometría serán gates tras autorización.
- Se conservan predicciones históricas y parciales. No reutilizarlas sin recibos
  ni generarles recibos retroactivos. Evaluación en una sesión completamente nueva.

El JSON contiene rutas e identificadores internos. No es un artefacto público.

## Ejecución propuesta

Orden B0 → A → B, un proceso GPU por brazo y orden de pacientes congelado en cada
brazo. `sync`, ventanas 128³, stride 64, promedio de probabilidades y argmax.
Mismo preprocesamiento del evaluador revisado; no intercambiarlo silenciosamente
por el pipeline oficial. Sólo se reutilizan nuevas predicciones con recibos que
coincidan en datos, checkpoint, protocolo y contenido.

El supervisor específico `research/run_frozen_final_evaluation.py` está implementado
y pasa ocho pruebas de fixture. Su modo predeterminado hace únicamente preflight
de metadatos: no lee contenido final, crea salida o usa GPU. La ejecución exige
simultáneamente autorización explícita, reconocimiento de reutilización y el hash
exacto del plan. Los runners de desarrollo no se reutilizan como lanzador final.

## Presupuesto, no promesa de duración

| Límite | Valor |
|---|---:|
| Piso físico C: | 20 GiB |
| Presupuesto combinado | 5 GiB |
| Datos referenciados | 889,607,223 bytes |
| Tres checkpoints referenciados | 407,688,025 bytes |
| Reserva adicional | 256 MiB |
| Artefactos nuevos y temporales, combinados | 3,802,978,416 bytes |
| RAM disponible mínima | 1.5 GiB |
| Swap máximo | 192 MiB |
| RSS máximo del árbol | 4.5 GiB |
| Techo temporal global propuesto | 4 horas |
| Reserva de cierre dentro del techo | 30 minutos |
| Techo por invocación de un modelo | 45 minutos |

Las cuatro horas son un presupuesto máximo propuesto, **no una estimación validada
de que 18 inferencias completas caben**. Un caso de desarrollo en aproximadamente
un minuto no permite extrapolar a volúmenes finales mayores. Después de autorizar,
los headers permitirán estimar ventanas y reservas antes de ejecutar. Si no cabe,
se informa y no se reduce resolución, cohorte o rigor de manera silenciosa.

El límite combinado incluye conservadoramente datos/checkpoints referenciados aun
sin copiarlos. Los históricos permanecen intactos y el piso de disco protege su
coexistencia. Prohibidas descargas automáticas; un dato faltante detiene admisión.

## Análisis congelado y cierre

Comparación principal descriptiva: diferencias pareadas de Dice tumoral B−A.
Reportar también A−B0 y B−B0, todos los pacientes, media, mediana, número mejorado
y denominadores. Secundarias: Dice renal clase 1 y conjunto renal, precisión,
sensibilidad; quiste exploratorio. HD95 sólo si es viable, con ausencias explícitas.
No convertir ambos-vacíos en un valor arbitrario ni ocultar los Dice cero.

El criterio orientativo previo (tumor +0.03 frente a B0, caída renal ≤0.02 y
mejora en ≥3/6 casos) es una señal piloto, no umbral clínico ni prueba estadística.
No se usarán estos datos para seleccionar o promover automáticamente candidatos.

Ante fallo: conservar recibos/resultados válidos, cerrar con estado incompleto,
no mezclar parciales históricos con resultados nuevos ni presentar una comparación
completa con menos de seis casos. No flexibilizar guardas después del fallo.

## Gates pendientes

1. COMPLETADO: auditoría textual histórica; confirma reutilización en r1 y no
   encontró referencias adicionales fuera de r1/r2 e inventarios, sin probar ausencia absoluta.
2. COMPLETADO: supervisor final y ocho pruebas offline; preflight real de metadatos aprobado.
3. COMPLETADO posteriormente: el usuario autorizó leer y evaluar los seis casos,
   reconociendo la reutilización de la cohorte. El supervisor verificó los hashes.
4. COMPLETADO en v6: evaluación completa y pareada B0/A/B con seis casos por
   brazo y 18 recibos verificados. Es una réplica exploratoria de la misma
   cohorte de r1, no una validación independiente.

La demostración de Shadow Trainer para Circuito14 puede avanzar sin esperar a
este gate científico. No cambia `sync` predeterminado ni cierra Stage38 largo.

## Registro de intentos autorizados, 21 de septiembre de 2026

- La sesión `stunet-r2-final-fresh-r1` usó lectura directa del proxy NIfTI y se
  detuvo a petición del usuario tras aproximadamente 18 minutos en el primer
  volumen grande. Cero casos confirmados; estado `incomplete`. Se conservó.
- La sesión `stunet-r2-final-v3-r1` usó caché decodificada local, ligada al SHA-256
  del archivo fuente y verificada mediante recibo. B0 confirmó 2/6 casos, pero
  el supervisor cerró toda la tanda a los 1,192 segundos por `swap_limit`:
  213,803,008 bytes observados frente al techo de 201,326,592. El mínimo de
  RAM disponible fue 2,912,432,128 bytes; el pico RSS del árbol fue
  4,268,519,424 bytes y el mínimo de disco libre 34,371,588,096 bytes.
  A y B no se iniciaron. No existe comparación primaria ni conclusión de eficacia.
- La lectura v3 avanzó donde la directa no lo hizo, pero **no es una medición
  formal de aceleración**: ambos intentos quedaron incompletos y el protocolo
  cambió. No reutilizar los dos resultados parciales para ajustar ni seleccionar.
- La implementación v4 lee los volúmenes comprimidos por bloques contiguos en
  el eje de almacenamiento NIfTI y conserva una caché Fortran-order con identidad
  de versión distinta. Pasaron 21 pruebas offline de caché, procedencia y
  supervisor, incluida una imagen NIfTI comprimida con escala. En ese momento
  faltaba equivalencia física de desarrollo y el swap residual rondaba 171 MiB.

La continuación v4 completó la prueba física de desarrollo: D1 reprodujo
exactamente la máscara histórica en RTX 3050 Ti, y la segunda invocación
reutilizó la predicción sellada. El plan v4 conservó exactamente cohorte,
checkpoints, métricas y límites de v3; su preflight de metadatos pasó.

La sesión `stunet-r2-final-v4-r1` confirmó B0 en 4/6 casos y se detuvo por
`swap_limit` tras 593 muestras: máximo 229,634,048 bytes frente al techo
201,326,592. El pico RSS del árbol fue 4,193,132,544 bytes, el mínimo de
RAM disponible 2,976,051,200 bytes y el de disco libre 31,652,499,456 bytes.
A/B no se iniciaron. Ninguna cifra de estos cuatro casos es una comparación
completa ni puede sustentar eficacia. Se conservaron sus recibos y trazas.

La siguiente iteración, v5, elimina únicamente la caché decodificada
verificada y reconstruible de cada caso después de sellar su predicción, y
evita materializar máscaras gigantes cuando HD95 ya se marcaba omitido por
presupuesto. Pasó 23 pruebas offline y la prueba física D1. Su sesión final
confirmó B0 4/6, pero también se detuvo por swap: 206,340,096 bytes frente
al techo de 201,326,592. El pico coincidió con el quinto caso, donde HD95
secundario aún exigía transformadas de distancia de alto consumo. A/B no se
iniciaron.

V6 declaró un umbral más conservador para HD95 (20 millones de vóxeles);
los seis casos finales lo exceden y registran `skipped_memory_budget`.
Dice, precisión y sensibilidad no cambiaron. Pasó 24 pruebas offline y D1
físico con igualdad exacta de máscara. El plan v6 mantuvo los mismos pacientes,
checkpoints y límites. La sesión `stunet-r2-final-v6-r1` terminó B0/A/B con
6/6 casos cada uno. El recibo posterior `research/evidence/stunet-r2-final-v6-r1/paired-audit.json`
verificó hashes, 18 predicciones, recursos y diferencias pareadas. El análisis
y sus limitaciones están en `docs/STUNET_R2_FINAL_V6_RESULTS_ES.md`.

Se conservan los cuatro intentos incompletos y sus guardas; no se redujo
la cohorte ni se relajó el techo de swap. V6 demuestra una ejecución completa
de evaluación volumétrica, no validación clínica o superioridad de B sobre A.
