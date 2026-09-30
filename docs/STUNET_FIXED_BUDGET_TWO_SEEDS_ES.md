# STU-Net-S: auditoría de dos semillas para 64 frente a 96 casos

**Estado:** cuatro entrenamientos completos y auditoría offline verificada el
24-09-2026. Esta es una comparación exploratoria sobre los mismos cuatro
pacientes de desarrollo; no equivale a una evaluación independiente.

## Pregunta y controles

¿Reemplazar la tercera exposición de 64 pacientes por 32 pacientes nuevos
mejora la segmentación si se mantienen 1,536 actualizaciones por brazo?
Cada semilla compara `64 × 3 × 8` contra `96 × 2 × 8` parches. Los 64 casos
iniciales, los 32 adicionales, el checkpoint Stage20, la pérdida A, el
preprocesamiento y los cuatro pacientes de desarrollo son idénticos entre
semillas. Solo cambia la semilla de entrenamiento (`20260922`, `20260923`).
Dentro de cada semilla, los primeros 512 pasos tienen los mismos pacientes,
centros, flips, clases objetivo y orden en ambos brazos. El staging fue
`sync` y no se tocó el test bloqueado ni la cohorte de seis casos ya abierta.

La primera comparación se diseñó tras conocer el resultado de 96 casos. La
segunda semilla se eligió después de ver esa primera comparación; por tanto,
es una comprobación de estabilidad, **no** una réplica preregistrada.
Igualar actualizaciones no iguala la exposición a pacientes distintos.

## Resultado a cómputo igualado: último estado, 1,536 pasos

Dice medio de clase en cuatro pacientes de desarrollo. `Δ` es 64 menos 96;
un valor positivo favorece a 64. Son métricas de inferencia volumétrica
calculadas por el worker al final de cada época.

| Semilla | Riñón 64 | Riñón 96 | Δ renal | Tumor 64 | Tumor 96 | Δ tumoral |
|---|---:|---:|---:|---:|---:|---:|
| 20260922 | 0.7374 | 0.5956 | +0.1418 | 0.0661 | 0.0505 | +0.0156 |
| 20260923 | 0.7624 | 0.7624 | +0.0001 | 0.0141 | 0.0266 | −0.0125 |

En la primera semilla, 64 mejoró el Dice renal en 4/4 pacientes, pero el
Dice tumoral sólo en 1/4; dos empeoraron y uno empató. En la segunda, 96
mejoró el Dice renal en 3/4, aunque la media renal quedó prácticamente
empatada por un caso favorable a 64. El tumor quedó 1/1/2 en mejoras de
64/96/empates. **El signo de la diferencia tumoral media cambió entre
semillas; no hay una mejora estable por aumentar los casos.**

| Semilla | Caso | Riñón 64 | Riñón 96 | Tumor 64 | Tumor 96 |
|---|---|---:|---:|---:|---:|
| 20260922 | D1 | 0.7113 | 0.6057 | 0 | 0.0002 |
| 20260922 | D2 | 0.8810 | 0.8516 | 0 | 0 |
| 20260922 | D3 | 0.6042 | 0.4887 | 0.1712 | 0.0411 |
| 20260922 | D4 | 0.7534 | 0.4365 | 0.0933 | 0.1609 |
| 20260923 | D1 | 0.7215 | 0.7665 | 0 | 0 |
| 20260923 | D2 | 0.8733 | 0.8832 | 0 | 0 |
| 20260923 | D3 | 0.6219 | 0.6565 | 0.0456 | 0.0237 |
| 20260923 | D4 | 0.8330 | 0.7433 | 0.0107 | 0.0828 |

## Resultado según selección de checkpoint

La regla vigente elige el Dice tumoral medio más alto si el Dice renal no
queda más de 0.02 por debajo de Stage20; si ninguna época cumple, mantiene
Stage20. Este análisis responde otra pregunta y **no conserva necesariamente
el mismo número de actualizaciones**:

| Semilla | Selección 64 | Selección 96 | Dice renal 64/96 | Dice tumoral 64/96 |
|---|---|---|---:|---:|
| 20260922 | época 3, paso 1536 | Stage20, paso 0 | 0.7374 / 0.6558 | 0.0661 / 0.0253 |
| 20260923 | época 2, paso 1024 | época 2, paso 1536 | 0.7278 / 0.7624 | 0.0704 / 0.0266 |

En la segunda semilla, la última época de 64 subió el Dice renal a 0.7624,
pero bajó el tumoral a 0.0141 y dejó de ser elegible. El checkpoint elegido
de 96 apenas superó la referencia tumoral Stage20 (0.0266 frente a 0.0253).
Por ello sería incorrecto citar el 0.0704 tumoral de 64 frente al 0.0266 de
96 como resultado «a 1,536 pasos iguales»: esa comparación usa 1,024 y
1,536 pasos, respectivamente.

## Integridad, recursos y decisión

La auditoría comprobó cuatro planes y semillas, listas de pacientes, ausencia
de solapamiento entrenamiento/desarrollo, 1,536 pasos finitos y confirmados
por brazo, orden compartido en los primeros 512 pasos, recálculo de las medias
desde los cuatro pacientes, selección de época y hashes de los dos nuevos
checkpoints elegidos. El brazo nuevo de 64 terminó sin incidentes. El de 96
se detuvo al observar 258.3 MiB de swap frente al límite de 256 MiB; quedó
sellado en el paso 1,064 y se reanudó desde ese checkpoint sin cambiar las
guardas. Sólo se eliminó la caché descargada y reconstruible del brazo nuevo
de 64, ya terminado, para preservar el techo combinado de 5 GiB. Ambos
brazos cerraron finalmente con 1,536/1,536 pasos, sin OOM.

**Decisión:** no afirmar que 96 casos mejoran consistentemente la calidad de
segmentación frente a 64 a presupuesto de pasos fijo. Tampoco afirmar que 64
es óptimo: cuatro pacientes de selección, dos semillas dependientes del
proceso de diseño y las diferencias entre repetir casos y aumentar diversidad
no permiten esa inferencia. La evidencia de sistemas sí se amplía: dos brazos
adicionales completaron entrenamiento STU-Net-S con staging acotado y
recuperación de una parada por recursos en la GPU física de 4 GiB.

Para una afirmación de aprendizaje más fuerte harían falta un protocolo
prospectivo, semillas adicionales, una cohorte nueva y una ablación que separe
diversidad de pacientes, número de exposiciones y selección de checkpoint.
No se atribuye EDA, aceleración universal, equivalencia larga de `prefetch`
ni validez clínica. El gate largo STU-Net `sync`/`prefetch` sigue pendiente.

Evidencia primaria: `research/workspace/stunet-fixed-budget-replication-seed20260923/audit/two-seed-audit.json`
y `development-pairs.csv`; planes, recibos y métricas de las cuatro sesiones
referenciadas por `research/audit_stunet_fixed_budget_replication.py`.
