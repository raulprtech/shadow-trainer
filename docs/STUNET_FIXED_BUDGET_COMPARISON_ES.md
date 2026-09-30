# STU-Net-S: contraste exploratorio de 64 y 96 casos con cómputo igualado

**Estado:** entrenamiento, auditoría y evaluación volumétrica exploratoria
completos. Este contraste se diseñó
después de observar el resultado del brazo de 96 casos, por lo que es pos hoc.
Una segunda semilla y su auditoría están documentadas en
[STUNET_FIXED_BUDGET_TWO_SEEDS_ES.md](STUNET_FIXED_BUDGET_TWO_SEEDS_ES.md).

## Pregunta y diseño

¿La mayor diversidad de 96 casos supera a repetir más veces los 64 casos
originales cuando ambos brazos reciben exactamente 1,536 actualizaciones?

| Brazo | Casos | Épocas | Parches/caso/época | Actualizaciones |
|---|---:|---:|---:|---:|
| 64 | 64 | 3 | 8 | 1,536 |
| 96 | 96 (64 originales + 32 nuevos) | 2 | 8 | 1,536 |

Ambos partieron del mismo Stage20 y usaron semilla `20260922`, la misma
pérdida, optimizador, arquitectura STU-Net-S, parches 128³, orden de los 64
casos compartidos y staging `sync`. Los primeros 512 pasos tuvieron idénticos
identificadores, centros, flips y clases objetivo; la máxima diferencia de
pérdida fue `9.54e-7`. Después de ese punto, 96 vio 32 casos nuevos mientras
64 inició su segunda exposición a los originales. Igualar pasos **no** iguala
la exposición de pacientes ni aísla una causa única.

Los cuatro pacientes de desarrollo son los mismos del protocolo anterior y
pueden usarse para seleccionar checkpoint. Los seis casos utilizados en la
comparación volumétrica ya se habían abierto; sólo son diagnósticos y no se
usarán para seleccionar, promover o afirmar generalización clínica.

## Integridad y recursos del brazo de 64

El supervisor completó 1,536/1,536 pasos, 192 fronteras de caso y tres
épocas en 6,410 segundos activos. La auditoría confirmó orden, pérdidas
finitas, plan congelado y procedencia del último checkpoint.
La época 3 produjo el checkpoint elegido en desarrollo; sus 130 tensores de
modelo son exactamente iguales a los del último checkpoint evaluado, aunque
los archivos completos tienen hashes distintos por otros campos.

- SHA-256 del plan: `bf02d0210faa957c7cabea6bc1b599830bc8858808473effb915b0cb2e2f4b91`.
- SHA-256 del checkpoint último: `b61ce26bda7a6c458fcf7c7003b7944d362eba991ce39db06ba83c51d375e5ef`.
- Espacio libre físico mínimo observado: 25.847 GiB; sesión final: 2.5 GiB.
- RAM disponible mínima: 2.91 GiB; RSS máximo del árbol: 2.863 GiB; swap máximo: 123.41 MiB.

Son observaciones de esta ejecución, no una garantía universal contra OOM.
El gate largo STU-Net de equivalencia `sync`/`prefetch` sigue pendiente.

## Desarrollo: cuatro pacientes

La regla previa exige mejorar Dice tumoral medio respecto a Stage20 sin
perder más de 0.02 de Dice renal medio. El brazo de 64 cumplió en las
épocas 2 y 3; eligió la época 3. El de 96 no cumplió en ninguna época.

| Checkpoint | Dice renal medio | Dice tumoral medio | Elegible |
|---|---:|---:|---|
| Stage20 | 0.6558 | 0.0253 | Referencia |
| 64, época 1 | 0.6646 | 0.0165 | No |
| 64, época 2 | 0.6959 | 0.0427 | Sí |
| 64, época 3 | 0.7374 | 0.0661 | Sí |
| 96, época 2 | 0.5956 | 0.0505 | No |

Con el mismo número de pasos, el último checkpoint de 64 supera al de 96
en Dice renal en los cuatro pacientes. En tumor la media de 64 es mayor,
pero la mediana de las diferencias por paciente es aproximadamente cero:
la ventaja media depende principalmente de D3.

| Caso de desarrollo | Riñón 64 | Riñón 96 | Tumor 64 | Tumor 96 |
|---|---:|---:|---:|---:|
| D1 | 0.7113 | 0.6057 | 0 | 0.0002 |
| D2 | 0.8810 | 0.8516 | 0 | 0 |
| D3 | 0.6042 | 0.4887 | 0.1712 | 0.0411 |
| D4 | 0.7534 | 0.4365 | 0.0933 | 0.1609 |

**Evaluación agrupada completa el 23-09-2026:** el evaluador acotado generó
4/4 predicciones para cada brazo y la auditoría pareada verificó los ocho
hashes, recibos, orden y protocolo. En desarrollo, «riñón con masas» también
favorece a 64 en los cuatro pacientes: Dice medio `0.7389` frente a `0.6208`
de 96 (diferencia `64 − 96 = +0.1181`). La agrupación «tumor con quiste»
favorece ligeramente a 96 en media (`0.0688` frente a `0.0626`) y en 3/4
casos; el tumor solo favorece a 64 en media, pero sólo 1/4 casos mejora y
dos empeoran. No hay una ventaja tumoral robusta. Estas siguen siendo las
cuatro muestras usadas para selección, no validación independiente.

## Volumen completo: seis casos ya reutilizados

Se evaluaron los últimos checkpoints de ambos brazos con el mismo evaluador
de ventanas 128³, stride 64 y acumulación acotada, usando los mismos seis
casos y referencias. La auditoría verificó hashes y vínculos de procedencia
de las doce predicciones. **La cohorte ya estaba abierta** y no sirve para
seleccionar checkpoint ni estimar generalización independiente.

| Dice medio en seis pares | 64 × 3 | 96 × 2 | Diferencia 96 − 64 | Casos donde 96 mejora/empeora/empata |
|---|---:|---:|---:|---:|
| Riñón, clase 1 | 0.6403 | 0.6841 | +0.0438 | 6 / 0 / 0 |
| Tumor, clase 2 | 0.2739 | 0.2052 | −0.0687 | 2 / 3 / 1 |
| Quiste, clase 3 | 0 | 0 | 0 | 0 / 0 / 2 pares definidos |

Las diferencias tumorales `96 − 64` por alias P1–P6 fueron `−0.0988`,
`+0.0668`, `+0.0431`, `0`, `−0.1496` y `−0.2735`. En otras palabras, la
ventaja renal de 96 es consistente en esta cohorte reutilizada, pero el tumor
empeora en tres pacientes y la media empeora. El patrón **no** reproduce la
ventaja renal de 64 en los cuatro casos de desarrollo; no se debe combinar
ambas cohortes en una sola cifra ni escoger el brazo según la más favorable.

La ventaja de la clase renal no persiste al usar la agrupación «riñón con
masas» de KiTS: Dice medio `0.6633` para 64 y `0.6595` para 96 (diferencia
`96 − 64 = −0.0037`). En clase renal, 64 tuvo sensibilidad media `0.8824`
frente a `0.8478` de 96, pero precisión `0.5154` frente a `0.5940`; por
tanto, 64 cubre más riñón a costa de más falsos positivos. En tumor, la
sensibilidad media fue `0.2605` para 64 y `0.1555` para 96. Son métricas de
este pequeño conjunto reutilizado, no puntuaciones oficiales de KiTS ni
validación independiente. En desarrollo, a diferencia de esta cohorte de
seis casos, la agrupación «riñón con masas» sí favoreció claramente a 64.

Una auditoría offline leyó por bloques los doce NIfTI de predicción y seis
referencias ya selladas, cotejó hashes, geometría y reprodujo exactamente
todos los Dice de clase y agrupación. Su matriz de confusión suma los vóxeles
de los seis pacientes; es diagnóstico de errores, **no** el estimador principal
de calidad por paciente. Una segunda invocación idempotente tardó 5.58 s y
tuvo un RSS máximo de 90,656 KiB en esta PC; no es un benchmark general:

| Flujo de vóxeles, referencia → predicción | 64 | 96 |
|---|---:|---:|
| Fondo → riñón (falsos positivos renales) | 2,446,625 | 1,562,577 |
| Riñón → fondo (riñón omitido) | 522,072 | 701,169 |
| Tumor → fondo (tumor omitido) | 1,037,375 | 1,338,795 |
| Tumor → tumor (aciertos) | 345,580 | 362,435 |

Así, 96 reduce mucho la sobresegmentación renal pero aumenta omisiones de
riñón y tumor. El Dice tumoral **macro** (media por paciente) favorece a 64,
`0.2739` frente a `0.2052`; el Dice **micro** de vóxeles agregados favorece
levemente a 96, `0.2686` frente a `0.2496`. El paciente de 1,059 cortes y
otros volúmenes grandes pesan mucho más en micro, por lo que ambos números
deben conservar su denominador y no sustituirse uno por otro. Ningún brazo
predijo quiste de clase 3 en estos seis casos.

La evaluación duró 10.29 minutos; espacio físico mínimo 29.902 GiB, RAM
disponible mínima 2.746 GiB, swap máximo 157.55 MiB y RSS máximo del árbol
4.134 GiB. No hubo OOM ni violación observada de las guardas. HD95 se omitió
por el límite de memoria.

**Recuperación del desarrollo:** las cuatro imágenes y sus cuatro máscaras
coinciden por SHA-256 entre cachés. La primera invocación se detuvo tras D1
al detectar 195 MiB de swap, sobre el límite de 192 MiB; D1 quedó sellado.
Al volver a cero el swap del entorno, el reintento verificó y reutilizó D1 y
terminó ambos brazos 4/4 sin alterar las guardas. La admisión conserva 48 MiB
de margen bajo el límite original de 192 MiB para prevenir otro intento sin
headroom. La telemetría del reintento exitoso se separa del intento fallido:
swap máximo 0 MiB, espacio libre mínimo 37.11 GiB, RSS del árbol máximo
2.70 GiB y RAM disponible mínima 3.24 GiB. Los tiempos de 64 y 96 no son
comparables porque 64 reutilizó D1 y 96 hizo cuatro inferencias nuevas.
No se abrió test ni se reinició WSL desde este trabajo.

## TurboConv sobre el nuevo checkpoint

La auditoría de 43 convoluciones 3D del checkpoint de 64 × 3 dio error L2
relativo de pesos W6: PTQ `0.06390`, TurboConv `0.04283`; W4: `0.27634` y
`0.18952`. Un piloto W6A8 calibró con el mismo CT de entrenamiento y midió
los mismos cuatro parches centrales de desarrollo usados para el modelo de
96. La mediana de error L2 relativo de logits fue `0.26224` para PTQ y
`0.23536` para TurboConv; TurboConv tuvo menor error en los cuatro parches.
La mediana de acuerdo de argmax, en cambio, fue `0.98659` para PTQ y
`0.98590` para TurboConv. Por tanto, la mejor fidelidad de logits **no**
autoriza afirmar mejores máscaras, Dice, latencia ni menor VRAM. Es
fake-quantización, no un kernel entero desplegable.

## Interpretación y próximos gates

En la primera semilla y su cohorte de desarrollo, repetir los 64 casos fue
más favorable para la cobertura renal que sustituir la tercera exposición
por 32 casos adicionales. La segunda semilla no reprodujo esa ventaja renal
media a 1,536 pasos; véase el informe de dos semillas enlazado arriba. Esto no
demuestra que 64 sea el tamaño óptimo: hay sólo cuatro pacientes de
desarrollo, el protocolo es pos hoc y las exposiciones no son iguales.
En la primera semilla, el checkpoint de 96 no se seleccionó. Los seis casos reutilizados
muestran heterogeneidad y no cambian esa selección; sí indican que la
cobertura renal y tumoral puede moverse en direcciones opuestas entre
cohortes.

Para una afirmación publicable harán falta un protocolo prospectivo,
réplicas/semillas, una cohorte realmente nueva bloqueada y una comparación
que distinga diversidad de datos frente a repetición de ejemplos. No se
atribuye EDA, aceleración universal, equivalencia de `prefetch` ni utilidad
clínica.

Evidencia primaria: `research/workspace/stunet-fixed-budget64-a-v1/plan.json`,
`supervisor.json`, `training-audit.json`, `arm_A/metrics.jsonl`, y los recibos
homólogos de `research/workspace/stunet-train96-a-v1`.
Evaluación pareada: `research/workspace/stunet-fixed-budget64-reused-six-r1/paired-audit.json`,
`confusion-audit.json` y `evaluation/evaluation/A/summary.json`.
Desarrollo agrupado: `research/workspace/stunet-fixed-budget-development-r1/paired-audit.json`.
TurboConv: `/home/raulprtech/clinical_core/results_vision/kits_stunet_turboconv_fixed_budget64_weight_audit_r1.json`
y `kits_turboconv_patch_fixed_budget64_w6a8_r1.json`.
