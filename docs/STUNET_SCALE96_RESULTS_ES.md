# STU-Net-S: escalón exploratorio de 96 casos y TurboConv

**Estado (23-09-2026):** entrenamiento, auditoría de integridad y comparación
volumétrica exploratoria completos. No es validación clínica ni una comparación
controlada que aísle el efecto de añadir 32 pacientes.

## Entrenamiento físico

El brazo A se inició desde Stage20 y usó los 64 casos del split local más 32
seleccionados por hash del manifiesto KiTS23, excluyendo todo el CSV de
validación. Dos épocas de ocho parches por caso produjeron 1,536/1,536 pasos
confirmados. El plan tiene SHA-256
`f4ce5836201a48fb375a31c842cb197788665676f988537d55f2697f3666f91f`.
Una expiración OAuth detuvo la primera invocación tras 32 pasos; después de la
reconexión de `datatesis_ro`, se reanudó sin repetir fronteras confirmadas.
`training-audit.json` comprobó el orden de las 192 fronteras de caso, las
1,536 pérdidas finitas y el checkpoint del paso final. El tiempo activo total
registrado fue 6,406 s (~1 h 47 min).

| Desarrollo, cuatro casos usados para selección | Stage20 | Época 1 | Época 2 |
|---|---:|---:|---:|
| Dice renal medio | 0.656 | 0.664 | 0.596 |
| Dice tumoral medio | 0.025 | 0.020 | 0.051 |

Ninguna época cumplió el criterio predefinido: aumentar Dice tumoral sin perder
más de 0.02 de Dice renal respecto a Stage20. Por tanto, el checkpoint de 96
**no se selecciona ni se promueve**. Se evaluó el último checkpoint únicamente
para diagnosticar la trayectoria de entrenamiento.

## Comparación de segmentación, seis casos ya abiertos

Se reutilizó la misma cohorte KiTS23 de la evaluación de 64 casos, con
ventanas 128³, stride 64, protocolo y referencias idénticos. Se verificaron
los recibos y hashes de las doce predicciones comparadas. **Estos seis casos
ya se habían abierto; no son una nueva evaluación independiente.** Además de
los 32 casos adicionales, cambiaron la semilla y el número total de pasos.

| Dice medio en seis pares | 64 casos | 96 casos | Diferencia | Casos mejores / peores / iguales |
|---|---:|---:|---:|---:|
| Riñón, clase 1 | 0.5937 | 0.6841 | +0.0904 | 6 / 0 / 0 |
| Tumor, clase 2 | 0.1866 | 0.2052 | +0.0187 | 4 / 1 / 1 |
| Quiste, clase 3 | 0 | 0 | 0 | 0 / 0 / 2 pares definidos |

En tumor, las diferencias por alias P1–P6 fueron `+0.0682`, `+0.0576`,
`+0.1326`, `0`, `+0.0097` y **`−0.1561`**. La media positiva oculta un
retroceso importante en P6. La señal renal es más uniforme, pero la caída
renal en desarrollo impide usar la cohorte reutilizada para elegir el modelo
que mejor luce. No se calculó HD95 por presupuesto de memoria.

### Diagnóstico del conflicto de cohortes

La caída renal de desarrollo tras la segunda época no es uniforme. Frente a
Stage20, el Dice de la clase renal por paciente cambió así:

| Caso de desarrollo | Stage20 | 96, época 2 | Δ | Vóxeles renales predichos, Stage20 → época 2 |
|---|---:|---:|---:|---:|
| D1 (`case_00569`) | 0.547 | 0.606 | +0.059 | 204,158 → 78,526 |
| D2 (`case_00587`) | 0.801 | 0.852 | +0.050 | 121,990 → 105,867 |
| D3 (`case_00483`) | 0.562 | 0.489 | −0.073 | 313,537 → 101,554 |
| D4 (`case_00497`) | 0.714 | 0.437 | −0.277 | 200,371 → 62,711 |

En D4, la predicción de riñón de clase 1 se contrajo 69% y su sensibilidad
cayó de 0.689 a 0.282, mientras la precisión subió de 0.740 a 0.967. El
Dice tumoral pasó de 0.022 a 0.161. Es una infra-segmentación renal medible;
las métricas agregadas no permiten atribuir causalmente el cambio a los 32
casos adicionales ni explicar dónde fueron todos los vóxeles perdidos. El
evaluador histórico y el acotado
reprodujeron exactamente el Dice de Stage20 en los cuatro casos, por lo que
una discrepancia obvia entre esos dos evaluadores no explica el conflicto.
Ni D4 ni los seis casos reutilizados justifican elegir el mejor modelo después
de mirar sus resultados.

## Recursos de Shadow Trainer

Staging `sync`, caché de entrenamiento ≤2 GiB y sesiones separadas. En
entrenamiento: espacio físico mínimo 30.856 GiB, RAM disponible mínima
2.698 GiB, swap máximo 108.54 MiB y RSS máximo del árbol 3.016 GiB. En
evaluación volumétrica: 29.573 GiB libres mínimos, 2.8 GiB de RAM
disponible mínima, 136.66 MiB de swap máximo y 4.024 GiB de RSS máximo del
árbol. La sesión de entrenamiento terminó alrededor de 2.4 GiB y la nueva
sesión de evaluación alrededor de 4.9 MiB; ambas bajo 5 GiB. Son medidas de
estas invocaciones, no una garantía universal de ausencia de OOM.

## TurboConv en ambos checkpoints

La auditoría CPU de los 43 pesos Conv3d W6 produjo error L2 relativo 0.06391
para PTQ y 0.04284 para TurboConv en el modelo de 64; 0.06390 y 0.04283,
respectivamente, en el de 96. En W4, los pares fueron 0.27639/0.18956 y
0.27634/0.18952. Es fidelidad de pesos, no inferencia cuantizada real.

Un piloto separado W6A8 usó un parche central 64³ de un CT de entrenamiento
para calibración y cuatro parches centrales de los casos de desarrollo para
comparación FP32/PTQ/TurboConv. La mediana de error L2 relativo de logits fue:

| Checkpoint | PTQ | TurboConv | Casos con menor error TurboConv |
|---|---:|---:|---:|
| 64 casos | 0.2774 | 0.2555 | 4/4 |
| 96 casos | 0.2242 | 0.1734 | 4/4 |

La mediana del coseno de logits para 96 fue 0.98375 (PTQ) frente a 0.98720
(TurboConv); el acuerdo de argmax fue 0.99211 frente a 0.99264. Aunque este
piloto favorece TurboConv, usa cuatro parches de desarrollo y fake-quant en
punto flotante. **No** demuestra Dice volumétrico retenido, latencia menor,
reducción real de VRAM ni un backend entero. El resultado antiguo de W4A8
sobre otro STU-Net de 105 clases tampoco se traslada a este checkpoint KiTS de
cuatro clases.

## Decisión y siguiente gate

No escalar automáticamente a 128 casos. Se completó un comparador pos hoc:
64 casos por tres épocas frente a los 96 por dos épocas completados, ambos
con 1,536 actualizaciones, Stage20, semilla 20260922, pérdida y estrategia
`sync` iguales. El brazo de 64 terminó y pasó auditoría. En desarrollo fue
mejor en Dice renal para los cuatro pacientes; en seis casos reutilizados,
96 conservó ventaja renal (+0.0438) pero 64 fue mejor en tumor (+0.0687).
El contraste iguala cómputo, no exposiciones por paciente, y fue diseñado
después de ver el resultado de 96. Los detalles y límites están en
`docs/STUNET_FIXED_BUDGET_COMPARISON_ES.md`. Una cohorte verdaderamente nueva
y un protocolo prospectivo siguen pendientes. Para TurboConv, el gate es
calibración más amplia y fidelidad de máscaras de volumen completo; el piloto
de cuatro parches solo justifica continuar investigándolo.

Evidencia primaria: `research/workspace/stunet-train96-a-v1/{plan.json,
supervisor.json,training-audit.json,arm_A/metrics.jsonl}` y
`research/workspace/stunet-train96-reused-six-r1/{plan.json,paired-audit.json,
evaluation/evaluation/A/summary.json}`. Auditorías TurboConv:
`/home/raulprtech/clinical_core/results_vision/kits_stunet_turboconv_64_weight_audit_r1.json`,
`kits_stunet_turboconv_96_weight_audit_r1.json`,
`kits_turboconv_patch64_w6a8_r1.json` y `kits_turboconv_patch96_w6a8_r1.json`
en el mismo directorio.
