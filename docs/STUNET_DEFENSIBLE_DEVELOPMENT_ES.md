# STU-Net: resultados volumétricos defendibles de desarrollo

Actualización: la [verificación directa de máscaras](coordination/stunet-mask-provenance-2026-09-21.md)
completó 24/24 predicciones y confirmó los conteos y la geometría. Las cifras
de este informe se mantienen. Las limitaciones de procedencia histórica y de
selección en desarrollo siguen vigentes; véase el cierre para el estado actual.

## Alcance y conclusión

Auditoría offline de las tandas r1/r2, sin nuevas inferencias ni entrenamiento.
No se abrieron métricas, imágenes o etiquetas de la cohorte final bloqueada.
Los cuatro pacientes de desarrollo son los mismos en ambas tandas y se usaron
para seleccionar checkpoints: **no son ocho pacientes ni evaluación independiente**.

Hay una señal descriptiva favorable a la pérdida B para tumor, pero el rendimiento
absoluto es bajo y dos pacientes mantienen Dice tumoral cero. No permite afirmar
validación clínica, superioridad general ni recomendar un modelo para uso clínico.

## Comparación completa, sin elegir sólo la mejor tanda

Dice macro: media no ponderada de pacientes con valor definido. Riñón es clase 1;
no confundir con la agrupación renal que incluye tumor y quiste. Escala 0–1.

| Tanda / modelo | Dice riñón | Dice tumor | Dice conjunto renal | Δ tumor frente a B0 | Δ riñón frente a B0 |
|---|---:|---:|---:|---:|---:|
| B0, idéntico en ambas | 0.655816 | 0.025272 | 0.654099 | — | — |
| r1 / A | 0.668282 | 0.030147 | 0.657035 | +0.004875 | +0.012467 |
| r1 / B | 0.636209 | 0.053827 | 0.635276 | +0.028555 | −0.019606 |
| r2 / A | 0.636778 | 0.046228 | 0.645506 | +0.020956 | −0.019038 |
| r2 / B | 0.637844 | 0.071475 | 0.646186 | +0.046203 | −0.017972 |

Las dos tandas registran semillas de entrenamiento 20260915 y 20260916.
Cada brazo completó 768 actualizaciones solicitadas (24 casos × 4 épocas ×
8 parches). Los cuatro checkpoints seleccionados corresponden a época 2,
no al modelo del paso 768. La selección exige mejora tumoral y caída renal
no mayor de 0.02 frente a B0; esta restricción forma parte de la selección,
no constituye una confirmación independiente de preservación renal.

A usa la pérdida anterior; B la combinación jerárquica renal/tumoral. Los
resúmenes A/B de cada tanda tienen configuración de muestras y hash inicial
iguales. Esto no demuestra por sí solo identidad de todos los tensores de entrada.

## Visibilidad por paciente: Dice tumoral

Alias D1–D4 asignados por orden estable de identificadores, sin seleccionar casos
por resultado. La tabla conserva los ceros; no son datos ausentes.

| Desarrollo | B0 | r1 A | r1 B | r2 A | r2 B |
|---|---:|---:|---:|---:|---:|
| D1 | 0.079243 | 0.089876 | 0.147298 | 0.143788 | 0.195475 |
| D2 | 0.021845 | 0.030714 | 0.068008 | 0.041123 | 0.090425 |
| D3 | 0 | 0 | 0 | 0 | 0 |
| D4 | 0 | 0 | 0 | 0 | 0 |

B supera a A en tumor en 2/4 pacientes y empata en los otros dos. Diferencia
media B−A: +0.023679 en r1 y +0.025247 en r2. En riñón, B−A es −0.032073
en r1 y +0.001066 en r2: no hay dominio consistente en todas las métricas.
No se calcularon p-valores ni se interpretaron las semillas como pacientes nuevos.

El criterio orientativo original (+0.03 de Dice tumoral frente a B0, caída renal
≤0.02 y mejora en al menos la mitad) se alcanza descriptivamente sólo en r2/B.
r1/B queda por debajo de +0.03. Ese criterio estaba destinado a evaluación
independiente: **su cumplimiento en desarrollo no cierra el gate científico**.

Quiste: Dice 0 en los dos pacientes con valor definido; en otros dos es nulo
por ausencia en referencia y predicción. No presentar una media de cuatro
mediciones válidas ni convertir nulos a cero. CSV incluye precisión, sensibilidad
y estado de HD95; no se recalculó HD95 desde superficies en esta auditoría.

## Qué se verificó y qué no

- Seis resúmenes completos de desarrollo, mismos cuatro identificadores,
  geometría y conteos de referencia; ausencia de solapamiento con los 24 casos
  de entrenamiento registrados. No equivale a auditar todo el uso histórico.
- SHA-256 de los seis checkpoints referenciados (B0 compartido) y de las 24
  predicciones NIfTI coincidentes con los resúmenes. Lectura binaria por bloques,
  sin cargar modelos o volúmenes.
- Dice, precisión y sensibilidad recalculados desde conteos; medias, medianas
  y Dice micro contrastados con agregados guardados. 144 filas sanitizadas.
- Código actual: ventanas 128³, stride 64, promedio de probabilidades, argmax;
  agrupaciones {1,2,3}, {2,3} y {2}. No son puntuaciones oficiales del desafío.
- En el segundo pase se recalcularon los conteos desde las 24 máscaras de
  desarrollo y se verificaron shape, affine, spacing y orientación frente a
  imagen/referencia. Los hashes prueban identidad con el registro actual, no la
  revisión exacta de código utilizada al ejecutar ni la ausencia de leakage.
- r1 conserva discrepancia de metadatos del supervisor (2 épocas frente a 4
  en brazos), documentada sin reescribir originales. No utilizar sus duraciones
  finales como tiempo total ni prometer recuperación exacta STU-Net.
- r2 conserva evaluación final incompleta por guarda de swap. No mezclar sus
  tres resultados parciales con una comparación completa de seis pacientes.

## Qué puede mostrarse a sinodales / Circuito14

“Tenemos entrenamiento STU-Net y métricas volumétricas registradas sobre cuatro
casos de desarrollo. Una pérdida alternativa aumenta el Dice tumoral observado,
pero persisten fallos y la evaluación independiente sigue pendiente.”

Para Circuito14, priorizar factibilidad de ejecución y trazabilidad. Los picos
de memoria de un resumen no certifican por sí solos todo el ciclo ni todos los
límites de disco. `sync` sigue predeterminado; Stage38 largo pendiente. No EDA,
aceleración universal ni validación clínica.

## Próximo gate, antes de más entrenamiento

1. COMPLETADO: verificar offline, sólo en desarrollo, geometría e integridad de
   imagen, referencia y predicción, y recomputar conteos desde máscaras por bloques.
2. IMPLEMENTADO, pruebas focales CPU aprobadas: recibos obligatorios para reutilizar
   nuevas predicciones, ligados a checkpoint/imagen/referencia/protocolo. Los
   archivos históricos sin recibo se rechazan y no se certifican retroactivamente.
   Se desactivó la caché CT temporal de identidad débil; ahora se lee por bloques
   del NIfTI de origen. El recorrido GPU actualizado aún no se ha ejecutado.
3. Congelar código, recibos de procedencia y protocolo de evaluación antes de
   autorizar completar r2. No cambiar hiperparámetros mirando resultados finales.
4. Sólo con gate de recursos y alcance autorizados, completar evaluación
   independiente sin reentrenar A/B; reportar todas las comparaciones pareadas.
5. Para artículo de aprendizaje falta generalización; para sistemas faltan gates
   STU-Net de recuperación y comparación larga exacta. No los sustituye Tiny3D.

## Evidencia reproducible

Auditor: `research/audit_development_evidence.py` (stdlib, sin GPU).
Bundle: `research/evidence/stunet-development-audit-20260921-r1/` contiene
`audit.json` (34 hashes de fuentes más hash del auditor) y
`development-per-case.csv`. Los originales permanecen sin cambios.

```bash
python3 research/audit_development_evidence.py \
  --session research/workspace/stunet-campaign-20260915-r1 \
  --session research/workspace/stunet-campaign-20260915-r2 \
  --output research/evidence/NUEVO_DIRECTORIO
```

El auditor rechaza salida existente, incoherencia de métricas, hashes, cohortes
o configuración A/B, y disco C: inferior al piso de 20 GiB más reserva mínima.
