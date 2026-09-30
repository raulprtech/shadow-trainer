# STU-Net: escalón exploratorio de 96 casos

El brazo A se inicia desde Stage20, no desde el checkpoint de 64 casos. Usa los
64 casos del split local de entrenamiento y 32 casos adicionales del manifiesto
KiTS23, elegidos por SHA-256 de `20260922:case_id` tras excluir todo el CSV de
validación y los casos que no quepan con los cuatro de desarrollo fijados en
la caché de 2 GiB. No se elige ningún caso por Dice ni por apariencia de la
imagen. Dos épocas y ocho parches por caso producen 1,536 actualizaciones
previstas. Staging `sync`, piso físico de 20 GiB libres y sesión ≤5 GiB.

La comparación con el modelo de 64 casos será **exploratoria**. Se medirán los
mismos seis casos KiTS ya abiertos en la evaluación de 64 casos, sin
renombrarlos como evaluación independiente nueva. Para evitar selección por
esos seis casos, el checkpoint de 96 se determinará usando únicamente los
cuatro casos de desarrollo. Se conservará además el último checkpoint completo
para una comparación de dosis de entrenamiento, declarando si el criterio de
selección en desarrollo no lo eligió. Ambos checkpoints deben etiquetarse
inequívocamente; el `best.checkpoint.pt` del brazo de 64 casos corresponde al
paso 0 y su evaluación previa utilizó `latest.checkpoint.pt` del paso 1,024.

El informe mostrará Dice renal, tumoral y de quiste por paciente, diferencias
pareadas frente a 64 y recursos. Los distintos números de pasos y la cohorte
reutilizada impiden atribuir causalmente una diferencia sólo al número de casos
o reclamar validación clínica. El resultado pronóstico de 75 CT TCIA permanece
separado y no se reinterpreta a partir del Dice KiTS.

Para cada nuevo checkpoint 3D entrenado se repetirá la auditoría CPU de pesos
TurboConv W4/W6 frente a PTQ del mismo bit-width. Es un filtro de fidelidad,
no una prueba de retención de segmentación ni de ahorro real de VRAM/latencia.
La prueba posterior de activaciones y máscaras requerirá calibración disjunta,
pares FP32/PTQ/TurboConv y GPU libre; no correrá junto al entrenamiento.
