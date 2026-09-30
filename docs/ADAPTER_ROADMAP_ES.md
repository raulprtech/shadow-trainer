# Shadow Trainer: expansión gradual por adaptadores

**Decisión de producto, 24-09-2026.** Shadow Trainer no se define por STU-Net
ni por KiTS23. Su núcleo coordina recursos, staging, ejecución, recuperación y
evidencia; cada familia de modelos entra mediante un adaptador explícito. La
adición de un nombre al roadmap **no significa** que esté entrenado, sea
compatible con 4 GiB o tenga un resultado de calidad validado.

## Límite arquitectónico

| Núcleo reutilizable | Responsabilidad del adaptador |
|---|---|
| Perfil y admisión de VRAM, RAM, swap y disco | Modelo, configuración y estado entrenable |
| Fuentes, manifiestos, integridad y caché acotada | Preparación, muestreo y semántica de datos |
| Eventos, tiempo máximo, interrupción y checkpoint durable | Serializar/restaurar modelo, optimizador, RNG y sampler |
| Reporte, hashes y procedencia | Métricas, comparador y condiciones de validez del dominio |

La estrategia `sync` es la predeterminada. El gate largo STU-Net de
equivalencia de `prefetch` sigue pendiente; los pares de fixtures pequeños no
autorizan prefetch universal. El staging de datos y la política de VRAM son
problemas separados: que los datos quepan en caché no garantiza que las
activaciones, gradientes y optimizador quepan en GPU.

## Secuencia de expansión

| Familia / candidato | Pregunta de adaptación | Primer gate, sin claim de calidad |
|---|---|---|
| CNN volumétrica: STU-Net-S | Consolidar entrenamiento, evaluación, recuperación y gate largo de backend | Recibos físicos y cohortes auditadas; ya hay pilotos, no validación clínica |
| nnU-Net v2 desde cero | Conectar planificación/preprocesamiento y carga aleatoria de casos a staging acotado; adaptar plan de memoria sin ocultar diferencias con nnU-Net estándar | Piloto 2–4 casos, luego mismos 64 casos/splits de STU-Net solo si disco y 4 GiB lo permiten |
| ResNet18/50 y 2.5D | Generalizar de volúmenes a imágenes/series; distinguir encoder congelado, ajuste y entrenamiento | Equivalencia de reanudación y comparación con baseline nativo, no solo fixture Tiny3D |
| Audio/TTS: Pocket TTS como candidato | Probar datasets de audio por shards, secuencias variables, CPU/GPU y checkpoints específicos | Revisar código de entrenamiento, licencias y consentimiento de voz; fixture CPU pequeño antes de calidad de audio |
| Modelos de decisión, incluidos Jev-like | Entrenar/evaluar clasificadores con contexto sin asumir que son LLM | Métricas de clasificación, calibración, latencia y coste contra reglas y modelos sencillos |
| Lenguaje: variantes pequeñas de Gemma/Llama; MedGemma cuando el caso sea legítimo | Empezar por adaptación de parámetros eficiente, datos/optimizer offload y evaluación congelada | Preflight por variante concreta; no prometer entrenamiento completo de pesos base en 4 GiB |
| Segmentación guiada: SAM/MedSAM y variantes | Adaptar encoder/decoder, prompts visuales y máscaras | Definir tarea y tipos de prompt; evaluar integridad de máscara y coste, no asumir capacidad lingüística en todas las variantes |
| Multimodal visión-lenguaje | Orquestar imagen y texto con datos y criterios propios | Solo tras contratos y gates anteriores; no mezclar con la tesis clínica sin pregunta aprobada |

Pocket TTS tiene código de entrenamiento publicado, pero su disponibilidad no
demuestra viabilidad local ni calidad tras adaptación. SAM/MedSAM son primero
modelos de **segmentación guiada por prompts**; algunas variantes aceptan
lenguaje, pero no se atribuirá esa capacidad a todas. MedGemma incluye
variantes médicas de texto/multimodal y exige un protocolo clínico separado
para cualquier afirmación médica.

## Gate común por adaptador

1. Congelar modelo/versión, licencia, dataset permitido, split y objetivo;
   comprobar espacio físico antes de crear cachés.
2. Fixture sintético offline: plan, caso válido/truncado, presupuesto,
   interrupción, restauración de estado y salida reproducible.
3. Piloto físico mínimo bajo guardas sin solapar trabajos GPU; registrar VRAM,
   RAM, swap, disco, transferencias y tiempo.
4. Carga real acotada y evaluación de dominio con baseline nativo. Distinguir
   factibilidad de sistemas, fidelidad de recuperación y calidad del modelo.
5. Promover un adaptador estable solo dentro de su alcance probado; conservar
   fallos y restricciones en el ledger. No inferir aceleración universal ni
   generalización clínica por transferir el runtime a otra arquitectura.

## Frontera con la tesis

La tesis puede usar Shadow Trainer para STU-Net, nnU-Net y ResNet. Si el tiempo
y los recursos alcanzan, un adaptador visual posterior —por ejemplo uno
basado en SAM/MedSAM o en una variante multimodal pertinente— también puede
proporcionar un **candidato** para sustituir el extractor de visión de
Clinical-Core. Esto es compatibilidad opcional, no obligación ni promoción
automática. TTS y los modelos de decisión mantienen su camino de producto;
solo entrarían a una pregunta doctoral distinta si se justificara explícitamente.

El gate de sustitución visual exige la misma cohorte, momento de predicción,
splits y tarea que el baseline; un artefacto visual/embedding bien definido,
la misma cabeza y regla de selección cuando corresponda, y evaluación pareada
contra ResNet y el encoder STU-Net pertinente. Primero probar representación
congelada; adaptar pesos solo si el piloto y el presupuesto lo justifican.
Un mejor Dice de KiTS23 no prueba mejor pronóstico ccRCC, y la cohorte externa
bloqueada no se usa para escoger candidato. El coste de VRAM/RAM/disco/tiempo
se informa junto con calidad. Esta línea no retrasa ni depende del artículo
anti-leakage prequirúrgico-tabular.

Clinical-Core decide la pregunta/endpoint; Clinical-Nigma aplica política
médica; Shadow Trainer ejecuta y entrega evidencia, no promueve un modelo
clínico por sí solo.

Fuentes para caracterización de candidatos: [nnU-Net oficial](https://github.com/MIC-DKFZ/nnUNet),
[Pocket TTS oficial](https://github.com/kyutai-labs/pocket-tts),
[SAM oficial](https://github.com/facebookresearch/segment-anything),
[MedSAM oficial](https://github.com/bowang-lab/MedSAM) y
[MedGemma oficial](https://deepmind.google/models/gemma/medgemma/).
