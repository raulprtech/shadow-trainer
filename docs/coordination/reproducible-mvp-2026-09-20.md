# Cierre — MVP reproducible local

Objetivo del usuario: ejecutar el MVP reproducible (entorno propio, dependencias/pruebas y demo actual en GPU con guardas). Completado en el host actual; no equivale a cerrar toda la tesis ni el producto comercial.

## Requisitos y evidencia

| Requisito | Evidencia observada |
|---|---|
| Entorno propio | .venv CPython3.12.3, include-system-site-packages=false; imports del paquete desde .venv; sin rutas clinical_core/Clinical-Nigma |
| Dependencias reproducibles | 38 versiones exactas y SHA-256 en research/requirements-mvp-linux-py312.lock; instalación offline con require-hashes; pip check correcto |
| Paquete instalable | Wheel no editable construido e instalado; 21 módulos Python coinciden con fuentes |
| Pruebas | Suite completa 65 aprobadas, sin fallos/errores/skips; JUnit conservado |
| GPU física | RTX3050Ti 4096MiB, CUDA13, 12 pasos Tiny3D finitos, sync, 22MiB reservados |
| Guardas | Exclusión por lock y comprobación sin procesos de cómputo NVIDIA; monitor cada ~1s, máximo180s, piso disco20GiB, RAM≥512MiB durante ejecución, swap≤256MiB, RSS≤4.5GiB |
| Artefactos | Checkpoint, job/plan/environment/events/summary, report-summary y HTML, hashes verificados |
| Reconstrucción | docs/REPRODUCIBLE_MVP.md y scripts de preparación/aceptación |

## Recursos y límites

Instalación prevista: 4,856,694,791 bytes extraídos +256MiB de reserva, inferior a5GiB. No hubo descarga. Se recuperaron ruedas de caché por hardlink de archivos comprimidos; los paquetes instalados son copias independientes. Los directorios wheelhouse-r1/r2 no duplican físicamente todas sus ruedas; r1 se conserva como intento diagnóstico. No se borró caché ni evidencia.

Durante demo: mínimo disco38,426,419,200bytes, mínimo RAM4,227,674,112bytes, máximo swap3,956,736bytes, máximo RSS árbol1,453,359,104bytes. El monitor observa muestras, no garantiza máximos entre muestras.

El resolvedor inicialmente eligió la variante CPU y después una etiqueta py2 de una rueda universal no fue aceptada. No se instaló esa resolución: se corrigieron pins exactos y selección de etiqueta Python3 antes de instalar CUDA. Identidad de ruedas locales no implica autenticidad del proveedor.

Los imports de staging histórico ahora se difieren hasta lanzar una campaña STU-Net. No se tocaron pacientes, checkpoints históricos, evaluación r2 ni Stage38. Tampoco se habilitó prefetch estable ni se afirmó mejora clínica.

## Qué sigue, fuera de este cierre

Paquete Circuito14: video, presentación, revisión de IP/equipo y evidencia comercial. Prueba en otra máquina requiere recursos, ruedas compatibles y autorización propia. Para resultados científicos quedan adjudicación/procedencia, evaluación r2 y el gate largo independiente.
