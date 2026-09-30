# Shadow Trainer — descripción ejecutiva para Circuito 14

Para el borrador vigente de postulación ver [resumen ejecutivo de revisión](circuito14-review-2026-09-20/executive-brief.md). Este documento conserva el contexto y las cifras históricas; no sustituye la revisión humana de equipo, afiliación, IP y mercado.

## Problema

Los equipos de ingeniería AI/edge trabajan con modelos, workspaces temporales y
datasets que frecuentemente superan la VRAM o el almacenamiento local
disponible. Esto obliga a reducir experimentos, configurar manualmente cada
ejecución, comprar estaciones de trabajo o mover datos sensibles a la nube.
Los fallos suelen atribuirse sólo a la GPU aunque también intervienen RAM, swap,
disco, transferencia y memoria temporal de operadores.

## Solución

Shadow Trainer es un runtime local que inspecciona los recursos reales, rechaza
trabajos inseguros, decide una estrategia conservadora, mueve casos completos
desde almacenamiento local o remoto mediante una caché acotada, conserva
checkpoints reanudables y produce evidencia auditable de cada ejecución.

La primera validación usa una CNN volumétrica porque representa una carga
especialmente exigente. KiTS23 es únicamente el workload de sistemas; Shadow
Trainer no se presenta como dispositivo ni solución clínica.

## Diferenciación

- Presupuesto conjunto de VRAM, RAM, swap, disco y localidad de datos.
- Staging atómico por casos con integridad y LRU.
- Admisión previa a reservar GPU o descargar datos.
- Reinicio reproducible con modelo, optimizador y RNG.
- Estrategias experimentales separadas de las habilitadas por defecto.
- Reporte legible y trazabilidad de resultados negativos.

## Validación histórica comprobable del MVP compacto

Estas cifras corresponden al corte original del MVP, no a una nueva ejecución
ni al conteo actual de pruebas. Para afirmaciones vigentes prevalece el
[ledger canónico](../research/CLAIMS_LEDGER.md). La matriz física posterior
aportó 24 pares exactos de fixtures; no cierra el gate largo STU-Net ni demuestra
aceleración universal. No se ha demostrado un workload o piloto EDA.

- Ejecución física en NVIDIA RTX 3050 Ti Laptop de 4096 MiB.
- Demo de producto de 12 pasos finalizada en 2.32 segundos.
- CNN NIfTI real sobre dos casos KiTS, patches 64 al cubo y pérdidas finitas.
- Pico de 64 MiB de VRAM reservada en el adaptador compacto.
- Caché física de 128 MiB con 83,549,098 bytes ocupados.
- Guarda de disco observando el espacio real de C: mediante /mnt/c.
- Calibraciones físicas ResNet18 y ResNet50 2.5D proxy con 258 y 488 MiB de VRAM reservada.
- Suite automatizada del corte original: 30 pruebas aprobadas, incluida equivalencia exacta de reanudación para Tiny3D; no reejecutadas para esta actualización documental.

Estas cifras describen el MVP compacto. Las calibraciones ResNet son pruebas
de integración, no comparaciones de velocidad o calidad predictiva. Los resultados de STU-Net del
laboratorio Stream-HOT se mantienen como evidencia separada y no se mezclan con
la demo del producto.

## Cliente inicial

Equipos de ingeniería y validación AI/edge en empresas de alta tecnología,
centros de I+D y proveedores especializados de Jalisco que necesitan iterar en
hardware limitado, mantener datos localmente y demostrar cómo se ejecutó cada
trabajo.

## Modelo comercial inicial

1. Piloto técnico pagado para caracterizar una carga y hardware del cliente.
2. Licencia anual del runtime y sus adaptadores.
3. Soporte, integración y desarrollo de políticas/workloads especializados.

Los precios se definirán después de entrevistas y pilotos; no se incluirán
cifras no validadas en la postulación.

## Relación con Jalisco

El proyecto se desarrolla desde el entorno CINVESTAV y puede convertirse en
software especializado, propiedad intelectual y servicios técnicos de alto
valor para empresas del ecosistema electrónico, embedded y AI de Jalisco. Su
crecimiento requeriría perfiles de sistemas, CUDA, machine learning,
benchmarking y validación.

## Siguiente hito

Conseguir una primera carga de un equipo AI/edge de Jalisco, generar su perfil
de recursos y comparar una ejecución manual contra Shadow Trainer con criterios
acordados antes del piloto.
