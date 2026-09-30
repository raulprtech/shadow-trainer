# Pitch de ocho láminas — borrador

## 1. Shadow Trainer

Entrenamiento local con límites explícitos

Un runtime para ejecutar y auditar cargas de IA cuando la VRAM, RAM y disco son limitados.

Límite: Borrador para revisión · Circuito14 · evidencia local, no validación clínica

## 2. El problema

El límite no es sólo la GPU

Un trabajo también consume RAM, espacio temporal, caché y transferencia de datos. Nuestra hipótesis es que coordinarlos reduce el trabajo manual de preparar y recuperar experimentos.

Límite: Hipótesis de cliente: equipos AI/edge e ingeniería de validación. Falta validación comercial.

## 3. La solución

Inspeccionar → admitir → ejecutar → recuperar → reportar

Shadow Trainer aplica presupuestos configurados, prepara datos por casos, conserva checkpoints y produce un informe local de ejecución.

Límite: sync es el modo estable. prefetch permanece experimental para el protocolo largo.

## 4. Demostración actual

MVP instalado en una RTX 3050 Ti de 4 GiB

12 pasos de una CNN Tiny3D sintética, ejecutados con CUDA y estrategia sync. Caché acotada, checkpoint y reporte HTML generados.

Límite: Demo pequeña de sistemas: no es un entrenamiento STU-Net completo ni una medida de calidad clínica.

## 5. Evidencia verificable

65 pruebas aprobadas · 38 dependencias fijadas

Entorno propio instalado offline; paquete no editable. Las versiones y los artefactos tienen hashes. La recuperación Tiny3D CPU reprodujo el estado de modelo, optimizador y RNG.

Límite: Validado en esta PC; no certificado en cualquier equipo. La demo GPU nueva fue ininterrumpida.

## 6. Diferenciación propuesta

Un flujo operativo auditable

Integramos admisión, staging, integridad, presupuestos, checkpoints y evidencia. Los resultados fallidos se conservan y no sustentan cifras comerciales.

Límite: No afirmamos que cada componente sea nuevo. Baseline comparable y revisión de IP pendientes.

## 7. Cliente y piloto

Un primer caso industrial medible

Proponemos un piloto acotado con una carga AI/edge: comparar operación habitual y Shadow Trainer, acordando antes éxito, recursos y reproducibilidad.

Límite: Modelo comercial por validar: piloto técnico, licencia y soporte. No hay tracción acreditada en este paquete.

## 8. Solicitud y próximos pasos

Validar encaje industrial en Jalisco

Buscamos acceso a una carga piloto, revisión de IP y acompañamiento comercial. Próximo paso: cerrar equipo, entrevistas y expediente.

Límite: Independiente del artículo de tesis. Sin aplicación EDA demostrada, aceleración universal ni validación clínica.
