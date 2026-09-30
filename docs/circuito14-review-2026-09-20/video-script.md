# Guion de video — aproximadamente 100–115 segundos

Borrador de narración; duración estimada, todavía no grabado ni ensayado.

## 0–18 s · Problema

Entrenar modelos de inteligencia artificial en una computadora limitada no depende sólo de la memoria de la GPU. También necesitamos RAM, disco y espacio temporal. Cuando esos recursos no se coordinan, preparar y recuperar experimentos exige mucho trabajo manual.

## 18–38 s · Solución

Shadow Trainer organiza la ejecución bajo presupuestos explícitos. Inspecciona el equipo, comprueba si el trabajo cumple sus límites, prepara datos por casos, guarda checkpoints y genera un reporte verificable.

## 38–65 s · Demostración

Aquí mostramos el MVP instalado en una RTX 3050 Ti de cuatro GiB. Una CNN tridimensional sintética completó doce pasos con estrategia síncrona, caché acotada y checkpoint. El entorno tiene treinta y ocho dependencias fijadas, y la suite pasó sesenta y cinco pruebas.

## 65–85 s · Alcance

Son pruebas de ingeniería, no resultados clínicos. La recuperación exacta está comprobada para Tiny3D en CPU. El protocolo largo STU-Net y la validación en otros equipos siguen pendientes. No prometemos aceleración universal.

## 85–110 s · Propuesta

El siguiente paso es probar una carga real de un equipo de ingeniería AI o edge de Jalisco. Queremos comparar recursos y operación con criterios acordados previamente. Buscamos un primer piloto, revisión de propiedad intelectual y validación del modelo comercial.

## Capturas y preparación

- Mostrar slides 1–3; después el resumen público de evidencia y slide 5.
- Si se graba una ejecución, requiere preflight y autorización; este paquete no la inicia.
- No mostrar logs, rutas de usuario, remotes, tokens, datasets, rostros de pacientes ni código.
- No editar una captura para ocultar un fallo o convertir una simulación en ejecución real.
- El reporte completo de ejecución sigue siendo interno; usar la evidencia sanitizada del paquete.
- Equipo, afiliación y contactos deben verificarse antes de grabación/envío.
