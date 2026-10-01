# Demo pública para Circuito14

Publicada el 30 de septiembre de 2026 (hora de Ciudad de México):

https://ia-local-circuito14-demo.raulavenger21.chatgpt.site

## Qué presenta

La página permite recorrer 33 eventos guardados de una demo Tiny3D del
20 de septiembre: doce actualizaciones, seis eventos de checkpoint y una
caché de 4 KiB para seis muestras sintéticas de 1 KiB. La gráfica, la pérdida
y la ocupación vienen de los registros originales. La reproducción ajusta
el ritmo de presentación; no ejecuta entrenamiento en el navegador.

La sección volumétrica informa por separado cuatro trabajos STU-Net-S con
64/96 casos y dos semillas, de 1,536 actualizaciones cada uno. No presenta
el consumo de Tiny3D como el consumo de STU-Net. No afirma eficacia clínica,
ahorro comercial, equivalencia larga de prefetch ni garantía universal de
recuperación. El modo predeterminado es sync.

El JSON público contiene únicamente campos autorizados, muestras sintéticas,
resultados agregados y hashes de procedencia. No incluye imágenes médicas,
filas de pacientes, checkpoints, rutas del equipo, domicilio ni credenciales.

## Guion de presentación · aproximadamente 90 segundos

1. Abrir la demo en pantalla completa y mostrar el resultado registrado.
   «Estamos construyendo una herramienta que organiza entrenamientos cuando
   memoria y disco son limitados. Este ejemplo pequeño permite ver el
   mecanismo sin depender de descargar datos durante la presentación».
2. Pulsar **Reproducir recorrido**.
   «Los datos entran por partes. Hay seis muestras, pero sólo cuatro lugares
   en la caché. Al necesitar una nueva muestra, se libera espacio para ella».
3. Mostrar la gráfica y pausar cerca de un checkpoint.
   «El modelo registra su error en cada actualización y guarda avances. Aquí
   vemos una ejecución real ya registrada con datos sintéticos. La página
   reproduce sus eventos; no está entrenando un modelo por Internet».
4. Desplazarse a la evidencia volumétrica.
   «En el laboratorio también completamos cuatro entrenamientos STU-Net-S
   de 1,536 actualizaciones cada uno en una GPU física de 4 GiB. Usamos 64 y
   96 casos y dos semillas. Hubo una parada por recursos y reanudación».
5. Cerrar con el siguiente hito.
   «Ahora buscamos validar una carga concreta de ingeniería AI/edge con una
   organización de alta tecnología: comparar recursos, trabajo manual y
   recuperación contra su proceso habitual».

Si preguntan por calidad: la curva es de una red sintética; la evidencia de
STU-Net acredita factibilidad de ejecución. La calidad clínica y el valor
industrial requieren evaluaciones distintas y no se dan por demostrados.

## Copia sin Internet

La copia para presentación está en la carpeta de tesis:
`Convocatorias/Circuito14_2026-09-29/03_demo_publica/`.
Abrir `index.html` en Chrome o Edge. Mantener junto a él `style.css`,
`app.js`, `evidence.js` y `evidencia-publica.json`. La descarga pública
también está disponible como archivo local. No requiere instalar paquetes.

## Procedencia y comprobaciones

La exportación verificó éxito, orden 1–12, pérdidas finitas, doce
actualizaciones, seis checkpoints, caché dentro de 4096 bytes y hardware
de 4 GiB. Los cuatro trabajos de STU-Net se contrastaron contra la auditoría
agregada de dos semillas. Los cronómetros históricos tienen varios alcances;
la demo omite afirmaciones de tiempo o aceleración.

Las comprobaciones de navegador pasaron en escritorio 1440×1000 y móvil
390×844: curva, navegación de la línea de tiempo, reinicio, reproducción,
pausa, siguiente evento, caché, descarga JSON y ausencia de errores JavaScript.
Se conservaron capturas locales. El despliegue devolvió `succeeded` y el
acceso se configuró como público a petición del responsable.

El código de la página se mantiene en su propio checkout Sites, separado del
repositorio privado del runtime. Esta publicación muestra la interfaz y el
resumen sanitizado; no publica el código de entrenamiento del producto.
