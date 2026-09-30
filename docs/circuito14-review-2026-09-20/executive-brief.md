# Shadow Trainer — borrador ejecutivo para Circuito14

Estado: material de trabajo para revisión humana; no enviado ni aprobado para divulgación. No sustituye verificar los requisitos de la convocatoria.

## Propuesta

Runtime local para ejecutar y auditar cargas de IA con presupuestos explícitos de VRAM, RAM y disco. Integra inspección y admisión, staging por casos, comprobación de integridad, caché acotada, checkpoints y reportes.

## Problema y cliente propuestos

Equipos de ingeniería AI/edge y validación que iteran con hardware limitado pueden necesitar reducir preparación manual y recuperar trabajos interrumpidos. Ésta es una hipótesis comercial: todavía faltan entrevistas y una carga piloto industrial.

## Evidencia actual

En esta PC se instaló un entorno independiente, con 38 dependencias fijadas mediante hashes. La suite pasó 65 pruebas sin fallos ni omisiones. La demo sintética Tiny3D completó 12 pasos en una RTX 3050 Ti física de 4 GiB usando sync; generó checkpoint, métricas finitas y reporte. La recuperación exacta de Tiny3D CPU es evidencia separada de la demo GPU nueva.

La evidencia histórica incluye adaptadores NIfTI 3D y ResNet 2.5D proxy; no se mezcla su rendimiento con la demo actual ni se presenta como mejora de segmentación. Stage38 largo sigue pendiente.

## Diferenciación y negocio a validar

La propuesta integra controles de recursos y evidencia en un flujo local pequeño. No se afirma novedad de cada componente ni superioridad medida frente a alternativas. Propuesta comercial inicial: piloto técnico acotado, licencia del runtime y servicios de integración/soporte. No hay precios ni tracción acreditados en este documento.

## Encaje propuesto en Jalisco

Buscar equipos de AI/edge, sistemas embebidos e ingeniería de validación para acordar un piloto medible. No se ha demostrado una aplicación EDA o un piloto en semiconductores. La relación institucional, roles del equipo y apoyos deben documentarse y autorizarse antes de afirmarlos externamente.

## Solicitud

Vinculación para un primer piloto, mentoría comercial y revisión de propiedad intelectual. La postulación puede avanzar con sync; no depende de publicar el artículo ni de cerrar la evaluación clínica.

## Pendientes del expediente

- CV, roles, dedicación y disponibilidad reales del equipo.
- Afiliación y uso autorizado de nombres/logos; sin presumir respaldo institucional.
- Revisión de IP y permisos de divulgación de este paquete.
- Entrevistas y organizaciones objetivo; evidencia de interés sólo si existe.
- Revisión de requisitos/formulario vigente y grabación del video.
- Decisión humana final de postular; sin envío automático.
