# Frontera entre Shadow Trainer y un Decision Runtime

Estado: decisión arquitectónica; no implementación de Jev ni cambio de alcance clínico.

## Separación de responsabilidades

```text
Shadow Trainer
  produce candidato + evidencia reproducible
        ↓
Evaluation Harness
  calcula métricas y comprueba integridad
        ↓
Deterministic Policy Gate
  aplica restricciones duras y elegibilidad
        ↓
Decision Runtime opcional
  recomienda entre acciones permitidas
        ↓
Human approval / external deployment system
```

Shadow Trainer entrena, adapta, ejecuta y reporta bajo presupuestos. El harness
calcula hechos reproducibles. Ningún modelo probabilístico —Jev, OpenJev, un SLM
o un modelo propio— sustituye las métricas, reetiqueta fallos o decide qué cohorte
es válida. El Decision Runtime es un proyecto independiente y un consumidor
opcional de evidencia sanitizada.

## Orden de autoridad

1. Restricciones duras: seguridad, integridad, leakage, cohortes, recursos,
   pruebas obligatorias y autorización humana. Su violación produce rechazo.
2. Reglas deterministas: gates científicos y operativos versionados.
3. Modelo de decisión: sólo si quedan varias acciones elegibles y reversibles.
4. Abstención o escalamiento: `human_review` es una salida de primera clase.

El modelo nunca puede promover cuando un gate duro falla. No tiene acceso directo
a GPU, datos clínicos, credenciales, checkpoints o APIs de despliegue. Recibe un
recibo allowlisted con métricas agregadas, límites, hashes/digests y acciones
predefinidas. Su salida es una recomendación auditable, no una orden ejecutable.

## Contrato conceptual mínimo

Entrada `decision-runtime.request/v1`:

- digest del recibo de ejecución y del protocolo;
- baseline y candidatos mediante identificadores opacos;
- métricas con denominadores, intervalos/ausencias y dirección de preferencia;
- restricciones duras ya evaluadas, incluyendo la evidencia de cada gate;
- acciones permitidas: `keep_baseline`, `request_more_evidence`, `retry`,
  `reject_candidate`, `human_review`; `promote` sólo cuando la política externa
  lo habilite explícitamente;
- contexto operacional mínimo y sanitizado, nunca datos por paciente.

Salida `decision-runtime.recommendation/v1`:

- distribución sobre acciones permitidas y opción elegida;
- abstención/confianza y proveedor utilizado;
- digest exacto de entrada, versión de política/modelo y latencia;
- explicación estructurada limitada a factores del recibo;
- `requires_human_approval=true` para promoción, clínica o alto riesgo.

## Ideas rescatadas para investigación, no prioridad inmediata

- Backend intercambiable: reglas, Jev, implementación abierta, SLM o modelo
  especializado. Compararlos con el mismo contrato.
- Métricas: calidad de decisión, calibración/ECE/Brier, riesgo selectivo,
  abstención, robustez OOD, latencia, memoria, energía/coste y tasa de escalamiento.
- Zona local objetivo hipotética: modelos especializados pequeños y cuantizados,
  pero no se acepta ninguna cifra de VRAM o calidad sin reproducirla.
- Jev se usa primero como benchmark externo. No destilar sus salidas ni hacer
  reverse engineering sin revisar términos/licencia y autorización específica.
- Shadow Trainer podría adaptar en el futuro un modelo de decisión, pero no
  evaluarse o promocionarse a sí mismo mediante ese modelo.

## Aplicación inmediata en este repositorio

El protocolo r2 sigue gobernado por gates deterministas. Un eventual Decision
Runtime sólo podrá consumir el reporte después de completar la cohorte y nunca
resolver una comparación incompleta. No cambia `sync` predeterminado, no abre
test, no cierra Stage38 ni añade una afirmación clínica o comercial.
