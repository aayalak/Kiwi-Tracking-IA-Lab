# Especificación resumida

## Objetivo

Diseñar un sistema para la gestión y monitoreo de producción de kiwi, integrando recorridos de campo, capturas de imágenes, procesamiento con modelos de IA, trazabilidad de resultados y reportes para la toma de decisiones.

## Componentes funcionales

### Gestión agronómica

- Cooperativa
- Campos
- Lotes
- Calles
- Plantas
- Frutos
- Campañas productivas

### Recorridos

El sistema contempla recorridos planificados y ejecutados sobre lotes y calles, registrando:

- fecha y hora;
- operador;
- vehículo;
- sensor/cámara;
- lote y calle recorrida;
- tipo de captura;
- observaciones;
- incidencias.

### Capturas de imágenes

Cada imagen capturada en campo queda asociada a metadata:

- fecha y hora;
- ubicación GPS;
- lote, calle o planta relacionada;
- operador;
- vehículo;
- sensor utilizado.

### Procesamiento con IA

Las imágenes pueden procesarse mediante modelos de detección para identificar:

- ramas;
- yemas;
- brotes;
- flores;
- hojas;
- frutos;
- anomalías en frutos;
- anomalías en hojas;
- plagas.

Cada detección registra coordenadas, nivel de confianza, imagen asociada, fecha, ubicación y recorrido.

## Etapas previstas

### Etapa 1: Dataset

Recolección, organización y validación de imágenes para entrenamiento.

### Etapa 2: Entrenamiento

Entrenamiento inicial con modelos YOLO y documentación de métricas.

### Etapa 3: MVP

API en Python/FastAPI para carga de imágenes, procesamiento, almacenamiento y consulta de resultados.

### Etapa 4: Evolución

Migración a PostgreSQL/PostGIS, mapas, dashboards, reportes e históricos.

### Etapa 5: Escalado

Despliegue futuro en nube o servidores propios, almacenamiento escalable y soporte multiusuario.
