# Exact paired timing summary

timing summaries require execution_seconds, exact audit, and complete protocol coverage

| Workload | Cache | Exact pairs | Eligible | Sync median (s) | Prefetch median (s) | Ratio | 95% bootstrap CI |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| nifti3d | cold | 3 | True | 4.9572 | 3.4889 | 1.4209 | [1.2842, 1.4739] |
| nifti3d | warm | 3 | True | 1.9927 | 1.9662 | 0.9088 | [0.7092, 1.4159] |
| resnet18 | cold | 3 | True | 1.3779 | 1.3266 | 1.1486 | [1.0290, 1.2190] |
| resnet18 | warm | 3 | True | 1.0712 | 1.0346 | 1.0561 | [0.8197, 1.1574] |
| resnet50 | cold | 3 | True | 1.1650 | 1.1400 | 1.0674 | [0.8399, 1.0913] |
| resnet50 | warm | 3 | True | 1.2010 | 1.1540 | 1.0660 | [0.9347, 1.1121] |
| tiny3d | cold | 3 | True | 0.8011 | 0.7918 | 0.9903 | [0.9770, 1.6516] |
| tiny3d | warm | 3 | True | 0.6950 | 0.7195 | 1.0222 | [0.9658, 1.2628] |
