# Prior-art and differentiation matrix

This is an engineering literature map, not a novelty opinion or patent search.
All claims must be checked against the primary paper before submission.

| System | Main constrained resource | Mechanism | Recovery/data contract | Relationship to Shadow Trainer |
| --- | --- | --- | --- | --- |
| ZeRO-Offload (USENIX ATC 2021) | GPU model-state memory | Moves optimizer state and computation to CPU. | Model-training oriented; remote case cache is not its central contract. | Complementary model-state backend; substantially stronger model-scale evidence. |
| CheckFreq (FAST 2021) | Lost work and checkpoint overhead | Adaptive fine-grained, pipelined checkpoints and resumable data iterator. | Strong checkpoint-frequency and data-use semantics. | Direct recovery prior art; Shadow Trainer adds resource admission and source/cache integrity but currently has a simpler fixed boundary. |
| MONAI Cache/SmartCache | Medical preprocessing and repeated data access | Domain-specific cached datasets and transformations. | Medical-imaging ecosystem rather than a cross-resource admission contract. | Executable framework-local baseline for the NIfTI experiments. |
| Out-of-core adaptive scheduling (2020) | GPU memory for model variables | Schedules GPU/CPU variable movement. | Primarily tensor/model-state locality. | Complementary to dataset and job-level orchestration. |
| STEER (EuroMLSys 2026) | Unified memory in edge retraining | Streams optimizer state from fast storage during backpropagation. | Targets optimizer-state persistence and transfer. | Closest recent edge-runtime neighbor; Shadow Trainer must distinguish its multi-resource admission, whole-case remote staging, evidence classes, and recovery bundle. |
| Shadow Trainer v0.1 | GPU, RAM, swap, disk, cache, and source locality | Pre-admission, atomic staging, bounded LRU, durable windows, evidence compiler. | Model + optimizer + RNG + dataset position, with integrity and environment artifacts. | Integration hypothesis; comparative novelty and performance remain to be established. |

## Primary sources

- ZeRO-Offload: https://www.usenix.org/conference/atc21/presentation/ren-jie
- CheckFreq: https://www.usenix.org/conference/fast21/presentation/mohan
- MONAI paper: https://arxiv.org/abs/2211.02701
- Out-of-core training: https://arxiv.org/abs/2010.14109
- STEER listing and abstract: https://euromlsys.eu/

## Novelty test before submission

The defensible research contribution cannot be “training with little VRAM” or
“using a cache.” It must establish that the unified contract changes what can
be executed or audited compared with individual mechanisms. At minimum:

1. show a workload that baseline local loading cannot admit under the same disk
   ceiling, while Shadow Trainer completes without exceeding it;
2. quantify runtime overhead relative to framework-local loading;
3. prove recovery state equivalence, not only step counts;
4. demonstrate at least one controlled remote-source fault;
5. report prefetch only if the paired exactness gate passes.
