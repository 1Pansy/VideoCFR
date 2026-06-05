# Third-Party Components

VideoCFR combines project-specific code with several open-source components. This file records the main bundled or referenced components so that downstream users can check the corresponding licenses and upstream documentation.

## Bundled Components

| Path | Source | Notes |
| --- | --- | --- |
| `CFR/r1-v/` | R1-V-style video RL training code | Adapted for VideoCFR training and CFR reward computation. The bundled subtree contains its own Apache-2.0 license text. |
| `CFR/qwen-vl-utils/` | Qwen-VL utilities | Local utility package for processing Qwen-style image/video inputs. The package metadata declares Apache-2.0. |

## Referenced Models and Libraries

- Qwen2.5-VL-compatible checkpoints are expected for inference and training.
- vLLM is used for high-throughput generation.
- DeepSpeed is used for distributed training.
- TRL-style GRPO training utilities are used as part of the RL workflow.

Please follow the license terms of each upstream model, dataset, and library. The Apache-2.0 license in this repository applies to the code released here and does not override third-party model or dataset terms.
