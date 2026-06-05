# Data and Model Files

This repository is designed to keep the git checkout lightweight. Large files should be stored outside git and placed under the local paths expected by the scripts.

## Not Included in Git

The following files are intentionally excluded:

- VideoCFR model checkpoints and optimizer states;
- SFT checkpoints;
- raw image and video training data;
- benchmark media files;
- generated rollouts, training logs, and full evaluation outputs.

## Expected Local Layout

Training and evaluation scripts run from `CFR/r1-v/` and expect data under:

```text
CFR/r1-v/
+-- Video-R1-data/
|   +-- Video-R1-COT-165k.json
|   +-- Video-R1-260k.json
|   +-- ...
+-- Evaluation/
|   +-- eval_mvbench.json
|   +-- eval_tempcompass.json
|   +-- eval_videomme.json
|   +-- eval_videommmu.json
|   +-- eval_vsibench.json
|   +-- eval_mmvu.json
+-- eval_results/
```

The JSON prompt files should contain relative media paths that resolve against `CFR/r1-v/Video-R1-data/` or `CFR/r1-v/`.

## Training Data Sources

The manuscript uses a cold-start SFT stage followed by CF-GRPO training. The training data includes:

- Video-R1 CoT-style supervised data for SFT;
- video question-answer samples from MSRVTT-QA, MSVD-QA, and ActivityNet-QA;
- synthetic temporal reasoning queries used for video RL training;
- image QA samples used to preserve fine-grained visual recognition during video RL training.

These datasets are not redistributed in this repository. Download or prepare them from their original sources and follow the corresponding licenses and terms of use.

## Evaluation Sources

The reported evaluation covers:

- VSI-Bench;
- VideoMMMU;
- MMVU;
- MVBench;
- TempCompass;
- VideoMME without subtitles.

Benchmark media and annotations should be obtained from the original benchmark releases. The evaluation script expects converted prompt JSON files under `CFR/r1-v/Evaluation/`.

## Model Sources

The scripts are written for Qwen2.5-VL-compatible checkpoints. Typical local paths include:

```text
/path/to/Qwen2.5-VL-7B-Instruct
/path/to/sft/checkpoint
/path/to/VideoCFR/checkpoint
```

The VideoCFR checkpoint is not included in the current repository snapshot. The README will be updated when a public checkpoint is released.

## Copying Files From a Private Machine

If you keep the large files on another machine, copy only the needed data or checkpoint directories into the layout above. For example:

```bash
rsync -avP liuchengwen@172.18.18.2:/path/to/Video-R1-data/ CFR/r1-v/Video-R1-data/
rsync -avP liuchengwen@172.18.18.2:/path/to/checkpoint/ /local/path/to/VideoCFR/checkpoint/
```

Replace the remote paths with the actual locations on your machine. Avoid committing copied data, checkpoints, or logs to git.
