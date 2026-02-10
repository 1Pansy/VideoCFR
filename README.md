Consensus Frame GRPO: Brain-Inspired Unlabeled Alignment for Guiding Visual Focus
<div align="center">
<!-- 替换为你的 Arxiv 链接，如果没有先写 Coming Soon -->
![alt text](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)
<!-- 替换为 HuggingFace 链接 -->
![alt text](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)

![alt text](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

![alt text](https://img.shields.io/badge/Code-Coming%20Soon-green)
Enhancing Trustworthy Reasoning in Video Multimodal LLMs via Process Supervision
</div>
📢 News
[2026-02-10] 🚀 This repository is created. We are currently organizing the code and will release it very soon! Please Star ⭐ this repo to stay tuned.
[2026-02] The paper is currently under review.
💡 Abstract
Recent advancements in reinforcement learning have enhanced reasoning in large language models, yet applying this to video multimodal models is challenged by the multi-dimensional redundancy of raw video streams. Existing methods often rely on outcome-based supervision, risking exploitation of language biases or temporal shortcuts without true visual grounding.
We introduce Consensus Frame GRPO (CF-GRPO), a label-free process reward framework. It aligns the model's attention with the video's intrinsic structure—fusing physical motion, scene transitions, and semantic relevance via multi-source priors. At its core, the Consensus Frame Reward (CFR) mechanism employs max pooling for sparse feature aggregation and low-temperature sharpening to boost signal-to-noise ratio in policy gradients.
Experiments show CFR achieves state-of-the-art performance on video reasoning benchmarks (e.g., VSI-Bench, VideoMMMU) and significantly enhances interpretability through visualized attention distributions.
<div align="center">
<img src="assets/teaser.png" width="80%" alt="Teaser Image (Please upload your teaser image to an 'assets' folder)">
<br>
<em>Figure 1: Visualization of our Consensus Frame Reward (CFR) mechanism.</em>
</div>
🗓️ Roadmap
We are working hard to prepare the codebase for open-source release. The planned release schedule is as follows:

📄 ArXiv Paper Release

🧩 Inference Code: Evaluation scripts for VSI-Bench, MMVU, and VideoMME.

🎨 Visualization Tools: Scripts for generating Temporal Attention (
α
α
) curves and Spatial Heatmaps.

🏋️ Training Code: Full GRPO training implementation with DeepSpeed ZeRO-3 support.

📦 Data Preprocessing: Scripts to generate Consensus Priors (CLIP + Scene Cut).

🤗 Model Weights: Release Video-CFR-7B checkpoints on HuggingFace.
🔍 Methodology Highlights
Our method introduces Consensus Frame Reward (CFR) to guide the RL training process:
Hierarchical Prior: Fusing Uniform, Scene, and Semantic signals to construct a robust "Ground Truth" attention distribution without human annotation.
Sparse Aggregation: Using Max Pooling to capture fine-grained visual cues that are often diluted by Mean Pooling.
Signal Sharpening: Applying a low temperature (
τ
=
0.1
τ=0.1
) to generate high-SNR gradients for policy optimization.
Soft-Gated Integration: Conditioning the visual reward on reasoning correctness to prevent reward hacking.
📈 Performance
Our Video-CFR-7B achieves consistent improvements over baselines:
Model	VSI-Bench	VideoMMMU	MMVU (MC)	VideoMME
Video-R1-7B	30.3	47.2	63.5	54.3
Video-CFR-7B (Ours)	32.6	49.3	67.0	56.1
🛠️ Usage (Coming Soon)
We will provide detailed instructions on how to set up the environment and run the code.
code
Bash
# Example placeholder command
git clone https://github.com/YourUsername/Consensus-Frame-GRPO.git
cd Consensus-Frame-GRPO
pip install -r requirements.txt
🖊️ Citation
If you find our work helpful, please consider citing:
code
Bibtex
@article{VideoCFR2026,
  title={Consensus Frame GRPO: Brain-Inspired Unlabeled Alignment for Guiding Visual Focus and Enhancing Trustworthy Reasoning in Video Multimodal LLMs},
  author={Anonymous Authors},
  journal={arXiv preprint arXiv:202X.XXXXX},
  year={2026}
}
📧 Contact
If you have any questions, please feel free to open an issue or contact [Your Email/Twitter].
