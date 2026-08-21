# *SIGNPOST-Bench*: Benchmarking Text–Vision Conflict Resolution in Multimodal Large Language Models

[📖 ArXiv](https://arxiv.org/abs/2608.04244) | [🐙 GitHub](https://github.com/inorganicwriter/SIGNPOST-Bench) | [🤗 Dataset](https://huggingface.co/datasets/inorganicwriter/SIGNPOST-Bench)

This repository contains the benchmark construction, evaluation, and analysis code for the paper *"SIGNPOST-Bench: Benchmarking Text–Vision Conflict Resolution in Multimodal Large Language Models"*.

---

## 🔔 Introduction

Multimodal large language models (MLLMs) make grounded predictions in real-world scenes by combining visual and textual cues, yet existing benchmarks rarely reveal how they arbitrate between these evidence sources when they conflict. **SIGNPOST-Bench** (Scene Image Geo-localization with Noisy Perturbation on Observed Sign Text) is a controlled counterfactual benchmark for evaluating text–vision conflict resolution.

Each source image is transformed into a **counterfactual quintuplet** of **Original**, **Blank**, **Similar**, **Random**, and **Adversarial** variants through synthetic, localized scene-text interventions designed to preserve non-textual content:

| Variant | Operation |
|---|---|
| `Original` | Unmodified source image |
| `Blank` | Removes the selected scene-text spans (text-ablated reference) |
| `Similar` | Replaces text with alternatives compatible with the ground-truth geographic context or language |
| `Random` | Introduces unrelated readable text without a designated geographic target |
| `Adversarial` | Injects a geographically conflicting cue; when geocodable, defines an injected target |

SIGNPOST-Bench contains **5,111 counterfactual groups** and **25,555 image variants** (10,084 scene-text spans) from four datasets (IM2GPS3K, YFCC4K, Google Street View, Baidu Street View). We evaluate **20 MLLMs from seven providers** on all five variants of all groups, yielding 511,100 model–image evaluations under a shared prompt.

---

## 📊 Key Findings

**Text–vision conflict substantially degrades localization.** Adversarial edits reduce WLA for every model and dataset (79 of 80 model–dataset cells). Mean WLA falls from 47.11 to 29.89 (a 36.6% relative drop), and median error grows 4.8× (282 km to 1,347 km). Acc@25/Acc@200/Acc@750 drop from 34.9%/50.1%/71.6% to 20.6%/31.8%/50.1%, and errors exceeding 2,500 km rise from 11.7% to 31.7%.

**Compatible, unrelated, and conflicting text produce distinct failure patterns.**
- *Semantic effects*: native scene text is generally useful (Original WLA exceeds Blank by 9.40 points on average). Relative to Blank, Similar replacements reduce error by 379 km on average, whereas Random and Adversarial replacements increase it by 959 km and 1,577 km.
- *Vulnerability grows with scene-text coupling*: proportional degradation rises from 25.0% (T1 Portable) to 36.4% (T2 Cultural) to 42.2% (T3 Geo-Specific); T3 images suffer the largest adversarial drop (25.14 points).
- *Geographic conflicts induce target-directed shifts*: among the 1,732 geocodable groups (33.9%), TFR ranges from 6.5% to 20.1% across models, and every model shows a positive mean paired Trap Distance Reduction (343–1,926 km). Adversarial TBS and TFR are strongly correlated (Spearman ρ = 0.836, p < 0.001).

**Capability and conflict robustness are distinct.** Gemini-3-Flash, Gemini-3.1-Pro, and Gemini-2.5-Pro lead the MCRS ranking (72.70, 72.23, 69.80); Claude-Haiku-4.5 and the two Moonshot-Vision models score lowest. Clean-input capability does not determine conflict robustness: Qwen3-VL-30B ranks fourth in robustness (R = 80.05) despite modest capability (C = 36.55), whereas Seed-2.0-Pro has higher capability (C = 55.01) but lower robustness (R = 71.21). Rankings remain stable under weight/exponent variations (minimum Kendall τ = 0.905).

**Conflict awareness does not reliably prevent prediction failure.** In a two-model probing analysis, Gemini-2.5-Flash reaches probing WLA of 38.28 but detects the conflict in only 14.2% of adversarial samples; GPT-4o-mini detects 32.8% but achieves only 15.10 WLA. A defense prompt raises Gemini-2.5-Flash's detection to 49.0% but *lowers* GPT-4o-mini's to 19.3%; neither model improves both detection and localization under the defense instruction.

---

## 🏆 Main Results

MCRS = 100 × C<sup>0.40</sup> × R<sup>0.60</sup>, where C is the model's Capability Score (average of Original/Blank WLA) and R is the Conflict Robustness Score (WLA retention under Random/Adversarial text plus normalized TBS/TFR penalties). Both are reported as percentages; higher is better. See the paper for the full five-condition WLA table, per-dataset and per-tier results, and probing/defense analyses.

| Rank | Model | MCRS | C (%) | R (%) |
|---|---|---|---|---|
| 1 | Gemini-3-Flash | 72.70 | 56.58 | 85.93 |
| 2 | Gemini-3.1-Pro | 72.23 | 57.08 | 84.49 |
| 3 | Gemini-2.5-Pro | 69.80 | 51.86 | 85.10 |
| 4 | Gemini-2.5-Flash | 64.31 | 48.36 | 77.77 |
| 5 | GPT-5 | 64.28 | 50.70 | 75.30 |
| 6 | Seed-2.0-Pro | 64.23 | 55.01 | 71.21 |
| 7 | Claude-Opus-4.6 | 62.56 | 45.95 | 76.84 |
| 8 | Claude-Sonnet-4.6 | 62.08 | 47.41 | 74.30 |
| 9 | Kimi-K2.5 | 61.74 | 50.52 | 70.57 |
| 10 | Seed-2.0-Lite | 61.16 | 52.78 | 67.47 |
| 11 | Qwen3-VL-30B | 58.50 | 36.55 | 80.05 |
| 12 | GPT-4o | 57.16 | 42.40 | 69.74 |
| 13 | Qwen3-VL-Plus | 56.67 | 38.13 | 73.80 |
| 14 | GPT-5.4 | 54.77 | 36.14 | 72.27 |
| 15 | Qwen3-VL-235B | 53.36 | 38.64 | 66.16 |
| 16 | GPT-4o-mini | 50.68 | 31.89 | 69.02 |
| 17 | Grok-4 | 44.05 | 32.62 | 53.82 |
| 18 | Moonshot-32K-Vision | 41.47 | 25.59 | 57.22 |
| 19 | Moonshot-128K-Vision | 41.46 | 25.77 | 56.92 |
| 20 | Claude-Haiku-4.5 | 38.87 | 24.10 | 53.46 |

*Values match the paper's Table 1. Bold marks the best value per metric in the paper; here ranks are by MCRS.*

---

## 🔢 Dataset Statistics

SIGNPOST-Bench contains **5,111 counterfactual groups**, each with one quintuplet, for **25,555 images** and **10,084 scene-text spans**:

| Dataset | Groups | Source |
|---|---|---|
| IM2GPS3K | 651 | Flickr (geotagged) |
| YFCC4K | 992 | Yahoo Flickr Creative Commons |
| GoogleSV | 2,337 | Google Street View |
| BaiduSV | 1,131 | Baidu Street View |
| **Total** | **5,111** | - |

Each group consists of an `Original` source image and four generated variants (`Blank`/`Similar`/`Random`/`Adversarial`), all sharing the same ground truth and selected scene-text spans.

Scene text is categorized by geographic identifiability into three tiers:

| Tier | Meaning | Groups |
|---|---|---|
| T1 Portable | little geographic specificity (global brands, generic warnings) | 347 (6.8%) |
| T2 Cultural | narrows to a language or cultural region without identifying a place | 3,851 (75.3%) |
| T3 Geo-Specific | directly identifies a place or distinctive local entity | 913 (17.9%) |

Quality assurance: two annotators independently assigned tiers to 350 stratified images (83.4% agreement, κ = 0.747), and 120 generated images were audited for text naturalness (4.00 ± 1.26), artifact severity (1.32 ± 0.78), and context damage (1.14 ± 0.52), with 87.5% of rendered text fully readable.

---

## ⚙️ Installation

```bash
git clone https://github.com/inorganicwriter/SIGNPOST-Bench.git
cd SIGNPOST-Bench

conda create -n signpost python=3.10 -y
conda activate signpost

pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env to fill in SPONSOR_API_BASE and SPONSOR_API_KEY
```

The dataset (attack texts, metadata, ground-truth labels, taxonomy, and human annotations) is released on Hugging Face. Download it and point `SIGNPOST_DATA_ROOT` at the extracted `Data/` folder, or keep the default layout with a sibling `Data/` directory.

---

## 🧠 Evaluation

All models receive the same prompt requesting direct coordinate estimates without chain-of-thought, and metrics are averaged equally across the four datasets.

Evaluate a single model on one dataset and variant:

```bash
python evaluate.py --dataset im2gps3k --variant Adversarial --model gemini-2.5-flash
```

Run the full benchmark (all datasets, all variants, all 20 registered models):

```bash
python evaluate.py --dataset all --variant all --model all
```

Results are written to `Data/<dataset>/results/<model>/results_<Variant>_<Model>.jsonl`.

Compute metrics and the MCRS leaderboard:

```bash
python -m analysis.compute_results --datasets im2gps3k yfcc4k googlesv baidusv
python -m analysis.compute_tfr --datasets im2gps3k yfcc4k googlesv baidusv
python -m evaluation.mcrs --out-dir results
```

Run the test suite:

```bash
pytest
```

37 tests covering metric calculators, evaluation CLI, TFR computation, and analysis scripts.

---

## 📁 Project Structure

```
SIGNPOST-Bench/
├── config.py                    # Centralized config (paths, API keys, datasets)
├── evaluate.py                  # Exp 1: Standard geo-localization evaluation
├── evaluate_probing.py          # Exp 2/3: Conflict probing & defense evaluation
├── evaluate_generalization.py   # Exp 4: Cross-task generalization evaluation
├── run_pipeline.py              # Full pipeline orchestrator (data gen -> eval)
├── evaluation/                  # API clients, metric calculators, MCRS core
├── analysis/                    # Metric computation, statistical tests, human agreement
├── data_collector/              # Attack generation, ComfyUI image synthesis, filtering
├── utils/                       # Shared helpers
└── tests/                       # Unit tests
```

---

## 📜 License

**Code**: released under the [MIT License](LICENSE).

**Dataset**: the benchmark content (attack texts, taxonomy labels, human annotations, metadata) is released for research use. The underlying images originate from third-party sources (Flickr/IM2GPS, Google Street View, Baidu Street View) and remain subject to their respective terms and licenses; users are responsible for compliance when redistributing imagery.

---

## 📚 Citation

If you use SIGNPOST-Bench in your research, please cite:

```bibtex
@article{li2026signpost,
  title={SIGNPOST-Bench: Benchmarking Text--Vision Conflict Resolution in Multimodal Large Language Models},
  author={Li, Sirun and Liu, Minghao and Dai, Ling and Li, Yong and Lyu, Haoxin and Zhou, Junting and Zhang, Fan},
  journal={arXiv preprint arXiv:2608.04244},
  year={2026},
  url={https://arxiv.org/abs/2608.04244}
}
```

---

## 📧 Contact

Sirun Li (corresponding author): sirun_li@stu.pku.edu.cn
