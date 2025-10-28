# Crop Disease Foundation Models: A Set of Technical Articles

A single, well-documented repository that collects papers, datasets, code, pretrained models, simulation/digital-twin environments, baseline notebooks, evaluation scripts, and demo UIs — all specifically for crop disease detection, monitoring, and management (text + vision + multimodal + decision-making).

# Crop-Disease-Foundation-Models

Curated resources, code, and benchmarks for foundation-model-enabled crop disease detection, monitoring, and management.

## What this repo contains
- `papers/` — annotated list of recent studies (LLMs, VLMs, multimodal) focused on crop diseases.
- `datasets/` — links + preprocessing scripts for PlantVillage, in-field datasets, hyperspectral disease sets.
- `models/` — baseline models (classification, detection, VLMs) and pointers to weights.
- `sim/` — digital twin environments and sim-to-real experiment configs.
- `experiments/` — RL/adaptive-learning recipes and evaluation scripts.
- `notebooks/` — walkthroughs: detection → explanation → prescription report → Q&A.

# Crop Disease Management: Curated Papers

## Table of Contents
- [Introduction](#introduction)
- [Curated Papers by Year](#curated-papers-by-year)
  - [2024](#2024)
  - [2023](#2023)
  - [2022](#2022)
- [Datasets](#datasets)
- [Models & Benchmarks](#models--benchmarks)
- [Simulation & Digital Twin Studies](#simulation--digital-twin-studies)
- [Human-in-the-Loop Approaches](#human-in-the-loop-approaches)
- [Experiments / Workflows](#experiments--workflows)
- [References](#references)

---

## Introduction
This repository curates recent studies on crop disease detection, monitoring, and management, with a focus on:
- Vision-Language Models (VLMs), Large Language Models (LLMs), and multimodal approaches  
- Early disease detection and targeted interventions  
- Human-in-the-loop collaboration for validation and adaptive learning  
- Simulation and digital twin environments for model testing and sim-to-real transfer  

---

## Curated Papers by Year

### 2024

| Title | Authors | Approach | Dataset | Key Finding |
|-------|--------|---------|--------|------------|
| Example Paper 1 | A. Author et al. | VLM + LLM | PlantVillage | Early detection improved by 15% |
| Example Paper 2 | B. Author et al. | RL + Digital Twin | In-field images | Targeted spraying policy optimized in sim |

### 2023

| Title | Authors | Approach | Dataset | Key Finding |
|-------|--------|---------|--------|------------|
| Example Paper 1 | C. Author et al. | Multimodal fusion | Hyperspectral + RGB | Multimodal approach improved classification accuracy |
| Example Paper 2 | D. Author et al. | LLM for prescription | Yield logs | Automatic recommendation matched expert decisions 85% of the time |

### 2022

| Title | Authors | Approach | Dataset | Key Finding |
|-------|--------|---------|--------|------------|
| Example Paper 1 | E. Author et al. | CNN detection | PlantVillage | Disease detection accuracy 92% |
| Example Paper 2 | F. Author et al. | Human-in-the-loop | Field trials | Annotation feedback reduced false positives by 20% |

---

## Datasets
- **PlantVillage**: Lab images of diseased crops  
- **In-field disease datasets**: Multi-season field imagery  
- **Hyperspectral datasets**: Spectral bands for early detection  

---

## Models & Benchmarks
- Classification: EfficientNetV2, ResNet  
- Detection/Segmentation: Faster R-CNN, YOLOX, U-Net  
- Multimodal / VLMs: CLIP, BLIP2, vision-language pipelines  
- RL / adaptive learning setups for targeted interventions  

---

## Simulation & Digital Twin Studies
- Virtual plant canopies with disease progression  
- Targeted spraying simulation with reward optimization  
- Sim-to-real transfer experiments with domain randomization  

---

## Human-in-the-Loop Approaches
- Shared autonomy for disease management  
- Validation of uncertain cases by humans  
- Annotation loops for continual model improvement  

---

## Experiments / Workflows
- Prescription report generation from visual and textual data  
- Q&A and extension service platforms  
- Grad-CAM explainability for disease detection  

---

## References
- Add full citations here (BibTeX or numbered list)  
s-by-year)

## Quick start
1. Clone: `git clone https://github.com/<you>/Crop-Disease-Foundation-Models`
2. Read `datasets/readme.md` for dataset setup.
3. Launch demo: `python demos/gradio_demo.py`

## Cite
If you use this repo, please cite: `@misc{...}`

Contributions welcome — see `CONTRIBUTING.md`.
