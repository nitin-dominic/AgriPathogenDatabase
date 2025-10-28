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
  
# Table of Contents

- [Introduction](#introduction)
- [Curated Papers by Year](#curated-papers-by-year)
  - [2024](#2024)
    - [LLM-focused Studies](#llm-focused-studies-2024)
    - [VLM-focused Studies](#vlm-focused-studies-2024)
  - [2023](#2023)
    - [LLM-focused Studies](#llm-focused-studies-2023)
    - [VLM-focused Studies](#vlm-focused-studies-2023)
  - [2022](#2022)
    - [LLM-focused Studies](#llm-focused-studies-2022)
    - [VLM-focused Studies](#vlm-focused-studies-2022)
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

## Summary of Recent Studies: LLMs for Crop Disease / Agricultural Text

### 2024

| Approach / Models | Model Type | Open-access source | Use Cases | Journal | Reference |
|------------------|-----------|-----------------|-----------|---------|-----------|
| Used LLM with Agricultural Knowledge Graphs (KGs), Graph Neural Networks (GNNs) | LLM | Not specified | Plant disease diagnosis, reasoning over symptoms, linking textual disease corpora with structured knowledge | MDPI *Agriculture* | [Zhao2024](#) |
| GPT-4 (OpenAI API) for automated literature synthesis on pest controllers | LLM | Proprietary (OpenAI) | Automating systematic reviews, reducing expert workload | *Methods in Ecology and Evolution* | [Scheepens2024](#) |
| GPT-based models for sensor + text queries | LLM | Proprietary (OpenAI) | Query-based plant health monitoring, e.g., explaining yellowing leaves using sensors + LLM reasoning | *Int. J. Computer Applications in Technology* | [Ahir2024](#) |
| Q&A systems using GPT-4 + knowledge graphs | LLM | --- | Plant disease diagnosis, pest identification, Q&A for sustainable management | *Resources, Conservation and Recycling* | [Yang2024](#) |
| GlyReShot (Chinese agricultural NER, few-shot + GROM module) | LLM | --- | Recognizing entities (disease, crop, pest, drug) in Chinese agricultural text | *Heliyon* | [Liu2024](#) |
| RAG chatbot with hybrid DeiT + VGG16 | VLM | Not explicitly available | Medicinal plant identification + bilingual insights using images + RAG | *Telematics and Informatics* | [Paneru2024](#) |
| Agricultural Knowledge Graph (AGKG) + LLMs | LLM | Not specified | Entity retrieval and Q&A via domain-specific AGKG | *Displays* | [Wang2024](#) |

### 2023

| Approach / Models | Model Type | Open-access source | Use Cases | Journal | Reference |
|------------------|-----------|-----------------|-----------|---------|-----------|
| GPT-3.5 for agricultural extension services | LLM | Proprietary (OpenAI) | Farmer advisory chatbots, pest/disease diagnosis, local language support | *Nature Food* | [Tzachor2023](#) |
| ChatAgri (ChatGPT-based agricultural text classification) | LLM | [GitHub link](https://github.com/albert-jin/agricultural_textual_classification_ChatGPT) | Cross-lingual agricultural news classification, few-shot and prompt-based learning | *Neurocomputing* | [Zhao2023](#) |

### 2022

| Approach / Models | Model Type | Open-access source | Use Cases | Journal | Reference |
|------------------|-----------|-----------------|-----------|---------|-----------|
| AgriBERT (BERT-based, knowledge-infused with FoodOn/Wikidata) | LLM | --- | Semantic matching of food descriptions, cuisine classification, agricultural NLP tasks | IJCAI 2022 | [Rezayi2022](#) |

### 2018

| Approach / Models | Model Type | Open-access source | Use Cases | Journal | Reference |
|------------------|-----------|-----------------|-----------|---------|-----------|
| Original GPT pre-training paper (OpenAI) | LLM | Not open-source initially | Laid foundation for later agricultural LLM applications | OpenAI | [Radford2018](#) |

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
