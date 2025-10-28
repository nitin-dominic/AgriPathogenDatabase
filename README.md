# Crop Disease Foundation Models: Technical Articles, Codes, and Resources

A single, well-documented repository that collects papers, datasets, code, pretrained models, simulation/digital-twin environments, baseline notebooks, evaluation scripts, and demo UIs — all specifically for crop disease detection, monitoring, and management (text + vision + multimodal + decision-making). Curated resources, code, and benchmarks for foundation-model-enabled crop disease detection, monitoring, and management.

## What this repo contains
- `papers/` — annotated list of recent studies (LLMs, VLMs, multimodal) focused on crop diseases.
- `datasets/` — links + preprocessing scripts for PlantVillage, in-field datasets, hyperspectral disease sets.
- `models/` — baseline models (classification, detection, VLMs) and pointers to weights.
- `sim/` — digital twin environments and sim-to-real experiment configs.
- `experiments/` — RL/adaptive-learning recipes and evaluation scripts.
- `notebooks/` — walkthroughs: detection → explanation → prescription report → Q&A.
  
# Table of Contents

- [Introduction](#introduction)
# Table of Contents
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
  - [2021](#2021)
    - [VLM-focused Studies](#vlm-focused-studies-2021)
  - [2018](#2018)
    - [LLM-focused Studies](#llm-focused-studies-2018)
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

## 2024

### LLM-focused Studies 2024

| Approach / Models | Model Type | Open-access source | Use Cases | Journal | Reference |
|------------------|-----------|-----------------|-----------|---------|-----------|
| Used LLM with Agricultural Knowledge Graphs (KGs), Graph Neural Networks (GNNs) | LLM | Not specified | Plant disease diagnosis, reasoning over symptoms, linking textual disease corpora with structured knowledge | MDPI *Agriculture* | [Zhao2024](https://www.mdpi.com/2077-0472/14/8/1359) |
| GPT-4 (OpenAI API) for automated literature synthesis on pest controllers | LLM | Proprietary (OpenAI) | Automating systematic reviews, reducing expert workload | *Methods in Ecology and Evolution* | [Scheepens2024](https://www.biorxiv.org/content/10.1101/2024.01.12.575330v1) |
| GPT-based models for sensor + text queries | LLM | Proprietary (OpenAI) | Query-based plant health monitoring, e.g., explaining yellowing leaves using sensors + LLM reasoning | *Int. J. Computer Applications in Technology* | [Ahir2024](https://www.inderscienceonline.com/doi/abs/10.1504/IJCAT.2024.146146) |
| Q&A systems using GPT-4 + knowledge graphs | LLM | --- | Plant disease diagnosis, pest identification, Q&A for sustainable management | *Resources, Conservation and Recycling* | [Yang2024](https://www.sciencedirect.com/science/article/pii/S0921344924000910) |
| GlyReShot (Chinese agricultural NER, few-shot + GROM module) | LLM | --- | Recognizing entities (disease, crop, pest, drug) in Chinese agricultural text | *Heliyon* | [Liu2024](https://www.sciencedirect.com/science/article/pii/S2405844024081246) |
| RAG chatbot with hybrid DeiT + VGG16 | VLM | Not explicitly available | Medicinal plant identification + bilingual insights using images + RAG | *Telematics and Informatics* | [Paneru2024](https://www.sciencedirect.com/science/article/pii/S2772503024000677) |
| Agricultural Knowledge Graph (AGKG) + LLMs | LLM | Not specified | Entity retrieval and Q&A via domain-specific AGKG | *Displays* | [Wang2024](https://www.mdpi.com/2079-9292/13/11/2209) |

### VLM-focused Studies 2024

| Models | Type | Open-access source | Use Cases | Journal | Reference |
|--------|------|-----------------|-----------|---------|-----------|
| DINOv2 | Vision model | [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/dinov2) | Self-supervised feature extraction, clustering of disease symptoms | ScienceDirect | [Bai2024](#) |
| BLIP / BLIP-2 | Multi-modal model | [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/blip-2) | Image captioning and visual reasoning for disease explanation | - | [Liang2024](#) |
| LLaVA | Multi-modal model | [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/llava) | Multi-modal reasoning for plant disease recognition | - | [Xu2025](#) |
| SAM | VLM | [GitHub](https://github.com/facebookresearch/segment-anything) | Wheat disease diagnosis through reasoning | ScienceDirect | [Zhang2024](#) |
| ViT + GPT-2 | VLM | [OpenAI](https://openai.com/index/gpt-2-1-5b-release/) / [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/vit) | Align plant disease phenotypes with trait descriptions | *Plant Phenomics* | [Zhao2024](#) |
| Inception-v4 + LSTM | VLM | - | Align crop disease images with question embeddings | *Plant Phenomics* | [Zhao2024](#) |
| PDC-VLD | Multi-modal (vision + text) | - | Tomato leaf disease detection with unseen class generalization | *Plant Phenomics* | [Li2024](#) |
| FHTW-Net | Vision-language model | [GitHub](https://github.com/ZhouGuoXiong/FHTW-Net) | Retrieve matching text from a query image (and vice versa) for rice leaf disease descriptions | *Plant Phenomics* | [Zhou2024](#) |
| ILCD | Multi-modal visual question answering | [GitHub](https://github.com/SdustZYP/ILCD-master/tree/main) | Addressed complex questions about crop disease stages and attributes | *Plant Phenomics* | [Zhao2024](#) |
| PhenoTrait (GPT-4 / GPT-4o) | Multi-modal (image-to-text) | [PlanText](https://plantext.samlab.cn/) | Generates plant disease descriptions from images | *Plant Phenomics* | [Zhao2024](#) |
| PepperNet | Multi-modal VLM | - | Detecting pepper diseases and pests using natural language descriptions | *Nature Scientific Reports* | [Liu2024](#) |
| Qwen-VL | Pre-trained VLM | [Google Drive](https://drive.google.com/drive/folders/1sl-nRDYGz9T4969QjS_nRnoZIfcREWOW) | Generate text descriptions for disease images as prompt for classifiers | MDPI *Sensors* | [Zhou2024](#) |
| Segment Anything Model (SAM) | Image segmentation | [SAM-Meta AI](https://segment-anything.com/) | Identifies and segments diseased regions in leaves | IEEE *Access* | [Moupojou2024](#) |
| Visual Answer Model (VQA) | Multi-modal VQA | - | Answer questions about fruit tree diseases using images + Q&A knowledge | *Frontiers in Plant Science* | [Lan2023](#) |


## 2023

### LLM-focused Studies 2023

| Approach / Models | Model Type | Open-access source | Use Cases | Journal | Reference |
|------------------|-----------|-----------------|-----------|---------|-----------|
| GPT-3.5 for agricultural extension services | LLM | Proprietary (OpenAI) | Farmer advisory chatbots, pest/disease diagnosis, local language support | *Nature Food* | [Tzachor2023](#) |
| ChatAgri (ChatGPT-based agricultural text classification) | LLM | [GitHub link](https://github.com/albert-jin/agricultural_textual_classification_ChatGPT) | Cross-lingual agricultural news classification, few-shot and prompt-based learning | *Neurocomputing* | [Zhao2023](#) |

### VLM-focused Studies 2023

| Models | Type | Open-access source | Use Cases | Journal | Reference |
|--------|------|-----------------|-----------|---------|-----------|
| ITLMLP | Vision-language pre-training | - | Few-shot cucumber disease recognition using image, text, and label information | *Computers and Electronics in Agriculture* | [Cao2023](#) |
| YOLO + GPT | Multi-modal model | [GitHub](https://github.com/ultralytics/ultralytics) / [OpenAI](https://platform.openai.com/docs/models) | Generate agricultural diagnostic reports with deep logical reasoning | *Computers and Electronics in Agriculture* | [Qing2023](#) |
| ITF-WPI | Cross-modal feature fusion | [GitHub](https://github.com/wemindful/Cross-modal-pest-Identifying) | Wolfberry pest identification using image + text | *Computers and Electronics in Agriculture* | [Dai2023](#) |
| Neuro-symbolic AI | Deep learning + knowledge graph | [GitHub](https://github.com/Research-Tek/xai-cassava-agriculture/tree/master) | Improves prediction accuracy and explains results for farmers | *Expert Systems with Applications* | [Chhetri2023](#) |
| ShuffleNetV2 + TextCNN | Multi-modal model | - | Extract textual features and semantic relationships from descriptive text | *Nature Scientific Reports* | [Qiu2023](#) |
| MMFGT | Multi-modal transformer | - | Few-shot pest recognition combining image + text information | MDPI *Electronics* | [Zhang2023](#) |
| ODP-Transformer | Multi-modal image-to-text + classification | - | Two-stage pest classification + caption generation | *Computers and Electronics in Agriculture* | [Wang2023](#) |

## 2022

### LLM-focused Studies 2022
| Approach / Models | Model Type | Open-access source | Use Cases | Journal | Reference |
|------------------|-----------|-----------------|-----------|---------|-----------|
| AgriBERT (BERT-based, knowledge-infused with FoodOn/Wikidata) | LLM | --- | Semantic matching of food descriptions, cuisine classification, agricultural NLP tasks | IJCAI 2022 | [Rezayi2022](#) |


## 2021

### VLM-focused Studies 2021
| Models | Type | Open-access source | Use Cases | Journal | Reference |
|--------|------|-----------------|-----------|---------|-----------|
| ITK-Net (Image-Text-Knowledge Network) | Multi-modal | - | Identify invasive diseases in tomato/cucumber using image + text + domain knowledge | *Computers and Electronics in Agriculture* | [Zhou2021](#) |

## 2018

### LLM-focused Studies 2018
<!-- Insert LLM table for 2018 here -->

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

## How to Contribute

If you have suggestions, come across missed papers, or find useful resources, we welcome your contributions via **pull requests**.

### Guidelines:

1. Use the following Markdown format when adding a new paper:

2. For preprints with multiple versions, use the **earliest submitted year**.

3. Display papers in **descending chronological order** (latest first).

4. Indicate clearly whether the study is **LLM-focused** or **VLM-focused**.

5. For code or datasets, provide a direct link and a short description.

### Example:

*Zhao, J., Li, H., and Wang, P.* **GlyReShot: Few-shot Chinese Agricultural NER.** <ins>Heliyon</ins> 2024. [[PDF](https://doi.org/xxx)]; [[GitHub](https://github.com/xyz)].

## Cite
If you use this repo, please cite: `@misc{...}`

Contributions welcome — see `CONTRIBUTING.md`.
