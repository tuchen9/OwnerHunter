# OwnerHunter

Official implementation and reproducibility package for:

> **OwnerHunter: Multilingual Website Owner Identification Powered by Large Language Models (LLMs)**  

OwnerHunter identifies website owners using webpage text and multimodal cues. It combines **LLM-based owner recognize** and **rule-guided hybrid ranking** to enhance reliability.

This repository provides:

✅ All prompts  
✅ In-context example pool generator  
✅ Preprocessing and post-processing scripts  
✅ Reproducible computation settings (seeds, hardware, decoding configs)

---

## 📁 Repository Structure

OwnerHunter/

├─ preprocessing.py # Randomly split data (train/dev/test)

├─ example.py # Example pool generator

├─ multi_aug.py # Multimodal augmentation: domain, logo cues

├─ api_owner.py # Candidate owner recognition and self-verification via LLM API

├─ prompt_generator.py # Prompt templates used for LLM inference

├─ hybird_ranking.py # Entity disambiguation & score aggregation

├─ evaluation.py # Compute P/R/F1 metrics

├─ tools.py # Utility functions

├─ example.py # Example pool generator

├─ FastChat-main/

│ └─ fastchat/src_all/

│ ├─ owner_reco.py # Candidate owner recognition via LLM

│ ├─ self_veri.py # LLM self-verification: hallucination reduction

│ └─ prompt_generator.py # Prompt templates used for LLM inference

└─ data/ # Dataset directories (user-provided)

---

## ✅ Install environment

```bash
pip install -r requirements.txt
```

---

## 🧠 Pipeline Overview

OwnerHunter includes **four sequential stages**:

1️⃣ Multimodal augmentation (`multi_aug.py`)  
2️⃣ Candidate owner recognition (`owner_reco.py`)  
3️⃣ Self-verification (`self_veri.py`)  
4️⃣ Disambiguation + Hybrid ranking (`hybird_ranking.py`)  

Evaluation performed by `evaluation.py`.

---

## 🔧 End-to-End Reproduction Example
Below shows how to reproduce the complete OwnerHunter pipeline on WOI-cn with Qwen1.5-14B:

### Stage 1: Multimodal augmentation
python multi_aug.py \
    --input_path ./data/woi_cn/test \
    --output_path ./data/woi_cn/ownerhunter_qwen/aug

### Stage 2: Candidate owner recognition
cd ./FastChat-main
python -m fastchat.src_all.owner_reco \
    --model ../Qwen-main/Qwen/Qwen1.5-14B-Chat \
    --input_path ../data/woi_cn/test \
    --output_path ../data/woi_cn/ownerhunter_qwen14b \
    --aug_path ../data/woi_cn/ownerhunter_qwen/aug \
    --mode aug-example \
    --num-gpus 2
    --k 2
cd ../

### Stage 3: Processing before self-verification
python hybird_ranking.py \
    --raw_path ./data/woi_cn/test \
    --res_path ./data/woi_cn/ownerhunter_qwen14b/results \
    --aug_path ./data/woi_cn/ownerhunter_qwen/aug \
    --verified False

### Stage 4: Self-verification
cd ./FastChat-main
python -m fastchat.src_all.self_veri \
    --model ../Qwen-main/Qwen/Qwen1.5-14B-Chat \
    --input_path ../data/woi_cn/ownerhunter_qwen14b/dealed \
    --aug_path ../data/woi_cn/ownerhunter_qwen/aug \
    --num-gpus 2
cd ../

### Final: Hybrid ranking with verification
python hybird_ranking.py \
    --raw_path ./data/woi_cn/test \
    --res_path ./data/woi_cn/ownerhunter_qwen14b/results \
    --aug_path ./data/woi_cn/ownerhunter_qwen/aug \
    --verified True

### Evaluation
python evaluation.py \
    --res_path ./data/woi_cn/ownerhunter_qwen14b/ranking

---

## 🔑 Experimental Settings
| Component                | Setting                                            |
| -------------------------| -------------------------------------------------- |
| Seeds                    | `{12, 34, 73, 147, 161}`                           |
| Decoding Setup           | `temperature=0.01, top_p=1.0`                      |
| In-context examples K    | `2`                                                |
| Weights α, β             | `(0.9, 0.1)`                                       |
| Thres_b, Thres_c         | `(0.1, 0.9)`                                       |

---

## 📦 Reproducibility Notes

- All scripts are **inference-only**
- Each stage produces deterministic results under fixed seeds
