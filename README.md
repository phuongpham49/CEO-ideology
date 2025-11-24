# 🧠 Measuring Corporate Ideology through Lobbying Language

**Author:** Phuong Pham  
**Affiliation:** University of Rochester  
**Repository:** [https://github.com/phuongpham49/CEO-ideology](https://github.com/phuongpham49/CEO-ideology)  
**Date:** November 2025  

---

## 📌 Project Overview

This project investigates whether **corporate ideology** can be inferred from **firm-generated language**, using a **regression-fine-tuned RoBERTa transformer model**.

Because real CEO interview data from TDM ProQuest became inaccessible, I created **synthetic but realistic firm interview data** and paired it with **synthetic PAC contribution data**. This allows a fully reproducible pipeline while preserving the methodological goal of the original project.

The project answers the following question:

> **Can transformer models identify ideological signals in firm communication, using PAC donation patterns as a numerical target?**

---

## 🗂️ Dataset Description

### 1. `fake_firm_interviews.csv`
Synthetic dataset containing ~5000 firm interviews.  
Each row includes:
- `interview_id`
- `filename`
- `company`
- `industry`
- `date`
- `length_type`
- `title`
- `text` (medium/long interview-style content)

### 2. `fake_firm_interviews_cleaned.csv`
Cleaned version of the text data

### 2. `synthetic_pac_contributions.csv`
Synthetic PAC donation dataset with:
- `company`
- `pac_total`
- `pac_to_dems`
- `pac_to_reps`
- `ideology_score` (scaled from −1 conservative → +1 liberal)

### 3. `training_data.csv`
Merged dataset combining interview text + ideology score for training.

---

## 🧪 Methods Summary

### **1. Data Generation**
Scripts:
- `synthetic_data.py`
- `synthetic_PAC.py`

These create reproducible synthetic datasets tied to 60 fixed U.S. firms across multiple industries.

### **2. Text Cleaning**
`clean_text.py`  
Removes noise, boilerplate, formatting artifacts, and normalizes text.

### **3. Merging Data**
`merge_interview_pac.py`  
Per-company merge to align text with ideology labels.

### **4. Model Training (RoBERTa Fine-Tuning)**
`train_roberta_regression.py`

- Uses **HuggingFace Transformers**
- Converts ideology to a **continuous regression target**
- Applies train/validation split
- Evaluates loss, RMSE, and correlation
- GPU-compatible (BlueHive cluster or local CUDA)

### **5. Model Interpretability**
`integrated_gradients.py`

- Uses **Captum** to compute token-level attribution
- Identifies which words/sentences signal ideological leaning
- Produces interpretable heatmaps

---

project-root/
│
├── data/
│   ├── fake_firm_interviews.csv
│   ├── fake_pac_data.csv
│   ├── fake_firm_interviews_cleaned.csv
│   └── training_data.csv
│
├── scripts/
│   ├── generate_synthetic_interviews.py
│   ├── generate_synthetic_pac.py
│   ├── merge_interview_pac.py
│   ├── text_cleaning.py
│   ├── train_roberta_regression.py
│   └── integrated_gradients.py
│
├── README.md
├── requirements.txt
├── Progress_Report.pdf
└── Project_Plan.tex


