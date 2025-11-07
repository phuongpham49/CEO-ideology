# 🧠 Measuring Corporate Ideology through Lobbying Language

**Author:** Phuong Pham  
**Affiliation:** University of Rochester  
**Repository:** [https://github.com/phuongpham49/CEO-ideology](https://github.com/phuongpham49/CEO-ideology)  
**Date:** October 2025  

---

## 📘 Overview

This project explores whether firms’ political orientations—revealed through their lobbying and campaign-finance activities—are systematically reflected in the language of their lobbying disclosures.  
Building on In Song Kim’s *Political Cleavages of Firms* framework, the study uses the **LobbyView** database ([https://lobbyview.org/](https://lobbyview.org/)), which contains all U.S. federal lobbying reports and PAC contribution data since 1999.  

Each lobbying report includes a *free-text issue description* in which firms describe the congressional bills, public laws, or regulatory topics they lobbied. These descriptions provide a rich corpus of political and economic framing language.  
The project fine-tunes a **RoBERTa-base transformer model** to predict firm-level ideological scores (based on PAC donations) from these lobbying texts. The aim is to determine whether and how corporate ideology can be inferred directly from lobbying language.

---

## 🎯 Research Question

> **Can firms’ ideological orientations be inferred from the language used in their lobbying disclosures?**

Specifically:
- Do Republican-leaning and Democratic-leaning firms use systematically different linguistic frames in lobbying text?  
- Can those differences be identified and quantified using transformer-based language models?

---

## ⚙️ Methodology Overview

1. **Data Acquisition**
   - Source: **LobbyView database** ([https://lobbyview.org/](https://lobbyview.org/))
   - Datasets used:
     - `report_level.csv`
     - `issue_level.csv`
     - `issue_text.csv` (contains the free-text field of lobbying issue descriptions)
   - Firm ideology scores are computed from PAC donation data:  
     \[
     \text{Ideology Score} = \frac{\text{Republican Donations}}{\text{Republican + Democratic Donations}}
     \]

2. **Data Cleaning and Preparation**
   - Merge issue-level text with report-level metadata and firm identifiers.
   - Remove duplicates and boilerplate language (e.g., “Lobbying on behalf of…”).
   - Tokenize and segment long lobbying texts for transformer input.
   - Match each text entry with firm-level ideology labels.

3. **Model Training**
   - Fine-tune **RoBERTa-base** using Hugging Face `transformers`.
   - Treat ideology prediction as a **regression task** (continuous scores).
   - Train and evaluate on the **BlueHive HPC cluster** at the University of Rochester.
   - Apply firm-level holdouts and entity masking to prevent memorization.

4. **Evaluation and Interpretation**
   - Evaluate predictive performance using Pearson correlation and MSE.
   - Apply **SHAP** and **Integrated Gradients** to identify key ideological terms.
   - Visualize framing differences across major policy domains (trade, environment, taxation, etc.).

---

## 🧱 Repository Structure (Planned)

obbyview_ideology_project/
│
├── data/
│ ├── raw/ # Original LobbyView CSVs (report, issue, text)
│ ├── processed/ # Cleaned, tokenized, and merged datasets
│ └── metadata/ # Firm identifiers and ideology scores
│
├── notebooks/
│ ├── 01_data_cleaning.ipynb # Text cleaning and preprocessing
│ ├── 02_merge_datasets.ipynb # Merge text and ideology data
│ ├── 03_finetune_roberta.ipynb # Model fine-tuning and evaluation
│ └── 04_interpretability.ipynb # SHAP/IG analysis and visualizations
│
├── src/
│ ├── data_preprocessing.py # Text and metadata cleaning functions
│ ├── model_training.py # RoBERTa fine-tuning pipeline
│ ├── interpretability.py # SHAP & Integrated Gradients scripts
│ └── utils.py # Logging and helper functions
│
├── slurm_scripts/
│ ├── finetune_roberta.slurm # BlueHive GPU training job script
│ └── preprocess_data.slurm # Data preparation batch job
│
├── results/
│ ├── figures/ # Plots and interpretability visualizations
│ ├── tables/ # Model metrics and summary results
│ └── model_checkpoints/ # Fine-tuned model weights
│
├── README.md # Project overview and structure (this file)
└── requirements.txt # Python dependencies

