#!/bin/bash
#SBATCH --job-name=roberta_reg
#SBATCH -p reserved
#SBATCH --reservation=ling282_2025
#SBATCH -A ling_282_482_2025
#SBATCH -c 9
#SBATCH --mem=44g
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH -o gen.out
#SBATCH -e gen.err
#SBATCH --output=logs/roberta_reg_%j.out
#SBATCH --error=logs/roberta_reg_%j.err

echo "JOB STARTED at $(date)"

# ---------------------------
# 1. Load modules
# ---------------------------
module load anaconda3
module load cuda/11.7

# ---------------------------
# 2. Activate env
# ---------------------------
eval "$(conda shell.bash hook)"
echo "conda initialized"
conda activate /scratch/ling_282_482_2025/env
echo "environment activated"

# ---------------------------
# 3. Move into script folder
# ---------------------------
cd /home/m1/CEO-ideology/scripts

# ---------------------------
# 4. Run training
# ---------------------------
python RoBERTa_training.py

echo "JOB FINISHED at $(date)"
