#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
conda activate gfnet
cd ~/PatternAnalysis-2026/recognition/GFNet_s4802604
python train.py --model baseline --epochs 15
python train.py --model baseline --epochs 15 --leaky
