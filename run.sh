#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:40:00
#SBATCH --mem=1GB
#SBATCH --mail-user=u1469481@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_1-%j
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nlp_mp1

OUT_DIR=/scratch/general/vast/u1469481/cs6957/assignment1/models
python main.py --output_dir ${OUT_DIR}
