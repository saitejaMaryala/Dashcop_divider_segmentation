#!/bin/bash
#SBATCH --job-name=divider_seg_old_annots_109
#SBATCH -A mobility_arfs
#SBATCH -p ihub
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnode056
#SBATCH --cpus-per-gpu=9
#SBATCH --mem-per-cpu=10000
#SBATCH --time=4-00:00:00
#SBATCH --output=/ssd_scratch/cvit/saiteja/output_segmentation_terminal.txt
#SBATCH --mail-user=saiteja.maryala@research.iiit.ac.in
#SBATCH --mail-type=ALL

# Load modules
# module load u18/cuda/10.2
# module load u18/cudnn/7.6.5-cuda-10.2


source /home2/saiteja3000/miniconda3/bin/activate
conda activate spenv

cd /home2/saiteja3000/Dashcop_wsd/divider_seg
# python generate_labels_images.py
python generate_labels_images_parallel.py

# cd /home2/saiteja3000/Dashcop_wsd/divider_seg
# python train.py

