#!/bin/bash
#SBATCH --job-name baselines_2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pascal:1 --exclude=c1-2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output /home/rb080/assignments/Project/Repositories/medseg/scripts/outs/baselines_2.out


nvidia-smi

module load anaconda3/2022.05
module load cuda/11.6

source activate medseg

python run_trainings.py --config-name segmentation_cfg run_name=unet_isic
python run_trainings.py --config-name segmentation_cfg run_name=smp_unet_isic model_type=smp_unet
python run_trainings.py --config-name segmentation_cfg run_name=smp_upp_isic model_type=smp_unet
python run_trainings.py --config-name segmentation_cfg run_name=smp_fpn_isic model_type=smp_unet
python run_trainings.py --config-name segmentation_cfg run_name=smp_psp_isic model_type=smp_unet
python run_trainings.py --config-name segmentation_cfg run_name=smp_man_isic model_type=smp_unet
python run_trainings.py --config-name segmentation_cfg run_name=smp_link_isic model_type=smp_unet