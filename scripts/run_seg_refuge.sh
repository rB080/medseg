#!/bin/bash
#SBATCH --job-name baselines_1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pascal:1 --exclude=c1-2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output /home/rb080/assignments/Project/Repositories/medseg/scripts/outs/baselines_1.out


nvidia-smi

module load anaconda3/2022.05
module load cuda/11.6

source activate medseg

python run_trainings.py --config-name segmentation_cfg run_name=unet_ref
python run_trainings.py --config-name segmentation_cfg run_name=smp_unet_ref model_type=smp_unet
python run_trainings.py --config-name segmentation_cfg run_name=smp_upp_ref model_type=smp_unet
python run_trainings.py --config-name segmentation_cfg run_name=smp_fpn_ref model_type=smp_unet
python run_trainings.py --config-name segmentation_cfg run_name=smp_psp_ref model_type=smp_unet
python run_trainings.py --config-name segmentation_cfg run_name=smp_man_ref model_type=smp_unet
python run_trainings.py --config-name segmentation_cfg run_name=smp_link_ref model_type=smp_unet