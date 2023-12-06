#!/bin/bash
#SBATCH --job-name baselines_ref_disc
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pascal:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output /home/rb080/assignments/Project/Repositories/medseg/scripts/outs/baselines_ref_disc.out


nvidia-smi

module load anaconda3/2022.05
module load cuda/11.6

source activate medseg


python -u run_trainings.py --config-name segmentation_cfg run_name=smp_unet_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
python -u run_trainings.py --config-name segmentation_cfg run_name=smp_upp_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
python -u run_trainings.py --config-name segmentation_cfg run_name=smp_fpn_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
python -u run_trainings.py --config-name segmentation_cfg run_name=smp_psp_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
python -u run_trainings.py --config-name segmentation_cfg run_name=smp_man_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
python -u run_trainings.py --config-name segmentation_cfg run_name=smp_link_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc


python -u eval.py --config-name segmentation_cfg run_name=smp_unet_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
python -u eval.py --config-name segmentation_cfg run_name=smp_upp_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
python -u eval.py --config-name segmentation_cfg run_name=smp_fpn_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
python -u eval.py --config-name segmentation_cfg run_name=smp_psp_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
python -u eval.py --config-name segmentation_cfg run_name=smp_man_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
python -u eval.py --config-name segmentation_cfg run_name=smp_link_ref_disc dataset=refuge model_type=smp_unet refuge_args.mask_type=disc
