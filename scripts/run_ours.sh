#!/bin/bash
#SBATCH --job-name ours
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pascal:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output /home/rb080/assignments/Project/Repositories/medseg/scripts/outs/ours.out


nvidia-smi

module load anaconda3/2022.05
module load cuda/11.6

source activate medseg

python run_trainings.py --config-name segmentation_cfg run_name=ours_ref_disc dataset=refuge model_type=ours refuge_args.mask_type=disc train_settings.epochs=50
python run_trainings.py --config-name segmentation_cfg run_name=ours_ref_cup dataset=refuge model_type=ours refuge_args.mask_type=cup train_settings.epochs=50
python run_trainings.py --config-name segmentation_cfg run_name=ours_isic dataset=isic model_type=ours train_settings.epochs=50

python eval.py --config-name segmentation_cfg run_name=ours_ref_disc dataset=refuge model_type=ours refuge_args.mask_type=disc
python eval.py --config-name segmentation_cfg run_name=ours_ref_cup dataset=refuge model_type=ours refuge_args.mask_type=cup
python eval.py --config-name segmentation_cfg run_name=ours_isic dataset=isic model_type=ours