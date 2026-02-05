#!/bin/bash
#SBATCH --job-name=celeba_train
#SBATCH --account=3155287
#SBATCH --partition=dsba
#SBATCH --nodes=1
#SBATCH --gres=gpu:nv-2080:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=20:09:00
#SBATCH --out=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=3155287@studbocconi.it

source activate hands_on_genai
cd ~/Hands-on-Generative-AI

# CPU threads mainly for dataloader / preprocessing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun --cpu-bind=cores --hint=nomultithread python -u train.py \
  --data-path="./data" \
  --epochs=50 \
  --num-workers=0 \
  --sde-type="vp" \
  --likelihood-weighting=False \
  --use-importance-sampling=False
