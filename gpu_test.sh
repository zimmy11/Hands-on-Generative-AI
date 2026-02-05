#!/bin/bash
#SBATCH --job-name=celeba_test
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

srun --cpu-bind=cores --hint=nomultithread python -u test.py \
  --config-path="experiments/base_config.yaml" \
  --checkpoint-path="checkpoints/weights/last-v6.ckpt" \
  --data-path="./data" \
  --batch-size=16 \
  --num-samples=10000 \
  --sde-type="ve" \
  --probability-flow \
  --nll-num-images=1000
