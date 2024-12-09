#!/usr/bin/zsh

### Maximum runtime
#SBATCH --time=00:15:00

#SBATCH -J gpu_serial
#SBATCH -o gpu_serial.%J.log
#SBATCH --gres=gpu:1

module load CUDA

# Print some debug information
echo; export; echo; nvidia-smi; echo