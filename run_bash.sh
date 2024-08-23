#!/bin/zsh
#SBATCH --job-name=Gempy3_HSI

### File / path where STDOUT will be written, the %J is the job id

#SBATCH --output=./Results_test/Salinas_%J.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters

#SBATCH --time=0:30:00

### Request all CPUs on one node
#SBATCH --nodes=1

### Request number of CPUs
#SBATCH --ntasks=32

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2

### Specify your mail address
#SBATCH --mail-user=ravi@aices.rwth-aachen.de
### Send a mail when job is done
#SBATCH --mail-type=END

### Request memory you need for your job in MB
#SBATCH --mem-per-cpu=4096M

source /home/jt925938/.bashrc
conda activate gempy_pyro
python run.py