#!/bin/bash
#SBATCH -J opt_delay
#SBATCH --gres=gpu:1
#SBATCH --partition=best_effort
#SBATCH --nodes=1
#SBATCH -t 5-0
#SBATCH --array=0-2%2
#SBATCH --output=out%A_%a.out
#SBATCH --error=out%A_%a.err
set -x

nvidia-smi

python_filename="brusselator.py"
directory_name_py="$(cut -d'.' -f1 <<< $python_filename)"
now=$(date +"%m_%d_%Y_%M:%S")

echo "Will execute file : $python_filename"
# $SLURM_ARRAY_TASK_ID
if [ -z "$1" ]
then
    default_dir=meta_data/$directory_name_py/$now
    default_dir+=$SLURM_ARRAY_TASK_ID

else
    default_dir=meta_data/$directory_name_py/$1/$now
    default_dir+=$SLURM_ARRAY_TASK_ID
fi 

echo "Creating directory for experiment $default_dir"

mkdir -p $default_dir


export python_filename
export default_dir

#srun python optimal_delays.py --exp_path=sinus
#srun python ks.py --exp_path=ks_4_features_only
python brusselator.py --delays=2

# Moving out files into meta_data folder
mv *.out $default_dir
mv *.err $default_dir

#srun python vdp.py --exp_path=vdp