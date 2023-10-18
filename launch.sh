#!/bin/sh
#SBATCH -J dde_solver
#SBATCH --gres=gpu:1
#SBATCH --partition=tau
#SBATCH --nodes=1
#SBATCH -t 5-0
#SBATCH --output=out%j.out              # nom du fichier de sortie
#SBATCH --error=out%j.err               # nom du fichier d'erreur (ici commun avec la sortie)
set -x

nvidia-smi

python_filename="brusselator.py"
directory_name_py="$(cut -d'.' -f1 <<<'brusselator')"
now=$(date +"%m_%d_%Y_%M:%S")

echo "Will execute file : $python_filename"

if [ -z "$1" ]
then
    default_dir_dde=meta_data/$directory_name_py/$now
else
    default_dir_dde=meta_data/$directory_name_py/$1/$now
fi 

echo "Creating directory for experiment $default_dir_dde"

mkdir -p $default_dir_dde


export python_filename
export directory_name_py
export default_dir_dde
#srun python optimal_delays.py --exp_path=sinus
#srun python ks.py --exp_path=ks_4_features_only
python brusselator.py --delays=5
#srun python vdp.py --exp_path=vdp