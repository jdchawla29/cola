module load cuda-12.4 

export venv_folder_name=cola-env
export cache_folder_name=cola-env-cache

virtualvenv3.12 /scratch/$venv_folder_name/
source /scratch/$venv_folder_name/bin/activate

export PIP_CACHE_DIR=/scratch/$cache_folder_name

pip install -r requirements.txt
pip install . --no-build-isolation