#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=16GB
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --job-name=pmv264-proj
#SBATCH --output=pmv264-proj.out
#SBATCH --requeue

module purge

/share/apps/images/run-cuda-12.2.2.bash bash -c '
# conda activate py311
source .venv/bin/activate

python --version

./uv pip install -r requirements.txt

echo "running tests..."
uv python -m src.test_runner
echo "tests complete."
'
