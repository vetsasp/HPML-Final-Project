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
source .venv/bin/activate

python --version

./uv sync

echo "running benchmark..."
./uv run python -m src.test_runner

echo "running quality evaluation..."
./uv run python -m src.eval_quality

echo "plotting results..."
./uv run python -m src.plot_results

echo "all runs complete."
'
