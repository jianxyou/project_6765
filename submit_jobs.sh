#!/bin/bash
#
# Submit a single job that launches parallel workers (one per GPU).
# Auto-detects available GPU count.
#
# Usage: bash submit_jobs.sh

SRC="/home/eoijuya/thinclient_drives/C:/Users/eoijuya/work/project_6765"
WORKDIR="/active_work/environment/project_6765"
VENV="/active_work/environment/.venv"
OUTDIR="/active_work/environment/benchmark_outputs"

mkdir -p "$OUTDIR/logs"

# Sync code to HPC local disk BEFORE submitting (while thin client is connected)
echo "Syncing code to HPC disk..."
rsync -a --delete --exclude='.git' --exclude='__pycache__' --exclude='.claude' --exclude='outputs' "${SRC}/" "${WORKDIR}/"
if [ $? -ne 0 ]; then
    echo "ERROR: rsync failed. Is thin client connected?"
    exit 1
fi
echo "Sync done."

sbatch <<'OUTER_EOF'
#!/bin/bash
#SBATCH --job-name=benchmark_sweep
#SBATCH --output=/active_work/environment/benchmark_outputs/logs/sweep_%j.log
#SBATCH --error=/active_work/environment/benchmark_outputs/logs/sweep_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Everything below runs from HPC local disk — no thin client dependency
WORKDIR="/active_work/environment/project_6765"
VENV="/active_work/environment/.venv"
OUTDIR="/active_work/environment/benchmark_outputs"

export HF_HOME="/active_work/environment/.cache/huggingface"
export TRANSFORMERS_CACHE="/active_work/environment/.cache/huggingface"

source ${VENV}/bin/activate
cd "${WORKDIR}"

echo "=== Benchmark Sweep Started at $(date) ==="
echo "Python: $(which python)"
echo "HF_HOME: $HF_HOME"
echo "WORKDIR: $WORKDIR"
nvidia-smi
echo ""

# Auto-detect number of GPUs
N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$N_GPUS" -eq 0 ]; then
    N_GPUS=1
fi
echo "Detected $N_GPUS GPUs"

# Launch workers in parallel, one per GPU
for GPU_ID in $(seq 0 $((N_GPUS-1))); do
    echo "Launching worker for GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python run_all_experiments.py $GPU_ID $N_GPUS \
        > ${OUTDIR}/logs/worker_gpu${GPU_ID}.log 2>&1 &
done

echo "All $N_GPUS workers launched. Waiting..."
wait
echo "=== All workers finished at $(date) ==="

# Merge results
python -c "
import json
from pathlib import Path
outdir = Path('${OUTDIR}')
all_results = []
for f in sorted(outdir.glob('all_results_gpu*.json')):
    all_results.extend(json.loads(f.read_text()))
# Filter out errors
valid = [m for m in all_results if 'error' not in m]
errors = [m for m in all_results if 'error' in m]
merged = outdir / 'all_results_merged.json'
merged.write_text(json.dumps(valid, indent=2, ensure_ascii=False))
print(f'Merged {len(valid)} valid results ({len(errors)} errors) -> {merged}')
"

echo "=== Done ==="
OUTER_EOF

echo "Job submitted. Check status: squeue -u $USER"
echo "Logs: $OUTDIR/logs/"
echo "Results: $OUTDIR/all_results_merged.json"
