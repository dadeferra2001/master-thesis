#!/usr/bin/env bash
#
# Parallel runner for the 2x2 SUMO-RL thesis matrix.
#
#   4 algorithms x 3 intensities x 2 variants (vehicle / peds) = 24 training jobs
#
# Each cell runs as its own OS process. This is REQUIRED: the code patches
# sumo_rl to libsumo at module scope, and libsumo allows only one simulation
# per process. Do not try to run cells inside one process / a thread pool.
#
# Usage:
#   ./run_all.sh              # run every phase in order
#   ./run_all.sh setup        # build network + generate routes (once)
#   ./run_all.sh train        # parallel training only
#   ./run_all.sh baseline     # parallel SUMO baseline eval
#   ./run_all.sh eval         # parallel checkpoint eval
#   ./run_all.sh compare      # build comparison dashboards
#
# Tuning (env vars):
#   THREADS=2  JOBS=24  SEEDS="0 1 2"  ./run_all.sh train
#
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the project root (scripts/ lives under it)

# ---------------------------------------------------------------- config ----
ALGOS=(centralized_ppo independent_ppo shared_ppo mappo)
INTENSITIES=(low medium high)
read -r -a SEEDS <<< "${SEEDS:-0}"          # default one seed -> 24 train jobs

# Threads PER process. Keep small; each run is one libsumo + small MLP.
export OMP_NUM_THREADS="${THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Concurrent processes. Rule of thumb: JOBS * THREADS ~= number of vCPUs.
JOBS="${JOBS:-$(( $(nproc) / OMP_NUM_THREADS ))}"
[ "$JOBS" -lt 1 ] && JOBS=1

mkdir -p logs

# --joblog + --resume means re-running a phase skips already-finished jobs,
# which is what you want on a spot/preemptible instance.
run_parallel () {  # $1 = jobfile  $2 = joblog
  parallel --jobs "$JOBS" --joblog "$2" --resume --bar < "$1"
}

# ---------------------------------------------------------- phase: setup ----
setup () {
  echo ">> building networks + generating routes (one-time, sequential)"
  python scripts/build_2x2_network.py
  python scripts/generate_routes.py
  python scripts/generate_routes.py --with-pedestrians
}

# ---------------------------------------------------------- phase: train ----
train () {
  echo ">> training: ${#ALGOS[@]} algos x ${#INTENSITIES[@]} intensities x 2 variants x ${#SEEDS[@]} seed(s)"
  : > logs/train_jobs.txt
  for algo in "${ALGOS[@]}"; do
    for intensity in "${INTENSITIES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for peds in "" "--with-pedestrians"; do
          echo "python scripts/train_${algo}.py --intensity ${intensity} --seed ${seed} ${peds}" \
            >> logs/train_jobs.txt
        done
      done
    done
  done
  echo "   $(wc -l < logs/train_jobs.txt) jobs | -j ${JOBS} | OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  run_parallel logs/train_jobs.txt logs/train.joblog
}

# ------------------------------------------------------- phase: baseline ----
baseline () {
  echo ">> SUMO baseline on held-out test routes"
  : > logs/baseline_jobs.txt
  for intensity in "${INTENSITIES[@]}"; do
    for peds in "" "--with-pedestrians"; do
      echo "python scripts/run_baseline.py --split test --intensity ${intensity} ${peds}" \
        >> logs/baseline_jobs.txt
    done
  done
  run_parallel logs/baseline_jobs.txt logs/baseline.joblog
}

# ----------------------------------------------------------- phase: eval ----
# NOTE: eval_dir() namespaces results by (algo, split, intensity, route_seed)
# but NOT by training seed. With more than one training seed, the later seed's
# eval overwrites the earlier one. Only multi-seed-safe if you first extend
# eval_dir() to include the seed. With SEEDS="0" (24 models) this is fine.
eval_ckpts () {
  echo ">> evaluating trained checkpoints"
  : > logs/eval_jobs.txt
  for algo in "${ALGOS[@]}"; do
    for intensity in "${INTENSITIES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        veh="results/checkpoints/${algo}/${intensity}/seed_${seed}/final.pt"
        ped="results/checkpoints/${algo}/peds/${intensity}/seed_${seed}/final.pt"
        echo "python scripts/eval_policy.py --checkpoint ${veh} --split test --intensity ${intensity}" \
          >> logs/eval_jobs.txt
        echo "python scripts/eval_policy.py --checkpoint ${ped} --split test --intensity ${intensity} --with-pedestrians" \
          >> logs/eval_jobs.txt
      done
    done
  done
  run_parallel logs/eval_jobs.txt logs/eval.joblog
}

# -------------------------------------------------------- phase: compare ----
compare () {
  echo ">> building comparison dashboards"
  python scripts/compare_results.py --split test
}

# ----------------------------------------------------------------- main -----
case "${1:-all}" in
  setup)    setup ;;
  train)    train ;;
  baseline) baseline ;;
  eval)     eval_ckpts ;;
  compare)  compare ;;
  all)      setup; train; baseline; eval_ckpts; compare ;;
  *) echo "unknown phase: $1" >&2; exit 1 ;;
esac
echo ">> done: ${1:-all}"