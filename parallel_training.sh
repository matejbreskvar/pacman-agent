#!/bin/bash
#
# Parallel Training Script - Use ALL your cores!
# ===============================================
#
# This script runs multiple training workers in parallel.
# Perfect for multi-core machines or compute clusters.
#
# Usage:
#   ./parallel_training.sh [num_workers] [games_per_worker]
#
# Examples:
#   ./parallel_training.sh 4 500      # 4 workers, 500 games each = 2000 total games
#   ./parallel_training.sh 16 1000    # 16 workers, 1000 games each = 16000 total games
#   ./parallel_training.sh 128 500    # 128 workers for massive cluster = 64000 games!

set -e

# Default values
NUM_WORKERS=${1:-16}
GAMES_PER_WORKER=${2:-500}
START_WORKER_ID=${3:-200}

echo "======================================================================"
echo "PARALLEL TRAINING SETUP"
echo "======================================================================"
echo "Workers: $NUM_WORKERS"
echo "Worker IDs: $START_WORKER_ID - $((START_WORKER_ID + NUM_WORKERS - 1))"
echo "Games per worker: $GAMES_PER_WORKER"
echo "Total games: $((NUM_WORKERS * GAMES_PER_WORKER))"
echo ""

# Check Python environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Checking dependencies..."
pip install -q numpy frozendict 2>/dev/null || true

# Create work directory
mkdir -p distributed_work
mkdir -p logs

echo ""
echo "======================================================================"
echo "STARTING $NUM_WORKERS PARALLEL WORKERS"
echo "======================================================================"
echo ""

# Start all workers in background
worker_pids=()
for i in $(seq $START_WORKER_ID $((START_WORKER_ID + NUM_WORKERS - 1))); do
    log_file="logs/worker_${i}.log"
    echo "Starting worker $i (log: $log_file)"
    
    python3 distributed_training.py worker \
        --id $i \
        --games $GAMES_PER_WORKER \
        > "$log_file" 2>&1 &
    
    worker_pids+=($!)
    
    # Small delay to stagger starts
    sleep 0.5
done

echo ""
echo "✓ All $NUM_WORKERS workers started!"
echo ""
echo "Worker PIDs: ${worker_pids[@]}"
echo ""
echo "======================================================================"
echo "MONITORING"
echo "======================================================================"
echo "To monitor progress:"
echo "  tail -f logs/worker_1.log        # Watch worker 1"
echo "  tail -f logs/worker_*.log        # Watch all workers"
echo "  ps aux | grep distributed_training.py  # Check running workers"
echo ""
echo "To stop all workers:"
echo "  pkill -f distributed_training.py"
echo ""
echo "======================================================================"

# Function to check if workers are done
check_workers() {
    local all_done=true
    for pid in "${worker_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            all_done=false
            break
        fi
    done
    echo $all_done
}

# Wait for completion
echo "Waiting for workers to complete..."
echo "(Press Ctrl+C to stop monitoring, workers will continue in background)"
echo ""

start_time=$(date +%s)
completed=0

while [ "$(check_workers)" = "false" ]; do
    sleep 30
    
    # Count completed workers
    new_completed=0
    for pid in "${worker_pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            ((new_completed++))
        fi
    done
    
    if [ $new_completed -gt $completed ]; then
        completed=$new_completed
        elapsed=$(($(date +%s) - start_time))
        remaining=$((NUM_WORKERS - completed))
        
        echo "[$(date '+%H:%M:%S')] Progress: $completed/$NUM_WORKERS workers done ($remaining remaining) - ${elapsed}s elapsed"
    fi
done

echo ""
echo "======================================================================"
echo "ALL WORKERS COMPLETED!"
echo "======================================================================"

elapsed=$(($(date +%s) - start_time))
echo "Total time: ${elapsed}s ($((elapsed / 60)) minutes)"
echo ""

# Merge results
echo "Merging results from all workers..."
worker_list=$(seq -s, $START_WORKER_ID $((START_WORKER_ID + NUM_WORKERS - 1)))

python3 distributed_training.py merge --workers "$worker_list"

echo ""
echo "======================================================================"
echo "TRAINING COMPLETE!"
echo "======================================================================"
echo ""
echo "Summary logs available in: logs/"
echo "Worker checkpoints in: distributed_work/"
echo "Merged agent: offensive_qlearning_merged.pkl, defensive_qlearning_merged.pkl"
echo ""
echo "To use the merged agent:"
echo "  mv offensive_qlearning_merged.pkl offensive_qlearning.pkl"
echo "  mv defensive_qlearning_merged.pkl defensive_qlearning.pkl"
echo "  python3 compare_agents.py  # Test the trained agent"
echo ""
echo "To continue training:"
echo "  ./parallel_training.sh $NUM_WORKERS $GAMES_PER_WORKER"
echo ""
