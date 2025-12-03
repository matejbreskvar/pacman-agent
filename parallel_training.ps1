#
# Parallel Training Script for Windows - Use ALL your cores!
# ===========================================================
#
# This script runs multiple training workers in parallel.
# Perfect for multi-core machines.
#
# Usage:
#   .\parallel_training.ps1 [num_workers] [games_per_worker] [start_worker_id]
#
# Examples:
#   .\parallel_training.ps1 16 500        # 16 workers, 500 games each = 8000 total games
#   .\parallel_training.ps1 16 1000 200   # 16 workers, IDs 200-215, 1000 games each

param(
    [int]$NumWorkers = 70,
    [int]$GamesPerWorker = 500,
    [int]$StartWorkerId = 200
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================"
Write-Host "PARALLEL TRAINING SETUP"
Write-Host "======================================================================"
Write-Host "Workers: $NumWorkers"
Write-Host "Worker IDs: $StartWorkerId - $($StartWorkerId + $NumWorkers - 1)"
Write-Host "Games per worker: $GamesPerWorker"
Write-Host "Total games: $($NumWorkers * $GamesPerWorker)"
Write-Host ""

# Check Python environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Checking dependencies..."
pip install -q numpy frozendict 2>$null

# Create work directories
New-Item -ItemType Directory -Force -Path "distributed_work" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host ""
Write-Host "======================================================================"
Write-Host "STARTING $NumWorkers PARALLEL WORKERS"
Write-Host "======================================================================"
Write-Host ""

# Start all workers in background
$jobs = @()
for ($i = $StartWorkerId; $i -lt ($StartWorkerId + $NumWorkers); $i++) {
    $logFile = "$PWD\logs\worker_$i.log"
    Write-Host "Starting worker $i (log: $logFile)"
    
    $job = Start-Job -ScriptBlock {
        param($workerId, $games, $logPath, $scriptDir)
        Set-Location $scriptDir
        $env:PYTHONIOENCODING = "utf-8"
        & .\venv\Scripts\Activate.ps1
        python distributed_training.py worker --id $workerId --games $games *> $logPath
    } -ArgumentList $i, $GamesPerWorker, $logFile, $PWD
    
    $jobs += @{Id = $i; Job = $job}
    
    # Small delay to stagger starts
    Start-Sleep -Milliseconds 100
}

Write-Host ""
Write-Host "All $NumWorkers workers started!"
Write-Host ""
Write-Host "Job IDs: $($jobs.Job.Id -join ', ')"
Write-Host ""
Write-Host "======================================================================"
Write-Host "MONITORING"
Write-Host "======================================================================"
Write-Host "To monitor progress:"
Write-Host "  Get-Content logs\worker_200.log -Wait   # Watch worker 200"
Write-Host "  Get-Job                                  # Check job status"
Write-Host ""
Write-Host "To stop all workers:"
Write-Host "  Get-Job | Stop-Job; Get-Job | Remove-Job"
Write-Host ""
Write-Host "======================================================================"

# Wait for completion
Write-Host "Waiting for workers to complete..."
Write-Host "(Press Ctrl+C to stop monitoring, workers will continue in background)"
Write-Host ""

$startTime = Get-Date
$completed = 0

while ($jobs.Job | Where-Object { $_.State -eq 'Running' }) {
    Start-Sleep -Seconds 30
    
    $newCompleted = ($jobs.Job | Where-Object { $_.State -eq 'Completed' }).Count
    
    if ($newCompleted -gt $completed) {
        $completed = $newCompleted
        $elapsed = [int]((Get-Date) - $startTime).TotalSeconds
        $remaining = $NumWorkers - $completed
        
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Progress: $completed/$NumWorkers workers done ($remaining remaining) - ${elapsed}s elapsed"
    }
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "ALL WORKERS COMPLETED!"
Write-Host "======================================================================"

$elapsed = [int]((Get-Date) - $startTime).TotalSeconds
Write-Host "Total time: ${elapsed}s ($([int]($elapsed / 60)) minutes)"
Write-Host ""

# Clean up jobs
$jobs.Job | Remove-Job -Force

# Merge results
Write-Host "Merging results from all workers..."
$workerList = ($StartWorkerId..($StartWorkerId + $NumWorkers - 1)) -join ','

python distributed_training.py merge --workers $workerList

Write-Host ""
Write-Host "======================================================================"
Write-Host "TRAINING COMPLETE!"
Write-Host "======================================================================"
Write-Host ""
Write-Host "Summary logs available in: logs\"
Write-Host "Worker checkpoints in: distributed_work\"
Write-Host "Merged agent: offensive_qlearning_merged.pkl, defensive_qlearning_merged.pkl"
Write-Host ""
Write-Host "To use the merged agent:"
Write-Host "  Move-Item offensive_qlearning_merged.pkl offensive_qlearning.pkl -Force"
Write-Host "  Move-Item defensive_qlearning_merged.pkl defensive_qlearning.pkl -Force"
Write-Host "  python compare_agents.py  # Test the trained agent"
Write-Host ""
Write-Host "To continue training:"
Write-Host "  .\parallel_training.ps1 $NumWorkers $GamesPerWorker"
Write-Host ""
