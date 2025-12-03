#!/usr/bin/env python3
"""Quick learning test - 5 games with shorter time limit"""

import sys
import os
import subprocess
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).parent
CONTEST_DIR = SCRIPT_DIR.parent / "pacman-contest"
sys.path.insert(0, str(CONTEST_DIR / "src"))

from training_framework import QLearningAgent

def get_stats(filepath):
    if not Path(filepath).exists():
        return 0, 0
    learner = QLearningAgent()
    learner.load(filepath)
    return learner.episodes, len(learner.q_values)

def run_game(num, layout="RANDOM"):
    agent_path = SCRIPT_DIR / "my_team_hybrid.py"
    contest_src_dir = CONTEST_DIR / "src" / "contest"
    
    cmd = [
        sys.executable, "../../runner.py",
        "-r", str(agent_path.absolute()),
        "-b", "baseline_team",
        "-l", layout,
        "-i", "300",  # Shorter games (300 moves instead of 1200)
        "-q"
    ]
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(CONTEST_DIR / "src")
    
    start = time.time()
    result = subprocess.run(cmd, cwd=str(contest_src_dir), capture_output=True, text=True, timeout=120, env=env)
    duration = time.time() - start
    
    output = result.stdout + result.stderr
    winner = 'LOSS'
    if 'red' in output.lower() and 'wins' in output.lower():
        winner = 'WIN'
    elif 'tie' in output.lower():
        winner = 'TIE'
    
    return winner, duration

print("\n" + "="*60)
print("QUICK LEARNING VERIFICATION TEST")
print("="*60)
print("5 games × 300 moves each (~2 minutes total)\n")

# Clean start
for f in ['offensive_strategy.pkl', 'defensive_strategy.pkl']:
    Path(f).unlink(missing_ok=True)

results = []
layouts = ["RANDOM", "defaultCapture", "mediumCapture"]

for i in range(1, 6):
    layout = layouts[(i-1) % len(layouts)]
    
    # Before stats
    eps_before, qvals_before = get_stats('offensive_strategy.pkl')
    
    print(f"\nGame {i}/{5} [{layout}]...", end=' ', flush=True)
    winner, duration = run_game(i, layout)
    
    time.sleep(0.3)  # Wait for file write
    
    # After stats
    eps_after, qvals_after = get_stats('offensive_strategy.pkl')
    delta = qvals_after - qvals_before
    
    results.append({'winner': winner, 'qvals': qvals_after, 'delta': delta})
    
    print(f"{winner} ({duration:.1f}s) | Q-values: {qvals_after} (+{delta})")

# Analysis
wins = sum(1 for r in results if r['winner'] == 'WIN')
total_qvals = results[-1]['qvals']
avg_delta = sum(r['delta'] for r in results) / len(results)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Performance: {wins}/5 wins ({wins/5*100:.0f}% win rate)")
print(f"Total Q-values learned: {total_qvals}")
print(f"Avg Q-values per game: {avg_delta:.1f}")

if total_qvals > 0:
    print(f"\n✅ LEARNING VERIFIED: Agent learned {total_qvals} strategy Q-values")
    if total_qvals > 200:
        print("✅ STRONG LEARNING: Substantial knowledge acquired")
else:
    print("\n❌ NO LEARNING DETECTED")

if wins >= 2:
    print(f"✅ COMPETITIVE: {wins/5*100:.0f}% win rate vs baseline")

print("\nNote: This is a quick test with 300 moves/game.")
print("Full training uses 1200 moves/game for better learning.")
print("="*60)
