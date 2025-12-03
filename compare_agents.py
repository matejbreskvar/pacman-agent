#!/usr/bin/env python3
"""
Agent Comparison Script
Tests and compares all available agents
"""

import subprocess
import sys
import os
from pathlib import Path

CONTEST_DIR = "/home/matej/FAKS/PacMan/pacman-contest"
AGENT_DIR = "/home/matej/FAKS/PacMan/pacman-agent"
PYTHON_BIN = str(Path(AGENT_DIR) / "venv" / "bin" / "python")

def run_match(red_team, blue_team, num_games=5):
    """Run a match between two teams"""
    cmd = [
        PYTHON_BIN,
        "runner.py",
        "-r", red_team,
        "-b", blue_team,
        "-n", str(num_games),
        "-q"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CONTEST_DIR}/src:{env.get('PYTHONPATH', '')}"
    
    try:
        result = subprocess.run(
            cmd,
            cwd=CONTEST_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=300
        )
        return parse_result(result.stdout + result.stderr)
    except Exception as e:
        return None

def parse_result(output):
    """Parse match result"""
    for line in output.split('\n'):
        if 'Red Win Rate:' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if '/' in part:
                    wins, total = part.split('/')
                    return {
                        'red_wins': int(wins),
                        'total': int(total.rstrip(')'))
                    }
    return None

def main():
    print("=" * 70)
    print("PACMAN CAPTURE AGENT COMPARISON")
    print("=" * 70)
    print()
    
    agents = [
        ("src/contest/baseline_team", "Baseline"),
        ("../pacman-agent/my_team", "Original"),
        ("../pacman-agent/my_team_v2", "Improved v2"),
        ("../pacman-agent/my_team_learning", "Learning AI"),
        ("../pacman-agent/variant_agents", "Aggressive/Cautious"),
    ]
    
    # Test if learning agent exists
    learning_pkl = Path(AGENT_DIR) / "offensive_qlearning.pkl"
    if not learning_pkl.exists():
        print("⚠️  Learning agent not trained yet. Train with:")
        print("   ./quickstart_training.sh")
        print()
        agents = agents[:-2]  # Remove learning and variant agents
    
    print(f"Testing {len(agents)} agents with 5 games each...")
    print()
    
    results = {}
    
    # Test each agent vs baseline
    baseline = agents[0][0]
    
    for agent_path, agent_name in agents[1:]:
        print(f"Testing {agent_name:20} vs Baseline... ", end='', flush=True)
        
        result = run_match(agent_path, baseline, num_games=5)
        
        if result:
            win_rate = result['red_wins'] / result['total'] * 100
            results[agent_name] = {
                'wins': result['red_wins'],
                'total': result['total'],
                'win_rate': win_rate
            }
            print(f"{win_rate:5.1f}% ({result['red_wins']}/{result['total']})")
        else:
            print("FAILED")
            results[agent_name] = None
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Agent':<25} {'Win Rate':>12} {'Record':>12}")
    print("-" * 70)
    
    for agent_name in sorted(results.keys(), 
                            key=lambda x: results[x]['win_rate'] if results[x] else 0, 
                            reverse=True):
        if results[agent_name]:
            r = results[agent_name]
            print(f"{agent_name:<25} {r['win_rate']:>11.1f}% {r['wins']:>5}/{r['total']:<5}")
        else:
            print(f"{agent_name:<25} {'FAILED':>12}")
    
    print()
    print("=" * 70)
    
    # Recommendations
    print()
    print("RECOMMENDATIONS:")
    print()
    
    best_agent = max(results.keys(), 
                    key=lambda x: results[x]['win_rate'] if results[x] else 0)
    
    if results[best_agent]:
        win_rate = results[best_agent]['win_rate']
        print(f"🏆 Best performer: {best_agent} ({win_rate:.1f}% win rate)")
        print()
        
        if "Learning" in best_agent:
            print("✨ Your learning agent is performing best!")
            print("   Continue training for even better results.")
        elif "v2" in best_agent:
            print("✨ Your improved agent v2 is working well!")
            if learning_pkl.exists():
                print("   Learning agent may improve with more training.")
            else:
                print("   Consider training the learning agent for potential improvement.")
        else:
            print("⚠️  Consider training the learning agent or using improved v2.")
    
    print()

if __name__ == '__main__':
    main()
