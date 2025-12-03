#!/usr/bin/env python3
"""
Training Script for Pacman Capture Agent
Runs multiple games to train learning agents against various opponents
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

# Configuration
CONTEST_DIR = "../pacman-contest"
AGENT_DIR = "."
PYTHON_BIN = None  # Will be set based on venv

def find_python():
    """Find the correct Python executable"""
    # Check for venv
    venv_python = Path("venv/bin/python")
    if venv_python.exists():
        return str(venv_python.absolute())
    
    # Fallback to system python3
    return "python3"

def run_training_game(red_team, blue_team, layout='RANDOM', num_games=1, quiet=True):
    """
    Run a training game
    
    Args:
        red_team: Path to red team agent
        blue_team: Path to blue team agent
        layout: Map layout to use
        num_games: Number of games to run
        quiet: Suppress graphics
    
    Returns:
        Result string from the game
    """
    global PYTHON_BIN
    
    if PYTHON_BIN is None:
        PYTHON_BIN = find_python()
    
    # Build command
    cmd = [
        PYTHON_BIN,
        "runner.py",
        "-r", red_team,
        "-b", blue_team,
        "-l", layout,
        "-n", str(num_games),
    ]
    
    if quiet:
        cmd.append("-q")
    
    # Set environment
    env = os.environ.copy()
    contest_src = os.path.join(CONTEST_DIR, "src")
    env["PYTHONPATH"] = f"{contest_src}:{env.get('PYTHONPATH', '')}"
    
    # Run game
    try:
        result = subprocess.run(
            cmd,
            cwd=CONTEST_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per game batch
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("Game timed out!")
        return ""
    except Exception as e:
        print(f"Error running game: {e}")
        return ""

def parse_results(output):
    """Parse game results from output"""
    lines = output.split('\n')
    
    results = {
        'red_wins': 0,
        'blue_wins': 0,
        'ties': 0,
        'avg_score': 0.0,
        'scores': []
    }
    
    for line in lines:
        if 'Red Win Rate:' in line:
            # Extract win rate
            parts = line.split()
            if len(parts) >= 3:
                wins_str = parts[3]  # Should be "X/Y"
                if '/' in wins_str:
                    wins, total = wins_str.split('/')
                    results['red_wins'] = int(wins)
        
        elif 'Blue Win Rate:' in line:
            parts = line.split()
            if len(parts) >= 3:
                wins_str = parts[3]
                if '/' in wins_str:
                    wins, total = wins_str.split('/')
                    results['blue_wins'] = int(wins)
        
        elif 'Average Score:' in line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    results['avg_score'] = float(parts[2])
                except ValueError:
                    pass
        
        elif 'Scores:' in line:
            # Extract individual scores
            parts = line.split(':')
            if len(parts) >= 2:
                score_str = parts[1].strip()
                scores = [s.strip().rstrip(',') for s in score_str.split()]
                results['scores'] = [float(s) for s in scores if s and s != ',']
    
    return results

def train_against_opponent(learning_team, opponent_team, num_games=10, layout='RANDOM'):
    """
    Train learning agent against a specific opponent
    """
    print(f"\n{'='*60}")
    print(f"Training {learning_team} vs {opponent_team}")
    print(f"Games: {num_games}, Layout: {layout}")
    print(f"{'='*60}")
    
    output = run_training_game(learning_team, opponent_team, layout, num_games)
    results = parse_results(output)
    
    print(f"Results: Red Wins: {results['red_wins']}, Blue Wins: {results['blue_wins']}")
    print(f"Average Score: {results['avg_score']:.2f}")
    
    return results

def training_curriculum(learning_team, num_iterations=10, games_per_iteration=10):
    """
    Run a training curriculum with increasing difficulty
    
    Args:
        learning_team: Path to learning agent
        num_iterations: Number of training iterations
        games_per_iteration: Games per iteration
    """
    
    print(f"\n{'#'*60}")
    print(f"# TRAINING CURRICULUM")
    print(f"# Learning Team: {learning_team}")
    print(f"# Iterations: {num_iterations}")
    print(f"# Games per iteration: {games_per_iteration}")
    print(f"{'#'*60}\n")
    
    opponents = [
        ("src/contest/baseline_team", "Baseline"),
        ("../pacman-agent/my_team", "Original Agent"),
        ("../pacman-agent/my_team_v2", "Improved Agent"),
    ]
    
    layouts = ['RANDOM', 'defaultCapture', 'mediumCapture']
    
    all_results = []
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Train against each opponent
        for opponent_path, opponent_name in opponents:
            for layout in layouts:
                results = train_against_opponent(
                    learning_team,
                    opponent_path,
                    num_games=games_per_iteration,
                    layout=layout
                )
                
                all_results.append({
                    'iteration': iteration + 1,
                    'opponent': opponent_name,
                    'layout': layout,
                    'results': results
                })
                
                # Brief pause between matches
                time.sleep(1)
        
        # Save checkpoint
        print(f"\nCompleted iteration {iteration + 1}")
        print("Learned models saved automatically by agents")
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1} SUMMARY")
        print(f"{'='*60}")
        
        recent_results = [r for r in all_results if r['iteration'] == iteration + 1]
        total_red_wins = sum(r['results']['red_wins'] for r in recent_results)
        total_blue_wins = sum(r['results']['blue_wins'] for r in recent_results)
        total_games = total_red_wins + total_blue_wins
        
        if total_games > 0:
            win_rate = total_red_wins / total_games * 100
            print(f"Overall Win Rate: {win_rate:.1f}% ({total_red_wins}/{total_games})")
        
        avg_scores = [r['results']['avg_score'] for r in recent_results]
        if avg_scores:
            print(f"Average Score: {sum(avg_scores) / len(avg_scores):.2f}")
    
    # Final summary
    print(f"\n{'#'*60}")
    print(f"# TRAINING COMPLETE")
    print(f"{'#'*60}\n")
    
    # Calculate improvement over time
    print("Performance over time:")
    for iteration in range(1, num_iterations + 1):
        iter_results = [r for r in all_results if r['iteration'] == iteration]
        total_red_wins = sum(r['results']['red_wins'] for r in iter_results)
        total_blue_wins = sum(r['results']['blue_wins'] for r in iter_results)
        total_games = total_red_wins + total_blue_wins
        
        if total_games > 0:
            win_rate = total_red_wins / total_games * 100
            print(f"  Iteration {iteration}: {win_rate:.1f}% win rate")

def self_play_training(learning_team, num_games=100):
    """
    Train agents through self-play
    """
    print(f"\n{'#'*60}")
    print(f"# SELF-PLAY TRAINING")
    print(f"# Learning Team: {learning_team}")
    print(f"# Total Games: {num_games}")
    print(f"{'#'*60}\n")
    
    batch_size = 10
    num_batches = num_games // batch_size
    
    for batch in range(num_batches):
        print(f"\nBatch {batch + 1}/{num_batches}")
        
        # Self-play on random layouts
        output = run_training_game(
            learning_team,
            learning_team,
            layout='RANDOM',
            num_games=batch_size
        )
        
        results = parse_results(output)
        print(f"Batch complete. Avg score: {results['avg_score']:.2f}")
        
        time.sleep(1)
    
    print("\nSelf-play training complete!")

def main():
    parser = argparse.ArgumentParser(description='Train Pacman Capture agents')
    parser.add_argument('--agent', default='../pacman-agent/my_team_learning',
                       help='Path to learning agent')
    parser.add_argument('--mode', choices=['curriculum', 'selfplay', 'both'],
                       default='curriculum',
                       help='Training mode')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of training iterations')
    parser.add_argument('--games-per-iter', type=int, default=5,
                       help='Games per iteration')
    parser.add_argument('--selfplay-games', type=int, default=50,
                       help='Self-play games')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# PACMAN CAPTURE AGENT TRAINING")
    print(f"{'#'*60}\n")
    print(f"Agent: {args.agent}")
    print(f"Mode: {args.mode}")
    print(f"Python: {find_python()}")
    
    start_time = time.time()
    
    if args.mode in ['curriculum', 'both']:
        training_curriculum(
            args.agent,
            num_iterations=args.iterations,
            games_per_iteration=args.games_per_iter
        )
    
    if args.mode in ['selfplay', 'both']:
        self_play_training(
            args.agent,
            num_games=args.selfplay_games
        )
    
    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed/60:.1f} minutes")
    print("\nLearned models saved to:")
    print("  - offensive_qlearning.pkl")
    print("  - defensive_qlearning.pkl")

if __name__ == '__main__':
    main()
