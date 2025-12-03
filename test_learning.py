#!/usr/bin/env python3
"""
Test script to verify the hybrid agent learns and improves over time
"""

import sys
import os
import subprocess
from pathlib import Path
import pickle
import time

# Setup paths
SCRIPT_DIR = Path(__file__).parent
CONTEST_DIR = SCRIPT_DIR.parent / "pacman-contest"
sys.path.insert(0, str(CONTEST_DIR / "src"))

from training_framework import QLearningAgent

def run_game(game_num, layout="RANDOM"):
    """Run a single game and return the result"""
    agent_path = SCRIPT_DIR / "my_team_hybrid.py"
    contest_src_dir = CONTEST_DIR / "src" / "contest"
    
    cmd = [
        sys.executable,
        "../../runner.py",
        "-r", str(agent_path.absolute()),
        "-b", "baseline_team",
        "-l", layout,
        "-i", "1200",
        "-q"  # Quiet mode
    ]
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(CONTEST_DIR / "src")
    
    print(f"\n{'='*60}")
    print(f"Game {game_num}: {layout}")
    print('='*60)
    
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(contest_src_dir),
        capture_output=True,
        text=True,
        timeout=300,
        env=env
    )
    duration = time.time() - start
    
    # Parse output for score
    output = result.stdout + result.stderr
    score = None
    winner = None
    
    for line in output.split('\n'):
        if 'wins by' in line.lower():
            if 'red' in line.lower():
                winner = 'RED'
                # Extract score
                parts = line.split('by')
                if len(parts) > 1:
                    score = int(parts[1].split()[0])
            elif 'blue' in line.lower():
                winner = 'BLUE'
                parts = line.split('by')
                if len(parts) > 1:
                    score = -int(parts[1].split()[0])
        elif 'tie game' in line.lower():
            winner = 'TIE'
            score = 0
    
    print(f"Result: {winner} (score: {score if score is not None else 'N/A'})")
    print(f"Duration: {duration:.1f}s")
    
    return winner, score, duration

def get_learner_stats(filepath):
    """Get stats from a learner file"""
    if not Path(filepath).exists():
        return {'episodes': 0, 'q_values': 0, 'epsilon': 0.2}
    
    learner = QLearningAgent()
    learner.load(filepath)
    
    return {
        'episodes': learner.episodes,
        'q_values': len(learner.q_values),
        'epsilon': learner.epsilon,
        'avg_reward': sum(learner.episode_rewards[-10:]) / len(learner.episode_rewards[-10:]) if learner.episode_rewards else 0
    }

def main():
    print("\n" + "="*70)
    print("HYBRID AGENT LEARNING TEST")
    print("="*70)
    print("\nThis will train for 10 games and verify learning progression")
    print("Each game takes ~45 seconds with 1200 moves")
    print()
    
    # Clean slate
    for f in ['offensive_strategy.pkl', 'defensive_strategy.pkl']:
        if Path(f).exists():
            Path(f).unlink()
            print(f"Removed existing {f}")
    
    # Track results
    results = []
    layouts = ["RANDOM", "defaultCapture", "mediumCapture", "officeCapture"]
    
    for game_num in range(1, 11):
        layout = layouts[(game_num - 1) % len(layouts)]
        
        # Get stats before game
        off_before = get_learner_stats('offensive_strategy.pkl')
        def_before = get_learner_stats('defensive_strategy.pkl')
        
        # Run game
        winner, score, duration = run_game(game_num, layout)
        
        # Wait a moment for files to be written
        time.sleep(0.5)
        
        # Get stats after game
        off_after = get_learner_stats('offensive_strategy.pkl')
        def_after = get_learner_stats('defensive_strategy.pkl')
        
        # Calculate learning delta
        off_delta = off_after['q_values'] - off_before['q_values']
        def_delta = def_after['q_values'] - def_before['q_values']
        
        results.append({
            'game': game_num,
            'layout': layout,
            'winner': winner,
            'score': score,
            'duration': duration,
            'off_qvals': off_after['q_values'],
            'def_qvals': def_after['q_values'],
            'off_delta': off_delta,
            'def_delta': def_delta,
            'off_eps': off_after['epsilon'],
            'off_reward': off_after['avg_reward']
        })
        
        print(f"Offensive: {off_after['q_values']} Q-values (+{off_delta}), ε={off_after['epsilon']:.3f}")
        print(f"Defensive: {def_after['q_values']} Q-values (+{def_delta})")
        
        # Show progress
        wins = sum(1 for r in results if r['winner'] == 'RED')
        losses = sum(1 for r in results if r['winner'] == 'BLUE')
        ties = sum(1 for r in results if r['winner'] == 'TIE')
        
        print(f"\nProgress: {wins}W-{losses}L-{ties}T ({wins/len(results)*100:.0f}% win rate)")
    
    # Final analysis
    print("\n" + "="*70)
    print("LEARNING ANALYSIS")
    print("="*70)
    
    print("\n📊 Game-by-Game Results:")
    print(f"{'Game':<6} {'Layout':<18} {'Result':<8} {'Score':<7} {'Off Q':<8} {'Def Q':<8} {'ΔQ Off':<8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['game']:<6} {r['layout']:<18} {r['winner']:<8} "
              f"{r['score'] if r['score'] is not None else 'N/A':<7} "
              f"{r['off_qvals']:<8} {r['def_qvals']:<8} {r['off_delta']:<8}")
    
    # Statistics
    total_off_qvals = results[-1]['off_qvals']
    total_def_qvals = results[-1]['def_qvals']
    avg_off_delta = sum(r['off_delta'] for r in results) / len(results)
    avg_def_delta = sum(r['def_delta'] for r in results) / len(results)
    
    wins = sum(1 for r in results if r['winner'] == 'RED')
    losses = sum(1 for r in results if r['winner'] == 'BLUE')
    ties = sum(1 for r in results if r['winner'] == 'TIE')
    
    total_score = sum(r['score'] for r in results if r['score'] is not None)
    avg_score = total_score / len([r for r in results if r['score'] is not None])
    
    print("\n📈 Learning Statistics:")
    print(f"  Total Q-values learned (Offensive): {total_off_qvals}")
    print(f"  Total Q-values learned (Defensive): {total_def_qvals}")
    print(f"  Avg Q-values per game (Offensive): {avg_off_delta:.1f}")
    print(f"  Avg Q-values per game (Defensive): {avg_def_delta:.1f}")
    print(f"  Final epsilon: {results[-1]['off_eps']:.3f}")
    
    print(f"\n🏆 Performance:")
    print(f"  Record: {wins}W-{losses}L-{ties}T")
    print(f"  Win rate: {wins/len(results)*100:.1f}%")
    print(f"  Avg score differential: {avg_score:+.1f}")
    
    # Check for improvement trend
    first_half_wins = sum(1 for r in results[:5] if r['winner'] == 'RED')
    second_half_wins = sum(1 for r in results[5:] if r['winner'] == 'RED')
    
    first_half_score = sum(r['score'] for r in results[:5] if r['score'] is not None) / 5
    second_half_score = sum(r['score'] for r in results[5:] if r['score'] is not None) / 5
    
    print(f"\n🔬 Learning Trend:")
    print(f"  Games 1-5:  {first_half_wins}W ({first_half_wins/5*100:.0f}%), avg score: {first_half_score:+.1f}")
    print(f"  Games 6-10: {second_half_wins}W ({second_half_wins/5*100:.0f}%), avg score: {second_half_score:+.1f}")
    
    if second_half_wins > first_half_wins:
        print(f"  ✅ IMPROVEMENT: Won {second_half_wins - first_half_wins} more games in 2nd half!")
    elif second_half_score > first_half_score:
        print(f"  ✅ IMPROVEMENT: Score improved by {second_half_score - first_half_score:+.1f} in 2nd half!")
    else:
        print(f"  ⚠️  No clear improvement yet (needs more training)")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)
    
    # Verify learning
    if total_off_qvals > 0:
        print("\n✅ LEARNING VERIFIED: Agent is learning Q-values")
    else:
        print("\n❌ WARNING: No Q-values learned!")
    
    if total_off_qvals > 500:
        print("✅ STRONG LEARNING: Agent learned significant strategy knowledge")
    
    if wins/len(results) > 0.4:
        print(f"✅ COMPETITIVE: {wins/len(results)*100:.0f}% win rate vs baseline")

if __name__ == '__main__':
    main()
