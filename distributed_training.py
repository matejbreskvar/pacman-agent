#!/usr/bin/env python3
"""
Distributed Training Manager for Pacman Agent
=============================================

This script enables:
1. Running multiple parallel training sessions
2. Merging results from different machines
3. Checkpoint management and recovery
4. Resume training from any point

Usage Examples:
--------------
# Start a training worker (run on each core/machine)
python3 distributed_training.py worker --id 1 --games 100

# Merge results from multiple workers
python3 distributed_training.py merge --workers 1,2,3,4

# Resume interrupted training
python3 distributed_training.py resume --checkpoint checkpoints/latest.pkl

# Multi-machine setup:
# Machine 1: python3 distributed_training.py worker --id 1 --games 500
# Machine 2: python3 distributed_training.py worker --id 2 --games 500
# Then merge: python3 distributed_training.py merge --workers 1,2
"""

import argparse
import os
import random
import sys
import time
import pickle
import subprocess
from pathlib import Path
from collections import defaultdict

# Add contest package to path
CONTEST_DIR = Path(__file__).parent.parent / "pacman-contest" / "src"
sys.path.insert(0, str(CONTEST_DIR))

from training_framework import QLearningAgent, DeepQLearningAgent


class CheckpointManager:
    """Manages training checkpoints for pause/resume"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, offensive_learner, defensive_learner, metadata):
        """Save a training checkpoint"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save agents
        offensive_path = self.checkpoint_dir / f"offensive_{timestamp}.pkl"
        defensive_path = self.checkpoint_dir / f"defensive_{timestamp}.pkl"
        
        offensive_learner.save(str(offensive_path), metadata)
        defensive_learner.save(str(defensive_path), metadata)
        
        # Save metadata
        meta_path = self.checkpoint_dir / f"metadata_{timestamp}.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Create/update 'latest' symlinks
        latest_off = self.checkpoint_dir / "latest_offensive.pkl"
        latest_def = self.checkpoint_dir / "latest_defensive.pkl"
        latest_meta = self.checkpoint_dir / "latest_metadata.pkl"
        
        if latest_off.exists():
            latest_off.unlink()
        if latest_def.exists():
            latest_def.unlink()
        if latest_meta.exists():
            latest_meta.unlink()
        
        latest_off.symlink_to(offensive_path.name)
        latest_def.symlink_to(defensive_path.name)
        latest_meta.symlink_to(meta_path.name)
        
        print(f"\n✓ Checkpoint saved at {timestamp}")
        print(f"  Offensive: {offensive_path.name}")
        print(f"  Defensive: {defensive_path.name}")
        
        return timestamp
    
    def load_checkpoint(self, checkpoint_id="latest"):
        """Load a training checkpoint"""
        if checkpoint_id == "latest":
            offensive_path = self.checkpoint_dir / "latest_offensive.pkl"
            defensive_path = self.checkpoint_dir / "latest_defensive.pkl"
            meta_path = self.checkpoint_dir / "latest_metadata.pkl"
        else:
            offensive_path = self.checkpoint_dir / f"offensive_{checkpoint_id}.pkl"
            defensive_path = self.checkpoint_dir / f"defensive_{checkpoint_id}.pkl"
            meta_path = self.checkpoint_dir / f"metadata_{checkpoint_id}.pkl"
        
        if not offensive_path.exists():
            print(f"✗ No checkpoint found: {checkpoint_id}")
            return None, None, None
        
        # Load agents
        offensive_learner = QLearningAgent()
        defensive_learner = QLearningAgent()
        
        offensive_learner.load(str(offensive_path))
        defensive_learner.load(str(defensive_path))
        
        # Load metadata
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
        
        print(f"\n✓ Checkpoint loaded: {checkpoint_id}")
        return offensive_learner, defensive_learner, metadata
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = []
        for path in sorted(self.checkpoint_dir.glob("offensive_*.pkl")):
            timestamp = path.stem.replace("offensive_", "")
            size_off = path.stat().st_size / 1024  # KB
            
            def_path = self.checkpoint_dir / f"defensive_{timestamp}.pkl"
            size_def = def_path.stat().st_size / 1024 if def_path.exists() else 0
            
            checkpoints.append({
                'timestamp': timestamp,
                'size_kb': size_off + size_def,
                'path': path
            })
        
        return checkpoints


class DistributedTrainer:
    """Manages distributed training across multiple workers"""
    
    def __init__(self, work_dir="distributed_work"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.checkpoint_mgr = CheckpointManager()
    
    def run_worker(self, worker_id, num_games, opponent="baseline"):
        """Run training worker"""
        print(f"\n{'='*60}")
        print(f"WORKER {worker_id}: Starting training")
        print(f"{'='*60}")
        print(f"Games to play: {num_games}")
        print(f"Opponent: {opponent}")
        print(f"Work directory: {self.work_dir}")
        
        # Map opponent names to actual paths (relative to src/contest where we run from)
        opponent_map = {
            'baseline': 'baseline_team',
            'team_template': '../../agents/team_template/my_team',
            'team_current': '../../agents/team_current/my_team',
        }
        
        # Use mapped opponent or the provided path
        opponent_path = opponent_map.get(opponent, opponent)
        
        # Create worker-specific filenames
        worker_off = self.work_dir / f"worker_{worker_id}_offensive.pkl"
        worker_def = self.work_dir / f"worker_{worker_id}_defensive.pkl"
        
        # Load existing checkpoint if resuming
        offensive_learner = QLearningAgent()
        defensive_learner = QLearningAgent()
        
        if worker_off.exists():
            print(f"\n→ Resuming from existing worker checkpoint...")
            offensive_learner.load(str(worker_off))
            defensive_learner.load(str(worker_def))
        
        # Save learners to the standard location so my_team_hybrid.py can load them
        # This is how the learning agents will access the training state
        offensive_learner.save('offensive_strategy.pkl', {'worker_id': worker_id, 'training': True})
        defensive_learner.save('defensive_strategy.pkl', {'worker_id': worker_id, 'training': True})
        
        # Run training games
        contest_dir = Path(__file__).parent.parent / "pacman-contest"
        agent_path = Path(__file__).parent / "my_team_hybrid.py"
        
        wins = 0
        losses = 0
        start_time = time.time()
        
        # Competition uses mix of random and official layouts
        layouts = [
            "RANDOM",  # Random layout with different seed each time
            "defaultCapture", "mediumCapture", "officeCapture",
            "strategicCapture", "crowdedCapture", "distantCapture"
        ]
        
        for game_num in range(num_games):
            layout = random.choice(layouts)
            print(f"\n--- Worker {worker_id}: Game {game_num + 1}/{num_games} [{layout}] ---")
            
            # Run game (must run from src/contest for layout loading to work)
            contest_src_dir = contest_dir / "src" / "contest"
            cmd = [
                sys.executable,
                "../../runner.py",  # Relative to src/contest
                "-r", str(agent_path.absolute()),
                "-b", opponent_path,
                "-l", layout,
                "-i", "1200",  # Competition time limit: 1200 moves (300 per agent)
                "-q"  # Quiet mode: no graphics, minimal output
            ]
            
            env = os.environ.copy()
            env['PYTHONPATH'] = str(CONTEST_DIR)
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(contest_src_dir),  # Run from src/contest for layouts
                    capture_output=False,  # Don't capture - let output flow to console
                    text=True,
                    timeout=300,  # 5 minutes per game
                    env=env
                )
                
                # Since we're not capturing, check return code
                if result.returncode == 0:
                    wins += 1
                    result_str = "WIN"
                else:
                    losses += 1  
                    result_str = "LOSS"
                
                print(f"  Result: {result_str} (W:{wins} L:{losses})")
                
                # Load updated learner state after each game
                # (my_team_hybrid.py saves to these files in final())
                try:
                    offensive_learner.load('offensive_strategy.pkl')
                    defensive_learner.load('defensive_strategy.pkl')
                    print(f"  → Loaded: Off Episodes={offensive_learner.episodes}, Def Episodes={defensive_learner.episodes}")
                except Exception as e:
                    print(f"  Warning: Could not load updated state: {e}")
                
            except subprocess.TimeoutExpired:
                print("  Game timed out")
                losses += 1
            except Exception as e:
                print(f"  Error: {e}")
                losses += 1
            
            # Save checkpoint every 10 games
            if (game_num + 1) % 10 == 0:
                offensive_learner.save(str(worker_off), {
                    'worker_id': worker_id,
                    'games_played': game_num + 1,
                    'wins': wins,
                    'losses': losses
                })
                defensive_learner.save(str(worker_def))
                print(f"  ✓ Checkpoint saved (every 10 games)")
        
        # Final save
        elapsed = time.time() - start_time
        win_rate = wins / num_games if num_games > 0 else 0
        
        metadata = {
            'worker_id': worker_id,
            'games_played': num_games,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'elapsed_time': elapsed,
            'completed': True
        }
        
        offensive_learner.save(str(worker_off), metadata)
        defensive_learner.save(str(worker_def), metadata)
        
        print(f"\n{'='*60}")
        print(f"WORKER {worker_id}: Training Complete")
        print(f"{'='*60}")
        print(f"Games: {num_games}")
        print(f"Wins: {wins} ({win_rate*100:.1f}%)")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Speed: {num_games/(elapsed/60):.1f} games/min")
        print(f"\nCheckpoint saved to:")
        print(f"  {worker_off}")
        print(f"  {worker_def}")
    
    def merge_workers(self, worker_ids):
        """Merge results from multiple workers"""
        print(f"\n{'='*60}")
        print(f"MERGING WORKERS: {worker_ids}")
        print(f"{'='*60}")
        
        # Load first worker as base
        base_id = worker_ids[0]
        base_off_path = self.work_dir / f"worker_{base_id}_offensive.pkl"
        base_def_path = self.work_dir / f"worker_{base_id}_defensive.pkl"
        
        if not base_off_path.exists():
            print(f"✗ Worker {base_id} not found")
            return
        
        merged_offensive = QLearningAgent()
        merged_defensive = QLearningAgent()
        
        merged_offensive.load(str(base_off_path))
        merged_defensive.load(str(base_def_path))
        
        print(f"\nBase worker: {base_id}")
        
        # Merge remaining workers
        total_episodes = merged_offensive.episodes
        total_games = 0
        total_wins = 0
        
        for worker_id in worker_ids[1:]:
            off_path = self.work_dir / f"worker_{worker_id}_offensive.pkl"
            def_path = self.work_dir / f"worker_{worker_id}_defensive.pkl"
            
            if not off_path.exists():
                print(f"✗ Worker {worker_id} not found, skipping")
                continue
            
            worker_off = QLearningAgent()
            worker_def = QLearningAgent()
            
            worker_off.load(str(off_path))
            worker_def.load(str(def_path))
            
            # Merge
            merged_offensive.merge(worker_off)
            merged_defensive.merge(worker_def)
            
            total_episodes += worker_off.episodes
            print(f"+ Merged worker {worker_id}")
        
        # Save merged result
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        merged_off_path = Path("offensive_qlearning_merged.pkl")
        merged_def_path = Path("defensive_qlearning_merged.pkl")
        
        metadata = {
            'merged_workers': worker_ids,
            'total_episodes': total_episodes,
            'merge_timestamp': timestamp
        }
        
        merged_offensive.save(str(merged_off_path), metadata)
        merged_defensive.save(str(merged_def_path), metadata)
        
        print(f"\n{'='*60}")
        print(f"MERGE COMPLETE")
        print(f"{'='*60}")
        print(f"Workers merged: {len(worker_ids)}")
        print(f"Total episodes: {total_episodes}")
        print(f"Merged agent saved to:")
        print(f"  {merged_off_path}")
        print(f"  {merged_def_path}")
        print(f"\nTo use merged agent:")
        print(f"  mv {merged_off_path} offensive_strategy.pkl")
        print(f"  mv {merged_def_path} defensive_strategy.pkl")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed Training Manager for Pacman Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run worker on single core/machine
  python3 distributed_training.py worker --id 1 --games 100
  
  # Run on multi-core machine (start 4 workers)
  for i in {1..4}; do
    python3 distributed_training.py worker --id $i --games 250 &
  done
  
  # Merge results from 4 workers
  python3 distributed_training.py merge --workers 1,2,3,4
  
  # List checkpoints
  python3 distributed_training.py checkpoints
  
  # Resume from checkpoint
  python3 distributed_training.py resume --checkpoint latest
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Worker command
    worker_parser = subparsers.add_parser('worker', help='Run training worker')
    worker_parser.add_argument('--id', type=int, required=True, help='Worker ID')
    worker_parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    worker_parser.add_argument('--opponent', default='baseline', help='Opponent to train against')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple workers')
    merge_parser.add_argument('--workers', required=True, help='Comma-separated worker IDs (e.g., 1,2,3,4)')
    
    # Checkpoints command
    subparsers.add_parser('checkpoints', help='List available checkpoints')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume from checkpoint')
    resume_parser.add_argument('--checkpoint', default='latest', help='Checkpoint ID or "latest"')
    
    args = parser.parse_args()
    
    if args.command == 'worker':
        trainer = DistributedTrainer()
        trainer.run_worker(args.id, args.games, args.opponent)
    
    elif args.command == 'merge':
        worker_ids = [int(w.strip()) for w in args.workers.split(',')]
        trainer = DistributedTrainer()
        trainer.merge_workers(worker_ids)
    
    elif args.command == 'checkpoints':
        checkpoint_mgr = CheckpointManager()
        checkpoints = checkpoint_mgr.list_checkpoints()
        
        if not checkpoints:
            print("No checkpoints found")
        else:
            print(f"\nAvailable checkpoints:")
            print(f"{'Timestamp':<20} {'Size (KB)':<12}")
            print("-" * 35)
            for cp in checkpoints:
                print(f"{cp['timestamp']:<20} {cp['size_kb']:>10.1f}")
    
    elif args.command == 'resume':
        checkpoint_mgr = CheckpointManager()
        off, def_agent, metadata = checkpoint_mgr.load_checkpoint(args.checkpoint)
        
        if off is None:
            print("Failed to load checkpoint")
            return
        
        print("\nCheckpoint info:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print("\nTo resume training:")
        print(f"  python3 train_agent.py --mode both --iterations 100")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
