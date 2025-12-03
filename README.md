# Pacman AI Agent

Advanced Pacman Capture-the-Flag agent with AI training capabilities.

## Quick Start

### 1. Setup (5 minutes)

```bash
git clone <repo-url>
cd pacman-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Test Agent (2 minutes)

```bash
python3 compare_agents.py
```

### 3. Train Agent

```bash
# Quick test (2 min, 20 games)
./parallel_training.sh 2 10

# Full training (use all your cores)
./parallel_training.sh 8 500        # 8 cores: ~6 hours, 4k games
./parallel_training.sh 64 1000      # 64 cores: ~2 hours, 64k games
./parallel_training.sh 1000 500     # 1000 cores: ~1 hour, 500k games
```

## Available Agents

- **`my_team.py`** - Original baseline (30% win rate)
- **`my_team_v2.py`** - Improved hand-crafted (60% win rate)
- **`my_team_hybrid.py`** ⭐ - RL + Advanced Search (trains to 80%+)

Uses AlphaZero-style approach: v2's intelligence + RL strategy optimization.

## Competition Settings

Training uses official competition rules:

- ✓ **Time Limit**: 1200 moves (300 per agent)
- ✓ **Layouts**: Random + official maps (defaultCapture, mediumCapture, officeCapture, strategicCapture, crowdedCapture, distantCapture)
- ✓ **Computation**: 1s per move, 15s initial setup, 3 warnings = forfeit
- ✓ **Graphics**: Disabled during training (`-q` flag)
- ✓ **Scoring**: Win = 3pts, Tie = 1pt, Loss = 0pts

## Training System

### Pause/Resume Training

```bash
./parallel_training.sh 8 1000    # Start
# Press Ctrl+C to stop anytime
./parallel_training.sh 8 1000    # Resume automatically
```

### Multi-Machine Training

```bash
# Machine 1
python3 distributed_training.py worker --id 1 --games 1000

# Machine 2
python3 distributed_training.py worker --id 2 --games 1000

# Merge results
python3 distributed_training.py merge --workers 1,2
```

### Performance Scaling

| Cores | Time (10k games) | Speedup |
| ----- | ---------------- | ------- |
| 1     | 100 hours        | 1×      |
| 4     | 28 hours         | 3.5×    |
| 16    | 8 hours          | 12×     |
| 64    | 2.5 hours        | 40×     |
| 1000  | 10 minutes       | 600×    |

## Key Features

**Hand-Crafted Agent (v2):**

- Particle filtering for opponent tracking
- A\* pathfinding with danger avoidance
- Dynamic carry thresholds
- Dead-end detection
- Strategic capsule hunting

**AI Learning Agent:**

- Q-Learning & Deep Q-Networks
- Experience replay
- Opponent modeling
- Self-play training
- Auto-checkpointing every 10 games

## Files

**Agent Files:**

- `my_team.py` - Original agent
- `my_team_v2.py` - Improved agent (use for competitions)
- `my_team_learning.py` - AI trainable agent

**Training:**

- `parallel_training.sh` - Multi-core training launcher
- `distributed_training.py` - Distributed training system
- `training_framework.py` - Q-Learning & DQN core
- `train_agent.py` - Single-core training script
- `variant_agents.py` - Training opponents

**Tools:**

- `compare_agents.py` - Test agents against each other
- `requirements.txt` - Python dependencies

## Documentation

For detailed training information, see inline help:

```bash
python3 distributed_training.py --help
python3 train_agent.py --help
```

## License

MIT
