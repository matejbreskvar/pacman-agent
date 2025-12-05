# Pacman AI Agent

Advanced Pacman Capture-the-Flag agent with AI training capabilities.

## Quick Start

### 1. Setup

```bash
git clone <repo-url>
cd pacman-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Test Agent

```bash
python3 compare_agents.py
```

### 3. Train Agent

**Linux/macOS:**

```bash
./train.sh                    # Default: 8 workers, 100 games each
./train.sh 16 500             # 16 workers, 500 games each
./train.sh 8 100 --quick      # Quick mode (fewer opponents)
```

**Windows (PowerShell):**

```powershell
.\train.ps1                   # Default: 8 workers, 100 games each
.\train.ps1 16 500            # 16 workers, 500 games each
.\train.ps1 -NumWorkers 8 -GamesPerWorker 100 -Quick
```

## Available Agents

| Agent                  | Description           | Use For                      |
| ---------------------- | --------------------- | ---------------------------- |
| `my_team_hybrid.py` ⭐ | RL + Advanced Search  | Competition (trains to 80%+) |
| `my_team_v2.py`        | Hand-crafted advanced | Fallback / Opponent          |
| `my_team.py`           | Original baseline     | Training opponent            |
| `my_team_learning.py`  | Pure learning agent   | Experimental                 |

## Training System

The training script (`train.sh`) trains your agent against a diverse pool of opponents:

**Built-in opponents:**

- `baseline` - Basic reflex agent
- `self-play` - Train against itself

**Our agents:**

- `v2` - Your hand-crafted advanced agent
- `original` - Your first version agent

**Fork agents (in `opponents/` folder):**

- Angela-Dimi, Anonymous-Sisters, Packstreet
- emai, pacman_comp, Adaptive

### Training Output

After training, your agent is saved to:

- `offensive_strategy.pkl` - Offensive agent strategy
- `defensive_strategy.pkl` - Defensive agent strategy

### Continue Training

Training is resumable - just run `./train.sh` again to continue from where you left off.

## Competition Settings

Training uses official competition rules:

- **Time Limit**: 1200 moves (300 per agent)
- **Layouts**: Random + official maps
- **Computation**: 1s per move, 15s initial setup
- **Scoring**: Win = 3pts, Tie = 1pt, Loss = 0pts

## Files

**Agent Files:**

- `my_team_hybrid.py` - Main competition agent (RL + hand-crafted)
- `my_team_v2.py` - Hand-crafted advanced agent
- `my_team.py` - Original baseline agent
- `my_team_learning.py` - Pure learning agent

**Training:**

- `train.sh` - Training launcher (Linux/macOS)
- `train.ps1` - Training launcher (Windows PowerShell)
- `training_worker.py` - Training worker (spawned by train scripts)
- `training_framework.py` - Q-Learning core

**Opponents:**

- `opponents/` - Fork agents for diverse training

**Tools:**

- `compare_agents.py` - Test agents against each other
- `requirements.txt` - Python dependencies

## License

MIT
