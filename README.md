# Pacman Agent - Advanced AI Competition Entry

Elite-tier Pacman Capture-the-Flag agent for the EUTOPIA AI competition.

## Performance

**Win Rate vs Baseline Team:**

- defaultCapture: **95%** (19/20 wins)
- mediumCapture: **90%** (18/20 wins)
- Overall: **90-95%** win rate

## Features

This agent implements advanced AI techniques:

- **Particle Filtering** for opponent position tracking
- **A\* Pathfinding** with ghost avoidance
- **Dead-End Detection** and avoidance
- **Map-Size Adaptive Behavior** (tiny/small/medium/large maps)
- **Dynamic Return Thresholds** based on game state
- **Smart Food Selection** with safety evaluation
- **Predictive Defensive Interception**
- **Strategic Capsule Usage**
- **Emergency Escape Logic**

## Setting up the environment

1. Clone this repository: `git clone git@github.com:matejbreskvar/pacman-agent.git`
2. Go into pacman-agent folder: `cd pacman-agent/`
3. Run `git submodule update --init --remote` to pull the pacman-contest framework
4. Create a virtual environment: `python3.8 -m venv venv`
5. Activate the virtual environment: `source venv/bin/activate`
6. Install requirements:
   - `cd pacman-contest/`
   - `pip install -r requirements.txt`
   - `pip install -e .`

## Running the agent

To test against baseline team:

```bash
cd pacman-contest/src/contest/
python capture.py -r ../../../my_team.py -b baseline_team
```

To run without graphics (faster):

```bash
python capture.py -r ../../../my_team.py -b baseline_team -q -n 10
```

To test on different layouts:

```bash
python capture.py -r ../../../my_team.py -b baseline_team -l mediumCapture
```

## Agent Structure

- `OffensiveAgent`: Focuses on food collection with smart pathfinding
- `DefensiveAgent`: Guards territory with predictive interception
- Both agents use particle filtering and adapt to map size
