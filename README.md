# Pacman CTF Agent

AI agent for the Pacman Capture-the-Flag competition.

## Features

- **Dynamic Role Switching** - Agents adapt between offense and defense based on game score
- **A\* Pathfinding** - Efficient navigation with ghost avoidance and dead-end detection
- **Particle Filtering** - Probabilistic tracking of opponent positions
- **Team Coordination** - Agents coordinate to avoid conflicts and optimize coverage
- **Capsule Strategy** - Smart power-up usage and scared ghost hunting

## Files

| File | Description |
|------|-------------|
| `my_team.py` | Main agent implementation |
| `TEAM.md` | Team information |
| `requirements.txt` | Python dependencies |

## Usage

This agent is designed to work with the [pacman-contest](https://github.com/aig-upf/pacman-contest) framework.

```bash
# Run a game
cd pacman-contest
python runner.py --red agents/team_name_1 --blue /path/to/this/repo
```

## Strategy

The agent uses a unified architecture where both team members can attack and defend:

- **When winning**: Both agents prioritize defense to protect the lead
- **When losing**: Both agents prioritize offense to catch up
- **Tie/close game**: One attacks, one defends based on role bias

This dynamic approach achieved an **82.6% win rate** in tournament testing against diverse opponents.
