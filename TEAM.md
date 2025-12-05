# Team Information

**Course:** EUTOPIA Pacman Capture-the-Flag AI Competition

**Semester:** Fall 2025

**Institution:** University of Ljubljana

**Team name:** Dynamic Duo

**Team members:**

- Matej Breskvar

## Agent Description

Our agent uses a dynamic role-switching architecture where both agents can perform offense and defense, adapting to the game situation in real-time.

### Key Techniques

- **Dynamic Role Switching** - When winning, prioritize defense; when losing, prioritize offense
- **A\* Pathfinding** - Optimal path planning with ghost prediction and dead-end avoidance
- **Particle Filtering** - Bayesian inference for tracking opponent positions when not visible
- **Team Coordination** - Shared state prevents both agents from targeting the same food
- **Capsule Strategy** - Prioritize power pellets when ghosts are nearby, hunt scared ghosts

### Architecture

Both agents are instances of `DynamicAgent` class with different role biases:
- Agent 1: Offense bias (but will defend when winning)
- Agent 2: Defense bias (but will attack when losing)

### Performance

Tested with 82.6% overall win rate in a 960-game tournament against diverse opponents.
