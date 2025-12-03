"""
Enhanced Pacman Capture-the-Flag Agent (Version 2)
Team: Madmadam

This agent implements an optimized architecture with:
- Dynamic role assignment (offensive/defensive)
- State machine for behavior management
- Optimized particle filtering for enemy tracking
- Team coordination and shared knowledge
- Performance safeguards to prevent timeouts
- Advanced offensive and defensive strategies
"""

import random
import time
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point, Counter, PriorityQueue, manhattan_distance

# Team configuration
def create_team(first_index, second_index, is_red,
               first='OffensiveRevisedAgent', second='DefensiveRevisedAgent', num_training=0):
    """
    Creates a team with one offensive and one defensive agent
    """
    return [eval(first)(first_index), eval(second)(second_index)]


class AgentState:
    """Enumeration for agent behavior states"""
    ATTACKING = 'attacking'
    RETURNING = 'returning'
    DEFENDING = 'defending'
    CAPSULE_HUNTING = 'capsule_hunting'
    SCARED_EVADING = 'scared_evading'


class SharedKnowledge:
    """
    Shared knowledge structure for team coordination
    Stored as class variables to persist across agent instances
    """
    # Particle filters for enemy tracking {opponent_index: [particles]}
    particles = {}
    
    # Enemy role inference {opponent_index: 'attacking'/'defending'}
    enemy_roles = {}
    
    # Capsule timers {team: scared_timer}
    scared_timers = {'red': 0, 'blue': 0}
    
    # Last known enemy positions {opponent_index: (x, y)}
    last_known_positions = {}
    
    # Food deposit history [(turn, food_count)]
    food_history = []
    
    # Role assignments {agent_index: 'offense'/'defense'}
    role_assignments = {}
    
    # Position history for deadlock detection {agent_index: deque of positions}
    position_history = defaultdict(lambda: deque(maxlen=10))
    
    # Chase cooldown {(agent_index, enemy_index): turns_remaining}
    chase_cooldowns = defaultdict(int)
    
    # Turn counter
    turn_count = 0
    
    @classmethod
    def reset(cls):
        """Reset all shared knowledge (called at game start)"""
        cls.particles = {}
        cls.enemy_roles = {}
        cls.scared_timers = {'red': 0, 'blue': 0}
        cls.last_known_positions = {}
        cls.food_history = []
        cls.role_assignments = {}
        cls.position_history = defaultdict(lambda: deque(maxlen=10))
        cls.chase_cooldowns = defaultdict(int)
        cls.turn_count = 0


class BaseAgent(CaptureAgent):
    """
    Base agent class with shared utilities and optimized helper methods
    """
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.state = AgentState.ATTACKING
        self.start_time = 0
        
        # Precomputed static data (filled in registerInitialState)
        self.border_positions = []
        self.patrol_points = []
        self.dead_ends = set()
        self.food_clusters = []
        self.escape_routes = {}
        
        # Cache for dynamic computations
        self.cached_food_distances = {}
        self.last_food_count = 0
        
    def register_initial_state(self, game_state):
        """
        Initialize agent with precomputation (15 second budget)
        """
        CaptureAgent.register_initial_state(self, game_state)
        
        # Reset shared knowledge at game start
        if self.index == min(self.get_team(game_state)):
            SharedKnowledge.reset()
        
        # Precompute static data
        self._precompute_map_data(game_state)
        
        # Initialize particles for all opponents
        self._initialize_particles(game_state)
        
        # Set initial role
        self._assign_initial_role(game_state)
        
    def _precompute_map_data(self, game_state):
        """Precompute static map information"""
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        
        # Find border positions (middle line of map)
        mid_x = width // 2
        self.border_positions = []
        
        if self.red:
            border_x = mid_x - 1
        else:
            border_x = mid_x
            
        for y in range(height):
            if not walls[border_x][y]:
                self.border_positions.append((border_x, y))
        
        # Create patrol points (evenly spaced border positions)
        patrol_spacing = max(1, len(self.border_positions) // 5)
        self.patrol_points = [self.border_positions[i] 
                             for i in range(0, len(self.border_positions), patrol_spacing)]
        
        # Detect dead ends (positions with only one non-wall neighbor)
        self.dead_ends = self._find_dead_ends(game_state)
        
        # Cluster food positions
        self.food_clusters = self._cluster_food(game_state)
        
    def _find_dead_ends(self, game_state) -> Set[Tuple[int, int]]:
        """Find all dead-end positions on the map"""
        walls = game_state.get_walls()
        dead_ends = set()
        
        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    neighbors = [(x+dx, y+dy) for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]]
                    non_wall_neighbors = sum(1 for nx, ny in neighbors 
                                            if 0 <= nx < walls.width and 
                                            0 <= ny < walls.height and 
                                            not walls[nx][ny])
                    if non_wall_neighbors == 1:
                        dead_ends.add((x, y))
        
        return dead_ends
    
    def _cluster_food(self, game_state) -> List[List[Tuple[int, int]]]:
        """
        Cluster food positions into connected components
        Returns list of food clusters (each cluster is a list of positions)
        """
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return []
        
        # BFS to find connected components
        visited = set()
        clusters = []
        
        for food in food_list:
            if food in visited:
                continue
                
            cluster = []
            queue = [food]
            visited.add(food)
            
            while queue:
                pos = queue.pop(0)
                cluster.append(pos)
                
                # Check adjacent food (within distance 3)
                for other_food in food_list:
                    if other_food not in visited:
                        if manhattan_distance(pos, other_food) <= 3:
                            visited.add(other_food)
                            queue.append(other_food)
            
            clusters.append(cluster)
        
        return clusters
    
    def _initialize_particles(self, game_state):
        """Initialize particle filter for each opponent"""
        opponents = self.get_opponents(game_state)
        
        for opponent in opponents:
            if opponent not in SharedKnowledge.particles:
                # Initialize particles uniformly on opponent's side
                SharedKnowledge.particles[opponent] = self._create_uniform_particles(
                    game_state, opponent
                )
    
    def _create_uniform_particles(self, game_state, agent_index: int, 
                                  num_particles: int = 20) -> List[Tuple[int, int]]:
        """Create uniformly distributed particles on agent's home side"""
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        
        # Determine which side to place particles
        if self.red:
            x_range = range(width // 2, width)
        else:
            x_range = range(0, width // 2)
        
        valid_positions = []
        for x in x_range:
            for y in range(height):
                if not walls[x][y]:
                    valid_positions.append((x, y))
        
        return [random.choice(valid_positions) for _ in range(num_particles)]
    
    def _assign_initial_role(self, game_state):
        """Assign initial role based on agent index"""
        # Override in subclasses
        pass
    
    def choose_action(self, game_state):
        """
        Main action selection with computation safeguards
        """
        self.start_time = time.time()
        
        try:
            # Update shared knowledge
            self._update_shared_knowledge(game_state)
            
            # Update particle filters (with time limit)
            if time.time() - self.start_time < 0.3:
                self._update_particles(game_state)
            
            # Update position history for deadlock detection
            my_pos = game_state.get_agent_position(self.index)
            SharedKnowledge.position_history[self.index].append(my_pos)
            
            # Check for deadlock and inject randomness if needed
            if self._is_in_deadlock():
                if random.random() < 0.3:
                    legal_actions = game_state.get_legal_actions(self.index)
                    legal_actions.remove(Directions.STOP) if Directions.STOP in legal_actions else None
                    return random.choice(legal_actions) if legal_actions else Directions.STOP
            
            # Get action from subclass strategy
            action = self._get_strategic_action(game_state)
            
            # Check computation time
            elapsed = time.time() - self.start_time
            if elapsed > 0.8:
                print(f"Warning: Agent {self.index} took {elapsed:.2f}s")
            
            return action
            
        except Exception as e:
            print(f"Error in agent {self.index}: {e}")
            # Fallback: move toward home
            return self._get_safe_fallback_action(game_state)
    
    def _update_shared_knowledge(self, game_state):
        """Update shared knowledge structures"""
        SharedKnowledge.turn_count += 1
        
        # Update scared timers
        team = 'red' if self.red else 'blue'
        if SharedKnowledge.scared_timers[team] > 0:
            SharedKnowledge.scared_timers[team] -= 1
        
        # Update chase cooldowns
        for key in list(SharedKnowledge.chase_cooldowns.keys()):
            SharedKnowledge.chase_cooldowns[key] -= 1
            if SharedKnowledge.chase_cooldowns[key] <= 0:
                del SharedKnowledge.chase_cooldowns[key]
        
        # Track food history
        food_left = len(self.get_food(game_state).as_list())
        if food_left != self.last_food_count:
            SharedKnowledge.food_history.append((SharedKnowledge.turn_count, food_left))
            self.last_food_count = food_left
    
    def _update_particles(self, game_state):
        """Update particle filters for all opponents"""
        opponents = self.get_opponents(game_state)
        noisy_distances = game_state.get_agent_distances()
        
        for opponent in opponents:
            # Get exact position if visible
            exact_pos = game_state.get_agent_position(opponent)
            
            if exact_pos is not None:
                # Reset particles to exact position
                SharedKnowledge.particles[opponent] = [exact_pos] * 20
                SharedKnowledge.last_known_positions[opponent] = exact_pos
            else:
                # Update particles based on noisy distance
                self._update_particle_filter(game_state, opponent, noisy_distances[opponent])
    
    def _update_particle_filter(self, game_state, opponent_index: int, noisy_distance: int):
        """Update particle filter for one opponent using noisy distance"""
        particles = SharedKnowledge.particles.get(opponent_index, [])
        if not particles:
            particles = self._create_uniform_particles(game_state, opponent_index)
        
        my_pos = game_state.get_agent_position(self.index)
        walls = game_state.get_walls()
        
        # Predict: simulate opponent movement
        new_particles = []
        for particle in particles:
            x, y = particle
            possible_positions = [(x, y)]  # Can stay still
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                    possible_positions.append((nx, ny))
            new_particles.append(random.choice(possible_positions))
        
        # Observe: weight by noisy distance
        weights = []
        for particle in new_particles:
            true_distance = manhattan_distance(my_pos, particle)
            # Probability decreases with distance from noisy observation
            diff = abs(true_distance - noisy_distance)
            weight = max(0.1, 10.0 - diff)  # Simple weight function
            weights.append(weight)
        
        # Resample based on weights
        total_weight = sum(weights)
        if total_weight == 0:
            # If all weights are 0, reset to uniform
            SharedKnowledge.particles[opponent_index] = self._create_uniform_particles(
                game_state, opponent_index
            )
        else:
            weights = [w / total_weight for w in weights]
            SharedKnowledge.particles[opponent_index] = [
                new_particles[self._weighted_sample(weights)] for _ in range(20)
            ]
    
    def _weighted_sample(self, weights: List[float]) -> int:
        """Sample index based on weights"""
        r = random.random()
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return i
        return len(weights) - 1
    
    def _is_in_deadlock(self) -> bool:
        """Detect if agent is in a deadlock (oscillating positions)"""
        history = SharedKnowledge.position_history[self.index]
        if len(history) < 6:
            return False
        
        # Check if oscillating between 2-3 positions
        recent = list(history)[-6:]
        unique_positions = set(recent)
        
        if len(unique_positions) <= 2:
            return True
        
        return False
    
    def _get_strategic_action(self, game_state):
        """Override in subclass to implement strategy"""
        raise NotImplementedError
    
    def _get_safe_fallback_action(self, game_state):
        """Safe fallback action when computation fails or times out"""
        my_pos = game_state.get_agent_position(self.index)
        legal_actions = game_state.get_legal_actions(self.index)
        
        # Remove STOP to keep moving
        if Directions.STOP in legal_actions and len(legal_actions) > 1:
            legal_actions.remove(Directions.STOP)
        
        # Move toward nearest border position
        best_action = legal_actions[0]
        best_dist = float('inf')
        
        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_position(self.index)
            dist = min(self.get_maze_distance(next_pos, border) 
                      for border in self.border_positions)
            if dist < best_dist:
                best_dist = dist
                best_action = action
        
        return best_action
    
    def get_successor(self, game_state, action):
        """Find the successor game state after taking an action"""
        successor = game_state.generate_successor(self.index, action)
        return successor
    
    def evaluate_action(self, game_state, action):
        """Evaluate an action (override in subclass)"""
        return 0
    
    def _get_enemy_positions(self, game_state) -> Dict[int, Tuple[int, int]]:
        """
        Get estimated positions of all enemies
        Returns dict {opponent_index: (x, y)}
        """
        positions = {}
        opponents = self.get_opponents(game_state)
        
        for opponent in opponents:
            exact_pos = game_state.get_agent_position(opponent)
            if exact_pos is not None:
                positions[opponent] = exact_pos
            else:
                # Use most likely particle position (mode of distribution)
                particles = SharedKnowledge.particles.get(opponent, [])
                if particles:
                    # Find most common position
                    position_counts = {}
                    for p in particles:
                        position_counts[p] = position_counts.get(p, 0) + 1
                    most_common_pos = max(position_counts.keys(), key=lambda k: position_counts[k])
                    positions[opponent] = most_common_pos
                elif opponent in SharedKnowledge.last_known_positions:
                    positions[opponent] = SharedKnowledge.last_known_positions[opponent]
        
        return positions
    
    def _bfs_to_target(self, game_state, start: Tuple[int, int], 
                      targets: List[Tuple[int, int]], 
                      avoid_positions: Set[Tuple[int, int]] = None) -> Optional[List[Tuple[int, int]]]:
        """
        BFS pathfinding to nearest target
        Returns path as list of positions, or None if no path exists
        """
        if avoid_positions is None:
            avoid_positions = set()
        
        walls = game_state.get_walls()
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            pos, path = queue.popleft()
            
            if pos in targets:
                return path + [pos]
            
            x, y = pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                
                if (0 <= nx < walls.width and 0 <= ny < walls.height and 
                    not walls[nx][ny] and next_pos not in visited and 
                    next_pos not in avoid_positions):
                    
                    visited.add(next_pos)
                    queue.append((next_pos, path + [pos]))
        
        return None
    
    def _a_star_search(self, game_state, start: Tuple[int, int], 
                      goal: Tuple[int, int],
                      avoid_positions: Set[Tuple[int, int]] = None,
                      heuristic_weight: float = 1.0) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding with configurable heuristic weight
        Returns path as list of positions
        """
        if avoid_positions is None:
            avoid_positions = set()
        
        walls = game_state.get_walls()
        
        frontier = PriorityQueue()
        frontier.push((start, []), 0)
        visited = {start: 0}
        
        max_iterations = 500  # Limit search to prevent timeout
        iterations = 0
        
        while not frontier.is_empty() and iterations < max_iterations:
            iterations += 1
            
            pos, path = frontier.pop()
            
            if pos == goal:
                return path + [pos]
            
            x, y = pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                
                if (0 <= nx < walls.width and 0 <= ny < walls.height and 
                    not walls[nx][ny] and next_pos not in avoid_positions):
                    
                    new_cost = len(path) + 1
                    
                    if next_pos not in visited or new_cost < visited[next_pos]:
                        visited[next_pos] = new_cost
                        heuristic = manhattan_distance(next_pos, goal) * heuristic_weight
                        priority = new_cost + heuristic
                        frontier.push((next_pos, path + [pos]), priority)
        
        return None


class OffensiveRevisedAgent(BaseAgent):
    """
    Optimized offensive agent with smart target selection and safe returns
    """
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.carry_threshold = 5  # Dynamic threshold for returning
        self.target_food = None
        self.target_cluster = None
        
    def _assign_initial_role(self, game_state):
        """Offensive agent role"""
        SharedKnowledge.role_assignments[self.index] = 'offense'
    
    def _get_strategic_action(self, game_state):
        """
        Main offensive strategy with state machine
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        carrying = my_state.num_carrying
        
        # Update carry threshold based on game state
        self._update_carry_threshold(game_state)
        
        # State transitions
        if my_state.is_pacman:
            # We're in enemy territory
            enemies_nearby = self._get_nearby_enemies(game_state, distance=5)
            scared_enemies = [e for e in enemies_nearby 
                            if game_state.get_agent_state(e).scared_timer > 0]
            dangerous_enemies = [e for e in enemies_nearby 
                               if game_state.get_agent_state(e).scared_timer == 0]
            
            if dangerous_enemies and carrying > 0:
                # Need to return with food
                self.state = AgentState.RETURNING
            elif carrying >= self.carry_threshold:
                # Reached carry threshold
                self.state = AgentState.RETURNING
            else:
                # Continue attacking
                capsules = self.get_capsules(game_state)
                if capsules and len(dangerous_enemies) > 0:
                    self.state = AgentState.CAPSULE_HUNTING
                else:
                    self.state = AgentState.ATTACKING
        else:
            # We're on home side
            self.state = AgentState.ATTACKING
        
        # Execute state-specific behavior
        if self.state == AgentState.RETURNING:
            return self._return_home(game_state)
        elif self.state == AgentState.CAPSULE_HUNTING:
            return self._hunt_capsule(game_state)
        else:  # ATTACKING
            return self._attack(game_state)
    
    def _update_carry_threshold(self, game_state):
        """Dynamically adjust carry threshold based on game state"""
        food_left = len(self.get_food(game_state).as_list())
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        
        # More conservative thresholds for safer play
        # Early game: carry moderate (4-5)
        # Late game: carry less (2-3)
        if food_left > 15:
            base_threshold = 4
        elif food_left > 8:
            base_threshold = 3
        else:
            base_threshold = 2
        
        # Adjust based on enemy positions - be more cautious
        if my_state.is_pacman:
            enemies = self._get_nearby_enemies(game_state, distance=5)
            dangerous_enemies = [e for e in enemies 
                               if not game_state.get_agent_state(e).is_pacman 
                               and game_state.get_agent_state(e).scared_timer == 0]
            if dangerous_enemies:
                # Return immediately if carrying anything and ghost nearby
                base_threshold = max(1, base_threshold - 2)
        
        # Also consider current score - if winning, be more conservative
        score = self.get_score(game_state)
        if score > 5:
            base_threshold = max(2, base_threshold - 1)
        
        self.carry_threshold = base_threshold
    
    def _get_nearby_enemies(self, game_state, distance: int = 5) -> List[int]:
        """Get enemy indices within specified distance"""
        my_pos = game_state.get_agent_position(self.index)
        enemies = []
        
        for opponent in self.get_opponents(game_state):
            enemy_pos = game_state.get_agent_position(opponent)
            if enemy_pos is not None:
                if manhattan_distance(my_pos, enemy_pos) <= distance:
                    enemies.append(opponent)
        
        return enemies
    
    def _attack(self, game_state):
        """Attack strategy: select and navigate to food"""
        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        
        if not food_list:
            # No food left, go to border
            return self._return_home(game_state)
        
        # Select best food target
        target_food = self._select_best_food(game_state, food_list)
        
        if target_food is None:
            # Fallback to closest food
            target_food = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        
        # Get safe path to target
        enemy_positions = self._get_enemy_positions(game_state)
        dangerous_positions = self._get_dangerous_positions(game_state, enemy_positions)
        
        path = self._a_star_search(game_state, my_pos, target_food, 
                                   avoid_positions=dangerous_positions,
                                   heuristic_weight=1.2)
        
        if path and len(path) > 1:
            next_pos = path[1]
            return self._get_action_to_position(game_state, next_pos)
        else:
            # Fallback: move toward target
            return self._move_toward(game_state, target_food)
    
    def _select_best_food(self, game_state, food_list: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Select best food target based on safety and distance
        """
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        
        enemy_positions = self._get_enemy_positions(game_state)
        
        best_food = None
        best_score = float('-inf')
        
        # Get dangerous enemy positions (ghosts that aren't scared)
        dangerous_enemies = {}
        for idx, pos in enemy_positions.items():
            enemy_state = game_state.get_agent_state(idx)
            if not enemy_state.is_pacman and enemy_state.scared_timer == 0:
                dangerous_enemies[idx] = pos
        
        for food in food_list[:25]:  # Slightly increased evaluation range
            # Distance to food
            food_dist = self.get_maze_distance(my_pos, food)
            
            # Skip if too far when carrying a lot
            if carrying > 2 and food_dist > 10:
                continue
            
            # Distance back to border from food
            border_dist = min(self.get_maze_distance(food, border) 
                            for border in self.border_positions[:10])
            
            # Enemy proximity (use maze distance for dangerous enemies)
            min_enemy_dist = float('inf')
            for enemy_idx, enemy_pos in dangerous_enemies.items():
                enemy_dist = self.get_maze_distance(food, enemy_pos)
                min_enemy_dist = min(min_enemy_dist, enemy_dist)
            
            # Skip food too close to dangerous enemies
            if min_enemy_dist < 3:
                continue
            
            # Dead-end penalty (much higher)
            dead_end_penalty = 0
            if food in self.dead_ends:
                dead_end_penalty = 100
                # Skip dead ends if carrying anything
                if carrying > 0:
                    continue
            
            # Score calculation with better weighting
            score = (
                -food_dist * 1.5                   # Prefer closer food
                - border_dist * (carrying + 1)     # Return path increasingly important
                + min_enemy_dist * 5               # Strongly avoid enemies
                - dead_end_penalty                 # Strongly avoid dead ends
            )
            
            if score > best_score:
                best_score = score
                best_food = food
        
        return best_food
    
    def _get_dangerous_positions(self, game_state, enemy_positions: Dict[int, Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Get positions to avoid (near enemy ghosts)"""
        dangerous = set()
        walls = game_state.get_walls()
        
        for opponent_idx, enemy_pos in enemy_positions.items():
            enemy_state = game_state.get_agent_state(opponent_idx)
            
            # Only avoid if enemy is ghost and not scared
            if not enemy_state.is_pacman and enemy_state.scared_timer == 0:
                # Add positions within distance 3 of enemy (increased range)
                x, y = enemy_pos
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = x + dx, y + dy
                        if abs(dx) + abs(dy) <= 3:
                            if 0 <= nx < walls.width and 0 <= ny < walls.height:
                                dangerous.add((nx, ny))
        
        return dangerous
    
    def _return_home(self, game_state):
        """Return to home territory safely"""
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        # Find best border positions - prefer those far from enemies
        enemy_positions = self._get_enemy_positions(game_state)
        
        border_scores = []
        for border in self.border_positions:
            dist_to_border = self.get_maze_distance(my_pos, border)
            
            # Distance from enemies
            min_enemy_dist = float('inf')
            for enemy_pos in enemy_positions.values():
                enemy_dist = self.get_maze_distance(border, enemy_pos)
                min_enemy_dist = min(min_enemy_dist, enemy_dist)
            
            score = -dist_to_border + min_enemy_dist * 2
            border_scores.append((score, border))
        
        border_scores.sort(reverse=True)
        target_border = border_scores[0][1]
        
        # Get safe path
        dangerous_positions = self._get_dangerous_positions(game_state, enemy_positions)
        
        # Try multiple border positions if first is blocked
        for _, border in border_scores[:3]:
            path = self._bfs_to_target(game_state, my_pos, [border], 
                                       avoid_positions=dangerous_positions)
            if path and len(path) > 1:
                next_pos = path[1]
                return self._get_action_to_position(game_state, next_pos)
        
        # Emergency: just move toward safest border
        return self._move_toward(game_state, target_border)
    
    def _hunt_capsule(self, game_state):
        """Navigate to and eat power capsule"""
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        capsules = self.get_capsules(game_state)
        
        if not capsules:
            return self._attack(game_state)
        
        # Check if capsule hunt is really worth it
        enemies_nearby = self._get_nearby_enemies(game_state, distance=5)
        if not enemies_nearby and my_state.num_carrying < 2:
            # No immediate threat and not carrying much - just attack instead
            return self._attack(game_state)
        
        # Go to nearest capsule
        target_capsule = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
        capsule_dist = self.get_maze_distance(my_pos, target_capsule)
        
        # If capsule is too far and we're carrying food, return instead
        if capsule_dist > 8 and my_state.num_carrying > 0:
            return self._return_home(game_state)
        
        path = self._a_star_search(game_state, my_pos, target_capsule)
        
        if path and len(path) > 1:
            next_pos = path[1]
            action = self._get_action_to_position(game_state, next_pos)
            
            # Update scared timer when we eat capsule
            if next_pos == target_capsule:
                team = 'red' if self.red else 'blue'
                SharedKnowledge.scared_timers[team] = 40
            
            return action
        else:
            return self._move_toward(game_state, target_capsule)
    
    def _move_toward(self, game_state, target: Tuple[int, int]):
        """Simple greedy move toward target"""
        my_pos = game_state.get_agent_position(self.index)
        legal_actions = game_state.get_legal_actions(self.index)
        
        best_action = legal_actions[0]
        best_dist = float('inf')
        
        for action in legal_actions:
            if action == Directions.STOP:
                continue
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(next_pos, target)
            
            if dist < best_dist:
                best_dist = dist
                best_action = action
        
        return best_action
    
    def _get_action_to_position(self, game_state, target_pos: Tuple[int, int]):
        """Get action that moves to target position"""
        my_pos = game_state.get_agent_position(self.index)
        legal_actions = game_state.get_legal_actions(self.index)
        
        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_position(self.index)
            if next_pos == target_pos:
                return action
        
        # Fallback
        return legal_actions[0]


class DefensiveRevisedAgent(BaseAgent):
    """
    Optimized defensive agent with zone patrol and coordinated chase
    """
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.patrol_index = 0
        self.turns_without_invaders = 0
        self.assigned_zone = 'top'  # or 'bottom'
        
    def _assign_initial_role(self, game_state):
        """Defensive agent role"""
        SharedKnowledge.role_assignments[self.index] = 'defense'
        
        # Assign patrol zone based on index
        team_indices = self.get_team(game_state)
        if self.index == min(team_indices):
            self.assigned_zone = 'bottom'
        else:
            self.assigned_zone = 'top'
    
    def _get_strategic_action(self, game_state):
        """
        Main defensive strategy
        """
        my_state = game_state.get_agent_state(self.index)
        
        # Check if we're scared
        if my_state.scared_timer > 0:
            self.state = AgentState.SCARED_EVADING
            return self._evade_invaders(game_state)
        
        # Check for invaders
        invaders = self._get_invaders(game_state)
        
        if invaders:
            self.turns_without_invaders = 0
            self.state = AgentState.DEFENDING
            return self._chase_invader(game_state, invaders)
        else:
            self.turns_without_invaders += 1
            
            # Check if we should help on offense
            score = self.get_score(game_state)
            food_left = len(self.get_food(game_state).as_list())
            
            # If losing badly or very few food left, switch to offense
            if score < -10 or (food_left <= 5 and self.turns_without_invaders > 10):
                return self._light_offense(game_state)
            
            # If no invaders for 30 turns and not winning big, scout
            if self.turns_without_invaders > 30 and score < 10:
                if food_left > 5:
                    return self._light_offense(game_state)
            
            # Normal patrol
            return self._patrol(game_state)
    
    def _get_invaders(self, game_state) -> List[int]:
        """Get list of opponent indices that are invading our territory"""
        invaders = []
        opponents = self.get_opponents(game_state)
        
        for opponent in opponents:
            opponent_state = game_state.get_agent_state(opponent)
            if opponent_state.is_pacman:
                invaders.append(opponent)
        
        return invaders
    
    def _patrol(self, game_state):
        """Patrol assigned zone of the border"""
        my_pos = game_state.get_agent_position(self.index)
        
        # Filter patrol points by zone
        zone_patrol_points = self._get_zone_patrol_points()
        
        if not zone_patrol_points:
            zone_patrol_points = self.patrol_points
        
        # Move to next patrol point
        target = zone_patrol_points[self.patrol_index % len(zone_patrol_points)]
        
        # Check if reached patrol point
        if my_pos == target or self.get_maze_distance(my_pos, target) <= 1:
            self.patrol_index += 1
            target = zone_patrol_points[self.patrol_index % len(zone_patrol_points)]
        
        return self._move_toward(game_state, target)
    
    def _get_zone_patrol_points(self) -> List[Tuple[int, int]]:
        """Get patrol points for assigned zone"""
        if not self.patrol_points:
            return []
        
        # Sort patrol points by y coordinate
        sorted_points = sorted(self.patrol_points, key=lambda p: p[1])
        mid = len(sorted_points) // 2
        
        if self.assigned_zone == 'bottom':
            return sorted_points[:mid] if mid > 0 else sorted_points
        else:  # top
            return sorted_points[mid:] if mid > 0 else sorted_points
    
    def _chase_invader(self, game_state, invaders: List[int]):
        """Chase and intercept invaders"""
        my_pos = game_state.get_agent_position(self.index)
        
        # Prioritize invader with most food
        target_invader = max(invaders, 
                            key=lambda i: game_state.get_agent_state(i).num_carrying)
        
        invader_pos = game_state.get_agent_position(target_invader)
        
        if invader_pos is None:
            # Use particle estimate
            enemy_positions = self._get_enemy_positions(game_state)
            invader_pos = enemy_positions.get(target_invader)
        
        if invader_pos is None:
            # Fallback to patrol
            return self._patrol(game_state)
        
        # Check chase cooldown to prevent infinite mutual chase
        cooldown_key = (self.index, target_invader)
        if SharedKnowledge.chase_cooldowns.get(cooldown_key, 0) > 0:
            # Switch to different target or patrol
            other_invaders = [i for i in invaders if i != target_invader]
            if other_invaders:
                target_invader = other_invaders[0]
                invader_pos = game_state.get_agent_position(target_invader)
            else:
                return self._patrol(game_state)
        
        # Predict escape path and intercept
        predicted_target = self._predict_invader_escape(game_state, target_invader, invader_pos)
        
        # Chase toward predicted position
        path = self._a_star_search(game_state, my_pos, predicted_target)
        
        if path and len(path) > 1:
            next_pos = path[1]
            return self._get_action_to_position(game_state, next_pos)
        else:
            return self._move_toward(game_state, invader_pos)
    
    def _predict_invader_escape(self, game_state, invader_index: int, 
                                invader_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Predict where invader will try to escape
        Returns predicted interception point
        """
        # Assume invader will go to nearest border position
        nearest_border = min(self.border_positions, 
                            key=lambda b: manhattan_distance(invader_pos, b))
        
        return nearest_border
    
    def _evade_invaders(self, game_state):
        """Evade when scared"""
        my_pos = game_state.get_agent_position(self.index)
        
        # Get invader positions
        invaders = self._get_invaders(game_state)
        invader_positions = set()
        
        for invader in invaders:
            pos = game_state.get_agent_position(invader)
            if pos is not None:
                invader_positions.add(pos)
        
        # Find safe position (away from invaders)
        legal_actions = game_state.get_legal_actions(self.index)
        best_action = legal_actions[0]
        best_dist = -float('inf')
        
        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_position(self.index)
            
            # Calculate minimum distance to invaders
            if invader_positions:
                min_dist = min(manhattan_distance(next_pos, inv_pos) 
                              for inv_pos in invader_positions)
            else:
                min_dist = 0
            
            if min_dist > best_dist:
                best_dist = min_dist
                best_action = action
        
        return best_action
    
    def _light_offense(self, game_state):
        """Light offensive scouting when no invaders"""
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        food_list = self.get_food(game_state).as_list()
        
        if not food_list:
            return self._patrol(game_state)
        
        # If carrying food, return home
        if my_state.num_carrying > 0:
            return self._return_home_defensive(game_state)
        
        # Find safe food near border
        enemy_positions = self._get_enemy_positions(game_state)
        safe_food = []
        
        for food in food_list[:15]:
            border_dist = min(self.get_maze_distance(food, b) for b in self.border_positions[:5])
            if border_dist <= 6:  # Slightly deeper
                # Check enemy distance
                min_enemy_dist = float('inf')
                for enemy_pos in enemy_positions.values():
                    min_enemy_dist = min(min_enemy_dist, self.get_maze_distance(food, enemy_pos))
                
                if min_enemy_dist > 4:  # Only if safe
                    safe_food.append(food)
        
        if safe_food:
            target_food = min(safe_food, key=lambda f: self.get_maze_distance(my_pos, f))
            return self._move_toward(game_state, target_food)
        else:
            # No safe food, patrol
            return self._patrol(game_state)
    
    def _return_home_defensive(self, game_state):
        """Return home when defensive agent has food"""
        my_pos = game_state.get_agent_position(self.index)
        target_border = min(self.border_positions, 
                           key=lambda b: self.get_maze_distance(my_pos, b))
        return self._move_toward(game_state, target_border)
    
    def _move_toward(self, game_state, target: Tuple[int, int]):
        """Simple greedy move toward target"""
        my_pos = game_state.get_agent_position(self.index)
        legal_actions = game_state.get_legal_actions(self.index)
        
        # Penalize STOP action heavily
        best_action = None
        best_dist = float('inf')
        
        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(next_pos, target)
            
            # Add penalty for STOP
            if action == Directions.STOP:
                dist += 100
            
            if dist < best_dist:
                best_dist = dist
                best_action = action
        
        return best_action if best_action else legal_actions[0]
    
    def _get_action_to_position(self, game_state, target_pos: Tuple[int, int]):
        """Get action that moves to target position"""
        my_pos = game_state.get_agent_position(self.index)
        legal_actions = game_state.get_legal_actions(self.index)
        
        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_position(self.index)
            if next_pos == target_pos:
                return action
        
        # Fallback
        return legal_actions[0] if legal_actions else Directions.STOP
