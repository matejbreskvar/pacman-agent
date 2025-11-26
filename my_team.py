"""
Competitive Pacman Capture-the-Flag Agent
==========================================
A sophisticated multi-agent system for the EUTOPIA Pacman CTF competition.

This implementation uses:
- Particle filtering for opponent tracking
- Dynamic role assignment (offense/defense)
- Advanced heuristic evaluation functions
- Coordinated team behavior
- Strategic capsule usage
"""

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    Creates a team of two agents. One primarily offensive, one primarily defensive.
    Both agents can adapt their behavior based on game state.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class BaseAgent(CaptureAgent):
    """
    Base class with shared functionality for all agents.
    Implements particle filtering, state evaluation, and utility methods.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.walls = None
        self.width = None
        self.height = None
        self.mid_x = None
        self.boundary = None
        self.initial_food_count = 0
        self.initial_defending_food = 0
        self.particles = {}
        self.last_food_defending = []
        self.patrol_points = []
        self.dead_ends = set()  # Cache dead end positions
        self.safe_food = []  # Food that's relatively safe to collect

    def register_initial_state(self, game_state):
        """
        Initialize the agent with game state information.
        Called once at the start of the game (15 second budget).
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # Map dimensions
        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height

        # Find the midline (boundary between sides)
        self.mid_x = self.width // 2
        if self.red:
            self.mid_x -= 1

        # Calculate valid boundary positions
        self.boundary = self._get_boundary_positions()

        # Food-related caching
        self.initial_food_count = len(self.get_food(game_state).as_list())
        self.initial_defending_food = len(self.get_food_you_are_defending(game_state).as_list())

        # Particle filter for opponent tracking (reduced to 50 particles for performance)
        opponent_indices = self.get_opponents(game_state)
        for opponent in opponent_indices:
            initial_pos = game_state.get_initial_agent_position(opponent)
            self.particles[opponent] = [initial_pos] * 50

        # Initialize strategic information
        self.last_food_defending = self.get_food_you_are_defending(game_state).as_list()
        self.patrol_points = self._compute_patrol_points(game_state)
        
        # Compute dead ends for smarter navigation
        self.dead_ends = self._find_dead_ends()
        
        # Determine map size category for adaptive behavior
        map_area = self.width * self.height
        if map_area < 100:
            self.map_size = 'tiny'
        elif map_area < 300:
            self.map_size = 'small'
        elif map_area < 500:
            self.map_size = 'medium'
        else:
            self.map_size = 'large'

    def _find_dead_ends(self):
        """Identify dead-end positions on the map."""
        dead_ends = set()
        for x in range(self.width):
            for y in range(self.height):
                if not self.walls[x][y]:
                    # Count open neighbors
                    neighbors = 0
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if not self.walls[nx][ny]:
                                neighbors += 1
                    # Dead end has only 1 exit
                    if neighbors == 1:
                        dead_ends.add((x, y))
        return dead_ends
    
    def _is_dead_end_nearby(self, pos, depth=3):
        """Check if position is in or near a dead end."""
        if pos in self.dead_ends:
            return True
        # BFS to check nearby positions
        if depth <= 0:
            return False
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = int(pos[0] + dx), int(pos[1] + dy)
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if not self.walls[nx][ny] and (nx, ny) in self.dead_ends:
                    return True
        return False
    
    def _a_star_search(self, start, goal, game_state, avoid_ghosts=True):
        """A* pathfinding that can avoid ghosts."""
        from contest.util import PriorityQueue
        
        frontier = PriorityQueue()
        frontier.push(start, 0)
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        # Get ghost positions if we should avoid them
        ghost_positions = set()
        if avoid_ghosts:
            enemies = self.get_defending_enemies(game_state)
            for _, pos in enemies:
                ghost_positions.add(pos)
                # Add danger zone around ghosts
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        gx, gy = int(pos[0] + dx), int(pos[1] + dy)
                        if 0 <= gx < self.width and 0 <= gy < self.height:
                            if not self.walls[gx][gy]:
                                ghost_positions.add((gx, gy))
        
        while not frontier.isEmpty():
            current = frontier.pop()
            
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for neighbor in self._get_legal_neighbors(current):
                # Higher cost for ghost positions
                new_cost = cost_so_far[current] + 1
                if neighbor in ghost_positions:
                    new_cost += 10  # Penalty for dangerous positions
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.get_maze_distance(neighbor, goal)
                    frontier.push(neighbor, priority)
                    came_from[neighbor] = current
        
        return None  # No path found

    def _get_boundary_positions(self):
        """Get all valid positions along the boundary between territories."""
        boundary = []
        for y in range(1, self.height - 1):
            if not self.walls[self.mid_x][y]:
                boundary.append((self.mid_x, y))
        return boundary

    def _compute_patrol_points(self, game_state):
        """Compute strategic patrol points for defense."""
        food = self.get_food_you_are_defending(game_state).as_list()
        if not food:
            return self.boundary

        # Find food clusters and their centroids
        patrol_points = []
        for pos in self.boundary:
            # Score each boundary position by proximity to defended food
            min_dist = min(self.get_maze_distance(pos, f) for f in food)
            patrol_points.append((pos, min_dist))

        # Sort by distance and take top positions
        patrol_points.sort(key=lambda x: x[1])
        return [p[0] for p in patrol_points[:max(3, len(patrol_points) // 2)]]

    def _get_legal_neighbors(self, pos):
        """Get legal neighboring positions."""
        x, y = int(pos[0]), int(pos[1])
        neighbors = []
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if not self.walls[nx][ny]:
                    neighbors.append((nx, ny))
        return neighbors

    def update_particles(self, game_state, opponent_index):
        """Update particle filter based on observations."""
        particles = self.particles[opponent_index]

        # Get actual position if observable
        actual_pos = game_state.get_agent_position(opponent_index)
        if actual_pos is not None:
            # We can see the opponent, reset particles to actual position
            self.particles[opponent_index] = [actual_pos] * 50
            return

        # Get noisy distance
        noisy_distances = game_state.get_agent_distances()
        if noisy_distances is None:
            return
        noisy_distance = noisy_distances[opponent_index]
        my_pos = game_state.get_agent_position(self.index)

        # Weight particles by how consistent they are with the noisy reading
        weights = []
        for particle in particles:
            true_dist = self.get_maze_distance(my_pos, particle)
            # Probability decreases as the difference increases
            prob = max(0.001, 1.0 - abs(true_dist - noisy_distance) / 10.0)
            weights.append(prob)

        # Resample particles
        total_weight = sum(weights)
        if total_weight == 0:
            # Reset if all weights are zero
            initial_pos = game_state.get_initial_agent_position(opponent_index)
            self.particles[opponent_index] = [initial_pos] * 50
            return

        # Normalize weights
        weights = [w / total_weight for w in weights]

        # Resample (reduced to 50 particles)
        new_particles = []
        for _ in range(50):
            # Select a particle based on weights
            r = random.random()
            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    # Move particle to a neighboring cell
                    particle = particles[i]
                    neighbors = self._get_legal_neighbors(particle)
                    if neighbors:
                        new_particles.append(random.choice(neighbors))
                    else:
                        new_particles.append(particle)
                    break

        self.particles[opponent_index] = new_particles

    def get_opponent_probable_positions(self, opponent_index):
        """Get most likely positions for an opponent."""
        particles = self.particles[opponent_index]
        position_counts = {}
        for p in particles:
            position_counts[p] = position_counts.get(p, 0) + 1

        # Sort by count and return top positions
        sorted_positions = sorted(position_counts.items(), key=lambda x: -x[1])
        return [pos for pos, count in sorted_positions[:5]]

    def is_on_our_side(self, pos):
        """Check if a position is on our side."""
        x = pos[0]
        if self.red:
            return x <= self.mid_x
        else:
            return x > self.mid_x

    def get_successor(self, game_state, action):
        """Compute the next game state after taking an action."""
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def evaluate(self, game_state, action):
        """Evaluate the quality of an action."""
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """Override in subclasses."""
        features = util.Counter()
        return features

    def get_weights(self, game_state, action):
        """Override in subclasses."""
        return {}

    def choose_action(self, game_state):
        """Choose the best action based on evaluation."""
        # Update particle filters
        for opponent in self.get_opponents(game_state):
            self.update_particles(game_state, opponent)

        # EMERGENCY: If carrying food and ghost is very close, flee immediately!
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        carrying = my_state.num_carrying
        
        if carrying > 0:
            enemies = self.get_defending_enemies(game_state)
            if enemies:
                min_ghost_dist = min(self.get_maze_distance(my_pos, pos) for _, pos in enemies)
                if min_ghost_dist <= 1:
                    # CRITICAL DANGER - Flee to boundary NOW!
                    return self._emergency_flee_home(game_state)

        # Get legal actions
        actions = game_state.get_legal_actions(self.index)

        # Evaluate each action
        values = [self.evaluate(game_state, a) for a in actions]

        # Choose the best action
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # If food is almost gone, return home
        food_left = len(self.get_food(game_state).as_list())
        
        # Adaptive endgame based on map size
        if self.map_size == 'tiny':
            # On tiny maps, return home even with 1-2 food left
            endgame_threshold = 3
        elif self.map_size == 'small':
            endgame_threshold = 2
        else:
            endgame_threshold = 2
            
        if food_left <= endgame_threshold:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)
    
    def _emergency_flee_home(self, game_state):
        """Emergency escape when ghost is about to catch us with food."""
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_position(self.index)
        
        # Remove STOP - we need to move!
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        
        if not actions:
            return Directions.STOP
        
        # Find action that gets us closest to boundary (home)
        best_dist = float('inf')
        best_action = None
        
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_position(self.index)
            
            # Check if this move takes us home (to our side)
            if self.is_on_our_side(new_pos):
                # We made it home! This action scores the food!
                return action
            
            # Otherwise, find closest to boundary
            if self.boundary:
                dist_to_home = min(self.get_maze_distance(new_pos, b) for b in self.boundary)
                if dist_to_home < best_dist:
                    best_dist = dist_to_home
                    best_action = action
        
        return best_action if best_action else random.choice(actions)

    def get_defending_enemies(self, game_state):
        """Get enemy ghosts that could catch us."""
        enemies = []
        for opponent in self.get_opponents(game_state):
            state = game_state.get_agent_state(opponent)
            if not state.is_pacman and state.scared_timer <= 0:
                pos = game_state.get_agent_position(opponent)
                if pos is not None:
                    enemies.append((opponent, pos))
        return enemies

    def get_invaders(self, game_state):
        """Get enemy Pacmen invading our side."""
        invaders = []
        for opponent in self.get_opponents(game_state):
            state = game_state.get_agent_state(opponent)
            if state.is_pacman:
                pos = game_state.get_agent_position(opponent)
                if pos is not None:
                    invaders.append((opponent, pos))
                else:
                    # Use particle filter estimate
                    probable_positions = self.get_opponent_probable_positions(opponent)
                    if probable_positions:
                        invaders.append((opponent, probable_positions[0]))
        return invaders

    def get_scared_ghosts(self, game_state):
        """Get scared enemy ghosts that we can eat."""
        scared = []
        for opponent in self.get_opponents(game_state):
            state = game_state.get_agent_state(opponent)
            if not state.is_pacman and state.scared_timer > 0:
                pos = game_state.get_agent_position(opponent)
                if pos is not None:
                    scared.append((opponent, pos, state.scared_timer))
        return scared


class OffensiveAgent(BaseAgent):
    """
    Aggressive agent focused on collecting food from opponent's side.
    Uses advanced pathfinding, escape planning, and smart food selection.
    """
    
    def _evaluate_food_safety(self, food_pos, game_state):
        """Evaluate how safe it is to go for this food."""
        my_pos = game_state.get_agent_position(self.index)
        
        # Distance to food
        food_dist = self.get_maze_distance(my_pos, food_pos)
        
        # Distance from food back to boundary
        if self.boundary:
            return_dist = min(self.get_maze_distance(food_pos, b) for b in self.boundary)
        else:
            return_dist = 0
        
        # Check for nearby ghosts
        enemies = self.get_defending_enemies(game_state)
        min_ghost_dist = float('inf')
        if enemies:
            for _, ghost_pos in enemies:
                ghost_to_food = self.get_maze_distance(ghost_pos, food_pos)
                min_ghost_dist = min(min_ghost_dist, ghost_to_food)
        
        # Penalty for dead ends
        dead_end_penalty = 50 if self._is_dead_end_nearby(food_pos) else 0
        
        # Safety score (lower is safer)
        # Prefer: close food, short return path, far from ghosts, not in dead end
        safety_score = food_dist + return_dist * 0.5 + dead_end_penalty
        if min_ghost_dist < float('inf'):
            safety_score += max(0, 10 - min_ghost_dist)  # Penalty if ghost close
        
        return safety_score
    
    def _should_return_home(self, game_state):
        """Decide if we should return home based on multiple factors."""
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        carrying = my_state.num_carrying
        
        if carrying == 0:
            return False
        
        # Get distance to home
        if self.boundary:
            dist_to_home = min(self.get_maze_distance(my_pos, b) for b in self.boundary)
        else:
            dist_to_home = 0
        
        # Check ghost proximity
        enemies = self.get_defending_enemies(game_state)
        min_ghost_dist = float('inf')
        if enemies:
            min_ghost_dist = min(self.get_maze_distance(my_pos, pos) for _, pos in enemies)
        
        # Calculate food remaining
        food_left = len(self.get_food(game_state).as_list())
        
        # Adaptive thresholds based on map size
        if self.map_size == 'tiny':
            # On tiny maps, be VERY aggressive - return quickly with any food
            if carrying >= 1:
                return True
        elif self.map_size == 'small':
            # On small maps, return with less food
            if carrying >= 3:
                return True
            if min_ghost_dist <= 4 and carrying >= 2:
                return True
        elif self.map_size == 'medium':
            # Medium maps - balanced approach
            if carrying >= 6:
                return True
            if min_ghost_dist <= 5 and carrying >= 3:
                return True
            if dist_to_home > 8 and carrying >= 4:
                return True
        else:
            # Large maps - can carry more before returning
            if carrying >= 8:
                return True
            if min_ghost_dist <= 5 and carrying >= 4:
                return True
            if dist_to_home > 12 and carrying >= 6:
                return True
        
        # Universal rules (apply to all map sizes)
        # 1. Return if very little food left
        if food_left <= 2 and carrying > 0:
            return True
        
        # 2. Return if we're in danger zone (dead end with ghost nearby)
        if self._is_dead_end_nearby(my_pos) and min_ghost_dist <= 7 and carrying > 0:
            return True
        
        # 3. Always return if ghost is extremely close
        if min_ghost_dist <= 2 and carrying > 0:
            return True
        
        return False

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Feature: Food count (negative because fewer is better - we ate them)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        # Feature: Distance to nearest SAFE food
        if len(food_list) > 0:
            # Evaluate food by safety score
            food_scores = [(food, self._evaluate_food_safety(food, game_state)) 
                          for food in food_list]
            food_scores.sort(key=lambda x: x[1])  # Sort by safety (lower is better)
            
            # Target the safest nearby food
            best_food = food_scores[0][0]
            min_distance = self.get_maze_distance(my_pos, best_food)
            features['distance_to_food'] = min_distance
            
            # Penalty if we're heading toward a dead end
            if self._is_dead_end_nearby(my_pos, depth=2):
                features['near_dead_end'] = 1

        # Feature: Distance to capsule (strategic usage)
        capsules = self.get_capsules(successor)
        enemies = self.get_defending_enemies(successor)
        
        if capsules and enemies:
            min_ghost_dist = min(self.get_maze_distance(my_pos, pos) for _, pos in enemies)
            
            # Only prioritize capsule if ghost is actually threatening
            if min_ghost_dist <= 6:
                closest_cap = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
                cap_dist = self.get_maze_distance(my_pos, closest_cap)
                features['distance_to_capsule'] = cap_dist
                
                # Extra incentive if ghost is very close
                if min_ghost_dist <= 3:
                    features['capsule_urgent'] = 1

        # Feature: Ghost distance (danger avoidance)
        if enemies:
            min_ghost_dist = min(self.get_maze_distance(my_pos, pos) for _, pos in enemies)
            features['ghost_distance'] = min_ghost_dist

            # Graduated danger levels
            if min_ghost_dist <= 1:
                features['critical_danger'] = 1
            elif min_ghost_dist <= 3:
                features['high_danger'] = 1
            elif min_ghost_dist <= 5:
                features['moderate_danger'] = 1

        # Feature: Scared ghost hunting
        scared_ghosts = self.get_scared_ghosts(successor)
        if scared_ghosts:
            # Only chase if enough time remaining and it's worth it
            for _, pos, timer in scared_ghosts:
                dist = self.get_maze_distance(my_pos, pos)
                if timer > dist + 5:  # Enough time to catch and do something
                    features['scared_ghost_distance'] = dist
                    break

        # Feature: Carrying food - incentivize returning when carrying a lot
        carrying = my_state.num_carrying
        features['carrying'] = carrying

        # Feature: Distance to home (dynamic based on game state)
        if carrying > 0 and self.boundary:
            dist_to_home = min(self.get_maze_distance(my_pos, boundary) for boundary in self.boundary)
            features['distance_to_home'] = dist_to_home

            # Dynamic return urgency
            if self._should_return_home(game_state):
                features['should_return'] = 1

        # Feature: Stop penalty
        if action == Directions.STOP:
            features['stop'] = 1

        # Feature: Reverse penalty (avoid oscillation)
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying

        # Base weights
        weights = {
            'successor_score': 100,
            'distance_to_food': -2,
            'near_dead_end': -30,
            'distance_to_capsule': -3,
            'capsule_urgent': 100,
            'ghost_distance': 2,
            'critical_danger': -2000,
            'high_danger': -500,
            'moderate_danger': -100,
            'scared_ghost_distance': -20,
            'carrying': 5,
            'distance_to_home': -1,
            'should_return': 500,
            'stop': -150,
            'reverse': -3
        }
        
        # Adaptive weights based on map size
        if self.map_size == 'tiny':
            # On tiny maps: be hyper-aggressive, return immediately
            weights['successor_score'] = 200  # Really want to eat food
            weights['distance_to_food'] = -5  # Go for food fast
            weights['carrying'] = 20  # Value carrying food highly
            weights['should_return'] = 1000  # Return ASAP
            weights['distance_to_home'] = -10  # Get home quickly
            weights['stop'] = -300  # Never stop!
            
        elif self.map_size == 'small':
            # Small maps: aggressive but slightly more cautious
            weights['successor_score'] = 150
            weights['distance_to_food'] = -3
            weights['carrying'] = 10
            weights['should_return'] = 700
            
        elif self.map_size == 'medium' or self.map_size == 'large':
            # Medium/large maps: more balanced, can afford to carry more
            weights['successor_score'] = 120
            weights['distance_to_food'] = -2
            weights['carrying'] = 5
            weights['near_dead_end'] = -50  # Avoid dead ends more on larger maps

        # Dynamic weight adjustment based on what we're carrying
        if carrying >= 5:
            weights['distance_to_home'] = -8
            weights['should_return'] = 800
            weights['distance_to_food'] = -0.5
            
        if carrying >= 8:
            weights['distance_to_home'] = -15
            weights['should_return'] = 1500
            weights['distance_to_food'] = 0

        # If ghost is very close and we have food, REALLY prioritize escape
        enemies = self.get_defending_enemies(game_state)
        if enemies and carrying > 0:
            my_pos = game_state.get_agent_position(self.index)
            min_ghost_dist = min(self.get_maze_distance(my_pos, pos) for _, pos in enemies)
            if min_ghost_dist <= 3:
                weights['distance_to_home'] = -20
                weights['critical_danger'] = -5000
                weights['should_return'] = 2000

        return weights


class DefensiveAgent(BaseAgent):
    """
    Agent focused on defending our territory.
    Uses particle filtering to track invaders and intercept them intelligently.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.patrol_index = 0
        self.target_invader = None
        self.last_invader_pos = None
    
    def _predict_invader_target(self, invader_pos, game_state):
        """Predict where the invader is likely going."""
        food_defending = self.get_food_you_are_defending(game_state).as_list()
        
        if not food_defending:
            return invader_pos
        
        # Find closest food to invader - likely their target
        closest_food = min(food_defending, 
                          key=lambda f: self.get_maze_distance(invader_pos, f))
        return closest_food
    
    def _get_intercept_position(self, invader_pos, target_pos, game_state):
        """Calculate best position to intercept invader."""
        my_pos = game_state.get_agent_position(self.index)
        
        # Try to position between invader and their escape route (boundary)
        # Find the boundary point closest to invader
        if not self.boundary:
            return invader_pos
        
        invader_escape = min(self.boundary,
                            key=lambda b: self.get_maze_distance(invader_pos, b))
        
        # Path from invader to escape
        # We want to be on this path
        my_dist_to_escape = self.get_maze_distance(my_pos, invader_escape)
        invader_dist_to_escape = self.get_maze_distance(invader_pos, invader_escape)
        
        # If we can't beat them to escape, chase them directly
        if my_dist_to_escape > invader_dist_to_escape:
            return invader_pos
        else:
            # Position at the escape point to intercept
            return invader_escape

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Feature: On defense
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # Feature: Number of invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Feature: Distance to invaders (visible or inferred) with interception
        if len(invaders) > 0:
            # We can see invaders - use intelligent interception
            best_intercept_dist = float('inf')
            
            for invader in invaders:
                invader_pos = invader.get_position()
                
                # Predict where invader is going
                target_pos = self._predict_invader_target(invader_pos, game_state)
                
                # Get intercept position
                intercept_pos = self._get_intercept_position(invader_pos, target_pos, game_state)
                
                # Distance to intercept position
                dist = self.get_maze_distance(my_pos, intercept_pos)
                best_intercept_dist = min(best_intercept_dist, dist)
            
            features['invader_distance'] = best_intercept_dist
            self.last_invader_pos = invaders[0].get_position()  # Track last seen position
        else:
            # No visible invaders - use particle filter to find suspected invaders
            inferred_invaders = []
            for opponent_idx in self.get_opponents(successor):
                opponent_state = successor.get_agent_state(opponent_idx)
                # Check if opponent is likely a Pacman (invading)
                if opponent_state.is_pacman:
                    # We know they're a Pacman but can't see them - use particle filter
                    probable_positions = self.get_opponent_probable_positions(opponent_idx)
                    if probable_positions:
                        inferred_invaders.append(probable_positions[0])
            
            if inferred_invaders:
                # Hunt the inferred invader positions
                dists = [self.get_maze_distance(my_pos, pos) for pos in inferred_invaders]
                features['invader_distance'] = min(dists)
                features['hunting_inferred'] = 1  # Hunting invisible invaders

        # Feature: Distance to patrol points when no invaders
        if len(invaders) == 0 and 'hunting_inferred' not in features and self.patrol_points:
            patrol_dist = min(self.get_maze_distance(my_pos, p) for p in self.patrol_points)
            features['patrol_distance'] = patrol_dist

        # Feature: Scared state
        if my_state.scared_timer > 0:
            features['scared'] = 1
            # Avoid invaders when scared (handled by weight change)

        # Feature: Food being defended
        food_defending = self.get_food_you_are_defending(successor).as_list()
        features['food_defending'] = len(food_defending)

        # Feature: Check for recently eaten food (detect invader location)
        current_food = self.get_food_you_are_defending(game_state).as_list()
        if len(self.last_food_defending) > len(current_food):
            # Food was eaten! Find which one and rush there
            eaten_food = set(self.last_food_defending) - set(current_food)
            if eaten_food:
                eaten_pos = list(eaten_food)[0]
                dist_to_eaten = self.get_maze_distance(my_pos, eaten_pos)
                features['eaten_food_distance'] = dist_to_eaten
                # High priority to get to eaten food location
                features['food_just_eaten'] = 1

        # Update last food defending
        self.last_food_defending = current_food

        # Feature: Stop penalty
        if action == Directions.STOP:
            features['stop'] = 1

        # Feature: Reverse penalty
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # Feature: Distance to boundary (good patrol position)
        if self.boundary and len(invaders) == 0:
            boundary_dist = min(self.get_maze_distance(my_pos, b) for b in self.boundary)
            features['boundary_distance'] = boundary_dist

        return features

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)

        weights = {
            'on_defense': 100,
            'num_invaders': -1000,
            'invader_distance': -15,
            'hunting_inferred': 0,
            'patrol_distance': -2,
            'scared': -50,
            'food_defending': 1,
            'eaten_food_distance': -20,
            'food_just_eaten': -200,
            'stop': -150,
            'reverse': -2,
            'boundary_distance': -0.5
        }
        
        # Adaptive defense based on map size
        if self.map_size == 'tiny' or self.map_size == 'small':
            # On small maps, be more aggressive in defense
            weights['invader_distance'] = -25  # Chase harder
            weights['food_just_eaten'] = -500  # React faster to stolen food
            weights['on_defense'] = 150  # Really stay on defense
            
        elif self.map_size == 'medium' or self.map_size == 'large':
            # On larger maps, be more strategic
            weights['invader_distance'] = -15
            weights['patrol_distance'] = -3  # Patrol more actively

        # If scared, FLEE from invaders instead of chasing
        if my_state.scared_timer > 0:
            weights['invader_distance'] = 50
            weights['food_just_eaten'] = 0

        return weights

    def choose_action(self, game_state):
        """Enhanced defensive action selection."""
        # Update particle filters
        for opponent in self.get_opponents(game_state):
            self.update_particles(game_state, opponent)

        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)

        # Get invaders
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # If no invaders and we have lots of food left, consider opportunistic offense
        food_defending = len(self.get_food_you_are_defending(game_state).as_list())
        enemy_food = len(self.get_food(game_state).as_list())

        if len(invaders) == 0 and not my_state.scared_timer and enemy_food > 2:
            # Consider opportunistic offense
            if food_defending > self.initial_defending_food - 5:
                # We're in good shape defensively, try to help offense
                return self._opportunistic_offense(game_state)

        # Standard defensive behavior
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def _opportunistic_offense(self, game_state):
        """Go on offense when safe to do so."""
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_position(self.index)

        # Find closest food on enemy side
        food = self.get_food(game_state).as_list()
        if not food:
            # No food to get, stay defensive
            return super().choose_action(game_state)

        # Check if there are dangerous ghosts
        enemies = self.get_defending_enemies(game_state)
        if enemies:
            min_ghost_dist = min(self.get_maze_distance(my_pos, pos) for _, pos in enemies)
            if min_ghost_dist <= 5:
                # Too dangerous, stay defensive
                return super().choose_action(game_state)

        # Go for closest food
        closest_food = min(food, key=lambda f: self.get_maze_distance(my_pos, f))

        # Find action that minimizes distance to closest food
        best_dist = float('inf')
        best_action = None
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(new_pos, closest_food)
            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action if best_action else random.choice(actions)


class AggressiveOffensiveAgent(BaseAgent):
    """
    A more aggressive offensive agent that takes risks to collect food.
    Uses power capsules strategically.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Food features
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            min_distance = min(self.get_maze_distance(my_pos, food) for food in food_list)
            features['distance_to_food'] = min_distance

        # Capsule features - aggressive usage
        capsules = self.get_capsules(successor)
        enemies = self.get_defending_enemies(successor)
        
        if capsules and enemies:
            min_ghost_dist = min(self.get_maze_distance(my_pos, pos) for _, pos in enemies)
            
            # Prioritize capsule if ghost is near
            if min_ghost_dist <= 5:
                closest_cap = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
                features['distance_to_capsule'] = self.get_maze_distance(my_pos, closest_cap)

        # Ghost avoidance
        if enemies:
            min_ghost_dist = min(self.get_maze_distance(my_pos, pos) for _, pos in enemies)
            if min_ghost_dist <= 1:
                features['immediate_danger'] = 1
            elif min_ghost_dist <= 3:
                features['ghost_nearby'] = 1

        # Scared ghost hunting (very aggressive)
        scared_ghosts = self.get_scared_ghosts(successor)
        if scared_ghosts:
            for _, pos, timer in scared_ghosts:
                dist = self.get_maze_distance(my_pos, pos)
                if timer > dist + 2:  # Enough time to catch
                    features['hunt_scared_ghost'] = dist

        # Carrying and returning
        carrying = my_state.num_carrying
        if carrying > 0 and self.boundary:
            dist_to_home = min(self.get_maze_distance(my_pos, b) for b in self.boundary)
            if carrying >= 3:
                features['return_home'] = dist_to_home

        # Stop/reverse penalties
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -3,
            'distance_to_capsule': -5,
            'immediate_danger': -1000,
            'ghost_nearby': -100,
            'hunt_scared_ghost': -50,
            'return_home': -4,
            'stop': -100,
            'reverse': -5
        }


class HybridAgent(BaseAgent):
    """
    Flexible agent that dynamically switches between offense and defense.
    Uses game state analysis to determine optimal role.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.current_role = 'offense'

    def choose_action(self, game_state):
        """Dynamically choose role based on game state."""
        # Update particle filters
        for opponent in self.get_opponents(game_state):
            self.update_particles(game_state, opponent)

        # Analyze game state
        score = self.get_score(game_state)
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        food_defending = len(self.get_food_you_are_defending(game_state).as_list())
        food_to_eat = len(self.get_food(game_state).as_list())
        my_state = game_state.get_agent_state(self.index)

        # Decision logic for role switching
        if len(invaders) >= 2:
            # Emergency defense
            self.current_role = 'defense'
        elif score > 5 and food_defending > food_to_eat:
            # We're winning, play more defensive
            self.current_role = 'defense'
        elif score < -5:
            # We're losing, need to be aggressive
            self.current_role = 'offense'
        elif my_state.num_carrying >= 5:
            # Have food, stay offensive to return it
            self.current_role = 'offense'
        else:
            # Default based on position
            if my_state.is_pacman:
                self.current_role = 'offense'
            else:
                self.current_role = 'defense'

        # Get action based on role
        actions = game_state.get_legal_actions(self.index)
        
        if self.current_role == 'offense':
            weights = self._offensive_weights(game_state)
        else:
            weights = self._defensive_weights(game_state)

        values = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            if self.current_role == 'offense':
                feat = self._offensive_features_for_state(successor, action, game_state)
            else:
                feat = self._defensive_features_for_state(successor, action, game_state)
            values.append(feat * weights)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def _offensive_features_for_state(self, successor, action, game_state):
        features = util.Counter()
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            min_dist = min(self.get_maze_distance(my_pos, food) for food in food_list)
            features['distance_to_food'] = min_dist

        enemies = self.get_defending_enemies(successor)
        if enemies:
            min_ghost_dist = min(self.get_maze_distance(my_pos, pos) for _, pos in enemies)
            if min_ghost_dist <= 2:
                features['ghost_danger'] = 1

        if action == Directions.STOP:
            features['stop'] = 1

        return features

    def _defensive_features_for_state(self, successor, action, game_state):
        features = util.Counter()
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1 if not my_state.is_pacman else 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1

        return features

    def _offensive_weights(self, game_state):
        return {
            'successor_score': 100,
            'distance_to_food': -2,
            'ghost_danger': -500,
            'stop': -100
        }

    def _defensive_weights(self, game_state):
        return {
            'on_defense': 100,
            'num_invaders': -1000,
            'invader_distance': -10,
            'stop': -100
        }
