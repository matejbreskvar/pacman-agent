# my_team.py
# ---------------
# Enhanced Pacman CTF Agent v5
# Features:
# - Dynamic role switching based on score
# - Unified agents that can attack AND defend
# - Advanced A* with ghost prediction and dead-end avoidance
# - Capsule strategy with scared ghost hunting
# - Team coordination to avoid conflicts
# - Particle filtering for opponent tracking
# ---------------

import random
import math
from collections import deque
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point
from collections import Counter


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='DynamicAgent', second='DynamicAgent', num_training=0):
    """
    Creates a team of two DynamicAgents - both can attack and defend.
    The first agent (lower index) tends toward offense, second toward defense,
    but both switch dynamically based on game state.
    """
    return [eval(first)(first_index, role_bias='offense'), 
            eval(second)(second_index, role_bias='defense')]


# Shared state for team coordination
class TeamState:
    """Shared state between agents for coordination."""
    current_attackers = set()
    current_defenders = set()
    target_food = {}  # agent_idx -> target food position
    last_update_time = -1
    
    @classmethod
    def reset(cls):
        cls.current_attackers = set()
        cls.current_defenders = set()
        cls.target_food = {}
        cls.last_update_time = -1


##########
# Agent  #
##########

class DynamicAgent(CaptureAgent):
    """
    A unified agent that dynamically switches between offensive and defensive roles
    based on game score, time, and team coordination.
    
    Key features:
    - Score-aware mode switching (defend when winning, attack when losing)
    - Particle filtering for opponent tracking
    - Advanced A* with ghost avoidance and dead-end detection
    - Capsule strategy
    - Team coordination
    """
    
    def __init__(self, index, time_for_computing=0.1, role_bias='offense'):
        super().__init__(index, time_for_computing)
        self.role_bias = role_bias  # 'offense' or 'defense'
        self.start = None
        self.beliefs = {}
        self.num_particles = 200
        self.last_positions = deque(maxlen=10)
        self.patrol_target = None
        self.patrol_direction = 1
        self.dead_ends = set()
        self.escape_routes = {}
        
    def register_initial_state(self, game_state):
        """Initialize agent state and precompute map analysis."""
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        
        # Reset team state at game start
        if game_state.data.timeleft > 1190:
            TeamState.reset()
        
        # Map info
        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.mid_x = self.width // 2
        
        # Territory boundaries
        if self.red:
            self.home_x = self.mid_x - 1
            self.enemy_x = self.mid_x
        else:
            self.home_x = self.mid_x
            self.enemy_x = self.mid_x - 1
        
        # Find valid border positions
        self.border_positions = []
        for y in range(self.height):
            if not self.walls[self.home_x][y]:
                self.border_positions.append((self.home_x, y))
        
        # Get all legal positions
        self.legal_positions = []
        for x in range(self.width):
            for y in range(self.height):
                if not self.walls[x][y]:
                    self.legal_positions.append((x, y))
        
        # Initialize opponent tracking
        self.opponents = self.get_opponents(game_state)
        self.initialize_beliefs(game_state)
        
        # Precompute dead ends and escape routes
        self.analyze_map()
        
        # Store initial food
        self.initial_food = len(self.get_food(game_state).as_list())
    
    def analyze_map(self):
        """Analyze map for dead ends and escape routes."""
        # Find dead ends (positions with only 1 neighbor)
        for pos in self.legal_positions:
            neighbors = self.get_neighbors(pos)
            if len(neighbors) == 1:
                self.dead_ends.add(pos)
                # Trace back to find escape route
                self.trace_dead_end(pos, neighbors[0])
    
    def get_neighbors(self, pos):
        """Get walkable neighbors of a position."""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if not self.walls[nx][ny]:
                    neighbors.append((nx, ny))
        return neighbors
    
    def trace_dead_end(self, start, escape_pos):
        """Trace dead end path and mark escape routes."""
        self.escape_routes[start] = escape_pos
        current = start
        prev = None
        
        while True:
            neighbors = [n for n in self.get_neighbors(current) if n != prev]
            if len(neighbors) != 1:
                break
            prev = current
            current = neighbors[0]
            self.escape_routes[current] = self.escape_routes.get(start, escape_pos)
    
    ###################
    # Belief Tracking #
    ###################
    
    def initialize_beliefs(self, game_state):
        """Initialize particle filter for opponent tracking."""
        for opponent in self.opponents:
            self.beliefs[opponent] = Counter()
            start_pos = game_state.get_initial_agent_position(opponent)
            self.beliefs[opponent][start_pos] = self.num_particles
    
    def update_beliefs(self, game_state):
        """Update belief state using observations."""
        my_pos = game_state.get_agent_position(self.index)
        noisy_distances = game_state.get_agent_distances()
        
        for opponent in self.opponents:
            opponent_pos = game_state.get_agent_position(opponent)
            
            if opponent_pos is not None:
                # Direct observation
                self.beliefs[opponent] = Counter()
                self.beliefs[opponent][opponent_pos] = self.num_particles
            else:
                # Update with noisy distance
                noisy_dist = noisy_distances[opponent]
                new_beliefs = Counter()
                
                # Get valid positions based on opponent state
                opp_state = game_state.get_agent_state(opponent)
                if opp_state.is_pacman:
                    if self.red:
                        valid_x = lambda x: x < self.mid_x
                    else:
                        valid_x = lambda x: x >= self.mid_x
                else:
                    if self.red:
                        valid_x = lambda x: x >= self.mid_x
                    else:
                        valid_x = lambda x: x < self.mid_x
                
                for pos, count in self.beliefs[opponent].items():
                    if count == 0:
                        continue
                    # Motion model: spread particles to neighbors
                    neighbors = self.get_neighbors(pos) + [pos]
                    for new_pos in neighbors:
                        if valid_x(new_pos[0]):
                            true_dist = self.get_maze_distance(my_pos, new_pos)
                            # Observation model
                            if abs(noisy_dist - true_dist) <= 7:
                                weight = max(0.1, 1 - abs(noisy_dist - true_dist) * 0.1)
                                new_beliefs[new_pos] += count * weight / len(neighbors)
                
                if sum(new_beliefs.values()) > 0:
                    # Normalize
                    total = sum(new_beliefs.values())
                    for pos in new_beliefs:
                        new_beliefs[pos] = new_beliefs[pos] * self.num_particles / total
                    self.beliefs[opponent] = new_beliefs
                else:
                    # Lost track - reinitialize uniformly on valid side
                    self.beliefs[opponent] = Counter()
                    valid_positions = [p for p in self.legal_positions if valid_x(p[0])]
                    for pos in random.sample(valid_positions, min(50, len(valid_positions))):
                        self.beliefs[opponent][pos] = self.num_particles / 50
    
    def get_likely_opponent_position(self, opponent):
        """Get most likely position of opponent."""
        if not self.beliefs[opponent]:
            return None
        return max(self.beliefs[opponent].keys(), key=lambda p: self.beliefs[opponent][p])
    
    def get_opponent_positions(self, game_state):
        """Get all opponent positions (observed or estimated)."""
        positions = {}
        for opponent in self.opponents:
            actual = game_state.get_agent_position(opponent)
            if actual:
                positions[opponent] = actual
            else:
                positions[opponent] = self.get_likely_opponent_position(opponent)
        return positions
    
    ###################
    # Role Assignment #
    ###################
    
    def determine_role(self, game_state):
        """
        Dynamically determine role based on game state.
        
        Key factors:
        - Current score (defend when winning, attack when losing)
        - Number of invaders
        - Food carried
        - Time remaining
        - Team coordination
        """
        score = self.get_score(game_state)
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        carrying = my_state.num_carrying
        time_left = game_state.data.timeleft
        
        # Get invader info
        enemies = [game_state.get_agent_state(i) for i in self.opponents]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        
        # Get ghost threats
        ghost_threats = []
        for opp in self.opponents:
            opp_state = game_state.get_agent_state(opp)
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos and not opp_state.is_pacman and opp_state.scared_timer <= 3:
                dist = self.get_maze_distance(my_pos, opp_pos)
                if dist <= 5:
                    ghost_threats.append((opp, opp_pos, dist))
        
        # Priority 1: Return food if carrying and in danger
        if carrying > 0 and ghost_threats:
            closest_ghost_dist = min(t[2] for t in ghost_threats)
            if closest_ghost_dist <= 3:
                return 'return'
        
        # Priority 2: Return food if carrying a lot or time running out
        return_threshold = 5 if score >= 0 else 8  # More aggressive when losing
        if carrying >= return_threshold:
            return 'return'
        if carrying > 0 and time_left < 150:
            return 'return'
        
        # Priority 3: Emergency defense - invaders when winning
        if score > 3 and len(invaders) > 0:
            # Coordinate: one defends, one attacks
            team = self.get_team(game_state)
            teammate_idx = [i for i in team if i != self.index][0]
            
            # Defense bias agent prioritizes defense
            if self.role_bias == 'defense':
                return 'defend'
            elif len(invaders) >= 2:
                return 'defend'  # Both defend if 2 invaders
        
        # Priority 4: Score-based strategy
        if score > 5:
            # Winning comfortably - be defensive
            if self.role_bias == 'defense':
                return 'defend'
            else:
                # Offense agent still attacks but carefully
                return 'attack_safe'
        elif score < -3:
            # Losing - all-out attack
            return 'attack_aggressive'
        else:
            # Close game - balanced approach
            if self.role_bias == 'defense' and len(invaders) > 0:
                return 'defend'
            return 'attack'
    
    ######################
    # Action Selection   #
    ######################
    
    def choose_action(self, game_state):
        """Main decision function."""
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        # Update beliefs
        self.update_beliefs(game_state)
        
        # Track positions for stuck detection
        self.last_positions.append(my_pos)
        
        # Determine role
        role = self.determine_role(game_state)
        
        # Update team state
        TeamState.current_attackers.discard(self.index)
        TeamState.current_defenders.discard(self.index)
        if 'attack' in role or role == 'return':
            TeamState.current_attackers.add(self.index)
        else:
            TeamState.current_defenders.add(self.index)
        
        # Execute role
        if role == 'return':
            return self.return_home(game_state)
        elif role == 'defend':
            return self.defend(game_state)
        elif role == 'attack_aggressive':
            return self.attack(game_state, aggressive=True)
        elif role == 'attack_safe':
            return self.attack(game_state, aggressive=False)
        else:
            return self.attack(game_state, aggressive=True)
    
    def attack(self, game_state, aggressive=True):
        """Offensive behavior with ghost avoidance."""
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        
        # Get ghost positions and threats
        ghost_positions = []
        scared_ghosts = []
        for opp in self.opponents:
            opp_state = game_state.get_agent_state(opp)
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos is None:
                opp_pos = self.get_likely_opponent_position(opp)
            
            if opp_pos and not opp_state.is_pacman:
                if opp_state.scared_timer > 3:
                    scared_ghosts.append((opp, opp_pos, opp_state.scared_timer))
                else:
                    ghost_positions.append((opp, opp_pos))
        
        # Check danger level
        danger_dist = float('inf')
        for _, ghost_pos in ghost_positions:
            if ghost_pos:
                dist = self.get_maze_distance(my_pos, ghost_pos)
                danger_dist = min(danger_dist, dist)
        
        # If in danger and carrying food, return home
        if danger_dist <= 4 and my_state.num_carrying > 0:
            return self.return_home(game_state)
        
        # If in danger and capsule is close, go for capsule
        if danger_dist <= 5 and capsules:
            closest_capsule = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
            cap_dist = self.get_maze_distance(my_pos, closest_capsule)
            if cap_dist < danger_dist:
                return self.a_star_move(game_state, closest_capsule, ghost_positions, aggressive=True)
        
        # Hunt scared ghosts if close
        for opp, ghost_pos, timer in scared_ghosts:
            if ghost_pos:
                ghost_dist = self.get_maze_distance(my_pos, ghost_pos)
                if ghost_dist <= timer - 2:  # Can reach before timer expires
                    return self.a_star_move(game_state, ghost_pos, [], aggressive=True)
        
        # Find best food target
        if food_list:
            # Avoid food that teammate is targeting
            available_food = food_list
            teammate_target = TeamState.target_food.get(
                [i for i in self.get_team(game_state) if i != self.index][0]
            )
            if teammate_target and teammate_target in available_food and len(available_food) > 1:
                available_food = [f for f in available_food if f != teammate_target]
            
            # Score each food based on distance and safety
            def food_score(food):
                dist = self.get_maze_distance(my_pos, food)
                # Penalize food near ghosts
                ghost_penalty = 0
                for _, ghost_pos in ghost_positions:
                    if ghost_pos:
                        ghost_dist = self.get_maze_distance(food, ghost_pos)
                        if ghost_dist <= 3:
                            ghost_penalty += (4 - ghost_dist) * 10
                # Penalize dead ends when ghosts are around
                if food in self.dead_ends and danger_dist < 8:
                    ghost_penalty += 20
                return dist + ghost_penalty
            
            target_food = min(available_food, key=food_score)
            TeamState.target_food[self.index] = target_food
            
            return self.a_star_move(game_state, target_food, ghost_positions, aggressive)
        
        # No food left - return home
        return self.return_home(game_state)
    
    def defend(self, game_state):
        """Defensive behavior - chase invaders or patrol."""
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        # Find invaders
        invaders = []
        for opp in self.opponents:
            opp_state = game_state.get_agent_state(opp)
            opp_pos = game_state.get_agent_position(opp)
            if opp_state.is_pacman:
                if opp_pos:
                    invaders.append((opp, opp_pos))
                else:
                    # Use belief state
                    estimated_pos = self.get_likely_opponent_position(opp)
                    if estimated_pos:
                        invaders.append((opp, estimated_pos))
        
        # If scared, avoid invaders
        if my_state.scared_timer > 0 and invaders:
            # Run away from invaders
            closest_invader = min(invaders, key=lambda x: self.get_maze_distance(my_pos, x[1]))
            return self.flee_from(game_state, closest_invader[1])
        
        # Chase closest invader
        if invaders:
            closest_invader = min(invaders, key=lambda x: self.get_maze_distance(my_pos, x[1]))
            return self.a_star_move(game_state, closest_invader[1], [], aggressive=True)
        
        # No invaders - patrol border
        return self.patrol(game_state)
    
    def patrol(self, game_state):
        """Patrol the border area."""
        my_pos = game_state.get_agent_position(self.index)
        
        # Set patrol target if needed
        if self.patrol_target is None or my_pos == self.patrol_target:
            if self.patrol_direction == 1:
                self.patrol_target = max(self.border_positions, key=lambda p: p[1])
                self.patrol_direction = -1
            else:
                self.patrol_target = min(self.border_positions, key=lambda p: p[1])
                self.patrol_direction = 1
        
        return self.a_star_move(game_state, self.patrol_target, [], aggressive=True)
    
    def return_home(self, game_state):
        """Return to home territory to deposit food."""
        my_pos = game_state.get_agent_position(self.index)
        
        # Get ghost positions
        ghost_positions = []
        for opp in self.opponents:
            opp_state = game_state.get_agent_state(opp)
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos and not opp_state.is_pacman and opp_state.scared_timer <= 2:
                ghost_positions.append((opp, opp_pos))
        
        # Find safest border position
        def border_safety(border_pos):
            dist = self.get_maze_distance(my_pos, border_pos)
            # Add ghost penalty
            ghost_penalty = 0
            for _, ghost_pos in ghost_positions:
                ghost_to_border = self.get_maze_distance(ghost_pos, border_pos)
                if ghost_to_border <= 3:
                    ghost_penalty += (4 - ghost_to_border) * 20
            return dist + ghost_penalty
        
        target = min(self.border_positions, key=border_safety)
        return self.a_star_move(game_state, target, ghost_positions, aggressive=False)
    
    def flee_from(self, game_state, threat_pos):
        """Move away from a threat."""
        my_pos = game_state.get_agent_position(self.index)
        actions = game_state.get_legal_actions(self.index)
        
        best_action = None
        best_dist = -1
        
        for action in actions:
            if action == Directions.STOP:
                continue
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(new_pos, threat_pos)
            if dist > best_dist:
                best_dist = dist
                best_action = action
        
        return best_action if best_action else Directions.STOP
    
    ######################
    # A* Pathfinding     #
    ######################
    
    def a_star_move(self, game_state, goal, ghost_positions, aggressive=True):
        """
        A* pathfinding with ghost avoidance.
        Returns the first action to take.
        """
        import heapq
        
        my_pos = game_state.get_agent_position(self.index)
        
        if my_pos == goal:
            return Directions.STOP
        
        # Priority queue: (f_score, tie_breaker, position, path)
        frontier = []
        tie_breaker = 0
        heapq.heappush(frontier, (0, tie_breaker, my_pos, []))
        
        visited = set()
        g_scores = {my_pos: 0}
        
        # Ghost danger zones
        ghost_danger = {}
        for _, ghost_pos in ghost_positions:
            if ghost_pos:
                for pos in self.legal_positions:
                    dist = self.get_maze_distance(pos, ghost_pos)
                    if dist <= 4:
                        old_danger = ghost_danger.get(pos, 0)
                        ghost_danger[pos] = max(old_danger, (5 - dist) * 50)
        
        while frontier:
            f, _, current, path = heapq.heappop(frontier)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                return path[0] if path else Directions.STOP
            
            # Expand neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # Calculate g score
                new_g = g_scores[current] + 1
                
                # Add ghost penalty
                ghost_cost = ghost_danger.get(neighbor, 0)
                if not aggressive:
                    ghost_cost *= 2  # More cautious when not aggressive
                
                # Dead end penalty when ghosts are near
                dead_end_cost = 0
                if neighbor in self.dead_ends and ghost_danger:
                    dead_end_cost = 100
                
                total_g = new_g + ghost_cost + dead_end_cost
                
                if neighbor not in g_scores or total_g < g_scores[neighbor]:
                    g_scores[neighbor] = total_g
                    h = self.get_maze_distance(neighbor, goal)
                    f = total_g + h
                    
                    # Determine action to get to neighbor
                    dx = neighbor[0] - current[0]
                    dy = neighbor[1] - current[1]
                    if dx == 1:
                        action = Directions.EAST
                    elif dx == -1:
                        action = Directions.WEST
                    elif dy == 1:
                        action = Directions.NORTH
                    else:
                        action = Directions.SOUTH
                    
                    new_path = path + [action] if path else [action]
                    tie_breaker += 1
                    heapq.heappush(frontier, (f, tie_breaker, neighbor, new_path))
        
        # No path found - take best available action
        return self.get_best_fallback_action(game_state, goal, ghost_positions)
    
    def get_best_fallback_action(self, game_state, goal, ghost_positions):
        """Fallback when A* fails."""
        my_pos = game_state.get_agent_position(self.index)
        actions = game_state.get_legal_actions(self.index)
        
        best_action = None
        best_score = float('-inf')
        
        for action in actions:
            if action == Directions.STOP:
                continue
            
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            
            # Score based on distance to goal
            score = -self.get_maze_distance(new_pos, goal)
            
            # Penalize moving toward ghosts
            for _, ghost_pos in ghost_positions:
                if ghost_pos:
                    ghost_dist = self.get_maze_distance(new_pos, ghost_pos)
                    if ghost_dist <= 2:
                        score -= (3 - ghost_dist) * 100
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action else random.choice(actions)
