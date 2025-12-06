# my_team.py
# ---------------
# Unified Belief-Based Agent for Pacman Capture the Flag
# Inspired by top competition agents using Behavior Trees and Belief State tracking
# ---------------

import random
import math
from collections import deque
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point, Counter
import contest.util as util


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='UnifiedAgent', second='UnifiedAgent', num_training=0):
    """
    Creates a team of two UnifiedAgents - both can attack and defend dynamically.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agent  #
##########

class UnifiedAgent(CaptureAgent):
    """
    A unified agent that can dynamically switch between offensive and defensive roles.
    Uses belief state tracking for opponent positions and behavior tree-like decision making.
    
    Key features:
    - Particle filtering for opponent position tracking
    - Dynamic role assignment based on game state
    - Aggressive food collection with smart return timing
    - Coordinated team behavior
    """
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.food_carried = 0
        self.beliefs = {}  # Particle filter beliefs for each opponent
        self.num_particles = 300
        self.last_observed_food = None
        self.team_roles = {}  # Track roles for coordination
        self.patrol_target = None
        self.stuck_counter = 0
        self.last_positions = deque(maxlen=8)
        
    def register_initial_state(self, game_state):
        """Initialize agent state and belief tracking."""
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        
        # Get map info
        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.mid_x = self.width // 2
        
        # Calculate border positions (for returning home)
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
        
        # Initialize particle filter for opponents
        self.opponents = self.get_opponents(game_state)
        self.initialize_beliefs(game_state)
        
        # Store initial food count
        self.initial_food = len(self.get_food(game_state).as_list())
        self.last_observed_food = self.get_food_you_are_defending(game_state).as_list()
        
        # Get all legal positions for pathfinding
        self.legal_positions = []
        for x in range(self.width):
            for y in range(self.height):
                if not self.walls[x][y]:
                    self.legal_positions.append((x, y))
    
    ###################
    # Belief Tracking #
    ###################
    
    def initialize_beliefs(self, game_state):
        """Initialize particle filter with uniform distribution on enemy side."""
        for opponent in self.opponents:
            self.beliefs[opponent] = Counter()
            # Start with particles on opponent's starting position
            start_pos = game_state.get_initial_agent_position(opponent)
            for _ in range(self.num_particles):
                self.beliefs[opponent][start_pos] += 1
            self.beliefs[opponent].normalize()
    
    def get_valid_positions_for_opponent(self, opponent, game_state):
        """Get valid positions where an opponent could be."""
        opponent_state = game_state.get_agent_state(opponent)
        if opponent_state.is_pacman:
            # Opponent is on our side
            if self.red:
                return [(x, y) for (x, y) in self.legal_positions if x < self.mid_x]
            else:
                return [(x, y) for (x, y) in self.legal_positions if x >= self.mid_x]
        else:
            # Opponent is a ghost on their side
            if self.red:
                return [(x, y) for (x, y) in self.legal_positions if x >= self.mid_x]
            else:
                return [(x, y) for (x, y) in self.legal_positions if x < self.mid_x]
    
    def update_beliefs(self, game_state):
        """Update belief state using observations and motion model."""
        my_pos = game_state.get_agent_position(self.index)
        noisy_distances = game_state.get_agent_distances()
        
        for opponent in self.opponents:
            # Check if we can directly observe the opponent
            opponent_pos = game_state.get_agent_position(opponent)
            
            if opponent_pos is not None:
                # Direct observation - reset beliefs to exact position
                self.beliefs[opponent] = Counter()
                self.beliefs[opponent][opponent_pos] = 1.0
            else:
                # Update based on noisy distance
                noisy_dist = noisy_distances[opponent]
                new_beliefs = Counter()
                
                # Get valid positions for this opponent
                valid_positions = self.get_valid_positions_for_opponent(opponent, game_state)
                
                for pos in valid_positions:
                    # Motion model: opponent could have moved from adjacent positions
                    prob = 0
                    for old_pos, old_prob in self.beliefs[opponent].items():
                        if old_prob > 0:
                            # Check if pos is reachable from old_pos in one step
                            dist = self.get_maze_distance(old_pos, pos) if old_pos in self.legal_positions else float('inf')
                            if dist <= 1:
                                prob += old_prob
                    
                    # Observation model: weight by how well noisy distance matches
                    if prob > 0:
                        true_dist = self.get_maze_distance(my_pos, pos)
                        # Noisy distance has Gaussian-like distribution
                        dist_diff = abs(noisy_dist - true_dist)
                        if dist_diff <= 7:  # Noisy readings can be off by up to 7
                            observation_prob = max(0, 1 - dist_diff * 0.1)
                            new_beliefs[pos] = prob * observation_prob
                
                if new_beliefs.total_count() > 0:
                    new_beliefs.normalize()
                    self.beliefs[opponent] = new_beliefs
                else:
                    # Lost track - reinitialize
                    self.initialize_beliefs_for_opponent(opponent, game_state)
    
    def initialize_beliefs_for_opponent(self, opponent, game_state):
        """Reinitialize beliefs for a single opponent."""
        self.beliefs[opponent] = Counter()
        valid_positions = self.get_valid_positions_for_opponent(opponent, game_state)
        if valid_positions:
            for pos in valid_positions:
                self.beliefs[opponent][pos] = 1.0 / len(valid_positions)
    
    def get_most_likely_position(self, opponent):
        """Get the most likely position of an opponent."""
        if self.beliefs[opponent].total_count() == 0:
            return None
        return max(self.beliefs[opponent], key=self.beliefs[opponent].get)
    
    def get_estimated_opponent_positions(self, game_state):
        """Get estimated positions for all opponents."""
        positions = {}
        for opponent in self.opponents:
            actual_pos = game_state.get_agent_position(opponent)
            if actual_pos is not None:
                positions[opponent] = actual_pos
            else:
                positions[opponent] = self.get_most_likely_position(opponent)
        return positions
    
    #######################
    # Role Assignment     #
    #######################
    
    def determine_role(self, game_state):
        """
        Dynamically determine whether this agent should attack or defend.
        
        Strategy:
        - If winning by a lot: both defend
        - If losing or tied: be aggressive
        - Coordinate with teammate to have at least one defender when needed
        """
        score = self.get_score(game_state)
        food_left = len(self.get_food(game_state).as_list())
        food_defending = len(self.get_food_you_are_defending(game_state).as_list())
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        # Check for invaders
        enemies = [game_state.get_agent_state(i) for i in self.opponents]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        num_invaders = len(invaders)
        
        # Check if invaders are estimated based on beliefs
        if num_invaders == 0:
            for opponent in self.opponents:
                opp_state = game_state.get_agent_state(opponent)
                if opp_state.is_pacman:
                    num_invaders += 1
        
        # Get teammate index
        team = self.get_team(game_state)
        teammate_idx = [i for i in team if i != self.index][0]
        teammate_pos = game_state.get_agent_position(teammate_idx)
        teammate_state = game_state.get_agent_state(teammate_idx)
        
        # Determine if I'm closer to home or teammate
        my_dist_to_home = min(self.get_maze_distance(my_pos, bp) for bp in self.border_positions)
        teammate_dist_to_home = min(self.get_maze_distance(teammate_pos, bp) for bp in self.border_positions) if teammate_pos else float('inf')
        
        # Am I carrying food?
        carrying = my_state.num_carrying
        
        # Decision logic (behavior tree style)
        
        # Priority 1: Return food if carrying a lot or game ending soon
        time_left = game_state.data.timeleft
        if carrying >= 5 or (carrying > 0 and time_left < 100):
            return 'return'
        
        # Priority 2: Return food if enemy ghost is very close
        for opponent in self.opponents:
            opp_pos = game_state.get_agent_position(opponent)
            opp_state = game_state.get_agent_state(opponent)
            if opp_pos and not opp_state.is_pacman and opp_state.scared_timer <= 2:
                if self.get_maze_distance(my_pos, opp_pos) <= 3 and carrying > 0:
                    return 'return'
        
        # Priority 3: Chase invaders if we're winning and there are invaders
        if score > 5 and num_invaders > 0:
            return 'defend'
        
        # Priority 4: If invaders and I'm closer to home, defend
        if num_invaders > 0 and my_dist_to_home < teammate_dist_to_home:
            return 'defend'
        
        # Priority 5: Coordinate - one attacks, one defends based on index
        if score > 0 and num_invaders > 0:
            if self.index == min(team):
                return 'attack'
            else:
                return 'defend'
        
        # Default: Attack aggressively
        return 'attack'
    
    ######################
    # Action Selection   #
    ######################
    
    def choose_action(self, game_state):
        """Main decision function - behavior tree style."""
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        # Update belief tracking
        self.update_beliefs(game_state)
        
        # Track position history for stuck detection
        self.last_positions.append(my_pos)
        
        # Check if we just got eaten (respawned)
        if self.get_previous_observation() is not None:
            prev_pos = self.get_previous_observation().get_agent_position(self.index)
            if prev_pos and self.get_maze_distance(prev_pos, my_pos) > 1:
                self.food_carried = 0
        
        # Track food eaten
        if self.get_previous_observation() is not None:
            prev_food = self.get_food(self.get_previous_observation()).as_list()
            curr_food = self.get_food(game_state).as_list()
            if len(prev_food) > len(curr_food):
                self.food_carried += len(prev_food) - len(curr_food)
        
        # Update actual carrying count
        self.food_carried = my_state.num_carrying
        
        # Determine role
        role = self.determine_role(game_state)
        
        # Execute role-specific behavior
        if role == 'return':
            return self.return_home(game_state)
        elif role == 'defend':
            return self.defend(game_state)
        else:
            return self.attack(game_state)
    
    def attack(self, game_state):
        """Aggressive food collection behavior."""
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        
        # Get enemy ghost positions
        enemy_ghosts = []
        for opponent in self.opponents:
            opp_state = game_state.get_agent_state(opponent)
            opp_pos = game_state.get_agent_position(opponent)
            if opp_pos is None:
                opp_pos = self.get_most_likely_position(opponent)
            if opp_pos and not opp_state.is_pacman and opp_state.scared_timer <= 2:
                enemy_ghosts.append((opponent, opp_pos))
        
        # Check if any ghost is dangerously close
        danger_zone = 4
        in_danger = False
        for _, ghost_pos in enemy_ghosts:
            if ghost_pos and self.get_maze_distance(my_pos, ghost_pos) <= danger_zone:
                in_danger = True
                break
        
        # If carrying food and in danger, return home
        if in_danger and self.food_carried > 0:
            return self.return_home(game_state)
        
        # If in danger and capsule is closer than home, go for capsule
        if in_danger and capsules:
            closest_capsule = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
            cap_dist = self.get_maze_distance(my_pos, closest_capsule)
            home_dist = min(self.get_maze_distance(my_pos, bp) for bp in self.border_positions)
            if cap_dist < home_dist:
                return self.a_star_action(game_state, closest_capsule, avoid_ghosts=True)
        
        # Go for closest food, avoiding ghosts
        if food_list:
            # Score each food by distance minus danger
            scored_food = []
            for food in food_list:
                dist_to_food = self.get_maze_distance(my_pos, food)
                # Penalize food near ghosts
                danger_penalty = 0
                for _, ghost_pos in enemy_ghosts:
                    if ghost_pos:
                        ghost_to_food = self.get_maze_distance(ghost_pos, food)
                        if ghost_to_food < 3:
                            danger_penalty += (3 - ghost_to_food) * 10
                
                score = dist_to_food + danger_penalty
                scored_food.append((food, score))
            
            # Sort by score (lower is better)
            scored_food.sort(key=lambda x: x[1])
            
            # Go for best food
            target_food = scored_food[0][0]
            return self.a_star_action(game_state, target_food, avoid_ghosts=True)
        
        # No food left, return home
        return self.return_home(game_state)
    
    def defend(self, game_state):
        """Defensive behavior - chase invaders or patrol."""
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        # Find invaders
        invaders = []
        for opponent in self.opponents:
            opp_state = game_state.get_agent_state(opponent)
            opp_pos = game_state.get_agent_position(opponent)
            if opp_state.is_pacman:
                if opp_pos:
                    invaders.append((opponent, opp_pos))
                else:
                    # Use belief state
                    estimated_pos = self.get_most_likely_position(opponent)
                    if estimated_pos:
                        invaders.append((opponent, estimated_pos))
        
        # If we're scared, stay away from pacmen
        if my_state.scared_timer > 0:
            if invaders:
                # Move away from invaders
                closest_invader = min(invaders, key=lambda x: self.get_maze_distance(my_pos, x[1]))
                return self.flee_from(game_state, closest_invader[1])
            else:
                # Patrol near border but safely
                return self.patrol(game_state)
        
        # Chase invaders
        if invaders:
            # Chase the closest invader
            closest_invader = min(invaders, key=lambda x: self.get_maze_distance(my_pos, x[1]))
            return self.a_star_action(game_state, closest_invader[1], avoid_ghosts=False)
        
        # No visible invaders - check if food was eaten recently
        current_food = self.get_food_you_are_defending(game_state).as_list()
        if self.last_observed_food and len(current_food) < len(self.last_observed_food):
            # Food was eaten! Find the missing food location
            missing_food = [f for f in self.last_observed_food if f not in current_food]
            if missing_food:
                # Go to where food was eaten
                target = missing_food[0]
                self.last_observed_food = current_food
                return self.a_star_action(game_state, target, avoid_ghosts=False)
        
        self.last_observed_food = current_food
        
        # Patrol the border
        return self.patrol(game_state)
    
    def patrol(self, game_state):
        """Patrol the border to intercept invaders."""
        my_pos = game_state.get_agent_position(self.index)
        
        # Choose patrol targets along the border
        if self.patrol_target is None or my_pos == self.patrol_target:
            # Pick a new patrol target
            # Prioritize positions near food
            food_defending = self.get_food_you_are_defending(game_state).as_list()
            if food_defending:
                # Find border position closest to most food
                scored_borders = []
                for bp in self.border_positions:
                    avg_dist_to_food = sum(self.get_maze_distance(bp, f) for f in food_defending) / len(food_defending)
                    scored_borders.append((bp, avg_dist_to_food))
                scored_borders.sort(key=lambda x: x[1])
                
                # Pick from top 3 positions
                top_positions = [bp for bp, _ in scored_borders[:3]]
                self.patrol_target = random.choice(top_positions)
            else:
                self.patrol_target = random.choice(self.border_positions)
        
        return self.a_star_action(game_state, self.patrol_target, avoid_ghosts=False)
    
    def return_home(self, game_state):
        """Return to home side to deposit food."""
        my_pos = game_state.get_agent_position(self.index)
        
        # Find closest safe border position
        closest_border = min(self.border_positions, key=lambda bp: self.get_maze_distance(my_pos, bp))
        
        return self.a_star_action(game_state, closest_border, avoid_ghosts=True)
    
    def flee_from(self, game_state, danger_pos):
        """Move away from a dangerous position."""
        my_pos = game_state.get_agent_position(self.index)
        actions = game_state.get_legal_actions(self.index)
        actions = [a for a in actions if a != Directions.STOP]
        
        # Pick action that maximizes distance from danger
        best_action = None
        best_dist = -1
        
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(new_pos, danger_pos)
            if dist > best_dist:
                best_dist = dist
                best_action = action
        
        return best_action if best_action else Directions.STOP
    
    ###############
    # Pathfinding #
    ###############
    
    def a_star_action(self, game_state, goal, avoid_ghosts=True):
        """Use A* to find path to goal, returns first action."""
        my_pos = game_state.get_agent_position(self.index)
        
        if my_pos == goal:
            return Directions.STOP
        
        # Get ghost positions to avoid
        ghost_positions = set()
        if avoid_ghosts:
            for opponent in self.opponents:
                opp_state = game_state.get_agent_state(opponent)
                opp_pos = game_state.get_agent_position(opponent)
                if opp_pos is None:
                    opp_pos = self.get_most_likely_position(opponent)
                if opp_pos and not opp_state.is_pacman and opp_state.scared_timer <= 2:
                    ghost_positions.add(opp_pos)
                    # Also add adjacent positions as dangerous
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        adj = (opp_pos[0] + dx, opp_pos[1] + dy)
                        if adj in self.legal_positions:
                            ghost_positions.add(adj)
        
        # A* search
        frontier = util.PriorityQueue()
        frontier.push((my_pos, []), 0)
        explored = set()
        
        while not frontier.is_empty():
            pos, path = frontier.pop()
            
            if pos == goal:
                if path:
                    return path[0]
                return Directions.STOP
            
            if pos in explored:
                continue
            explored.add(pos)
            
            # Get successors
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.direction_to_vector(action)
                next_x, next_y = int(pos[0] + dx), int(pos[1] + dy)
                next_pos = (next_x, next_y)
                
                if next_pos in self.legal_positions and next_pos not in explored:
                    new_path = path + [action]
                    # Cost: distance + penalty for ghost proximity
                    cost = len(new_path)
                    
                    if avoid_ghosts and next_pos in ghost_positions:
                        cost += 100  # Heavy penalty for ghost positions
                    
                    heuristic = self.get_maze_distance(next_pos, goal)
                    frontier.push((next_pos, new_path), cost + heuristic)
        
        # A* failed, fall back to greedy action
        return self.greedy_action(game_state, goal)
    
    def greedy_action(self, game_state, goal):
        """Fallback: pick action that minimizes distance to goal."""
        my_pos = game_state.get_agent_position(self.index)
        actions = game_state.get_legal_actions(self.index)
        actions = [a for a in actions if a != Directions.STOP]
        
        best_action = None
        best_dist = float('inf')
        
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(new_pos, goal)
            if dist < best_dist:
                best_dist = dist
                best_action = action
        
        return best_action if best_action else Directions.STOP
