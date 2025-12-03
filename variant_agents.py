"""
Alternative Agent Strategies for Training Diversity
Provides different play styles to train against
"""

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point, manhattan_distance
import random


def create_team(first_index, second_index, is_red,
               first='AggressiveAgent', second='CautiousAgent', num_training=0):
    """Creates a team with aggressive offense and cautious defense"""
    return [eval(first)(first_index), eval(second)(second_index)]


class AggressiveAgent(CaptureAgent):
    """
    Highly aggressive offensive agent - always pushes deep, takes risks
    Good for training defensive agents
    """
    
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.target = None
    
    def choose_action(self, game_state):
        """Aggressive food collection - ignore most dangers"""
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP
        
        # Remove STOP
        if Directions.STOP in legal_actions and len(legal_actions) > 1:
            legal_actions.remove(Directions.STOP)
        
        # If carrying 10+ food, return home
        if my_state.num_carrying >= 10:
            return self._go_home(game_state, legal_actions)
        
        # Get food
        food_list = self.get_food(game_state).as_list()
        
        if not food_list:
            return self._go_home(game_state, legal_actions)
        
        # Target furthest food (deep penetration)
        if self.target is None or self.target not in food_list:
            self.target = max(food_list, 
                            key=lambda f: self.get_maze_distance(my_pos, f))
        
        # Move toward target
        best_action = min(legal_actions,
                         key=lambda a: self.get_maze_distance(
                             game_state.generate_successor(self.index, a).get_agent_position(self.index),
                             self.target))
        
        return best_action
    
    def _go_home(self, game_state, legal_actions):
        """Return to home side"""
        walls = game_state.get_walls()
        width = walls.width
        mid_x = width // 2
        
        my_pos = game_state.get_agent_position(self.index)
        
        if self.red:
            target_x = mid_x - 1
        else:
            target_x = mid_x
        
        target = (target_x, my_pos[1])
        
        best_action = min(legal_actions,
                         key=lambda a: self.get_maze_distance(
                             game_state.generate_successor(self.index, a).get_agent_position(self.index),
                             target))
        
        return best_action


class CautiousAgent(CaptureAgent):
    """
    Very cautious defensive agent - never leaves home, patrols conservatively
    """
    
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        
        # Find patrol area
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2
        
        if self.red:
            patrol_x = mid_x - 2
        else:
            patrol_x = mid_x + 1
        
        self.patrol_points = [(patrol_x, y) for y in range(height) 
                             if not walls[patrol_x][y]]
        self.patrol_index = 0
    
    def choose_action(self, game_state):
        """Stay on home side, patrol and chase invaders"""
        my_pos = game_state.get_agent_position(self.index)
        legal_actions = game_state.get_legal_actions(self.index)
        
        if not legal_actions:
            return Directions.STOP
        
        if Directions.STOP in legal_actions and len(legal_actions) > 1:
            legal_actions.remove(Directions.STOP)
        
        # Look for invaders
        invaders = []
        for opponent in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opponent)
            if opp_state.is_pacman:
                opp_pos = game_state.get_agent_position(opponent)
                if opp_pos:
                    invaders.append(opp_pos)
        
        # Chase nearest invader
        if invaders:
            target = min(invaders, key=lambda p: self.get_maze_distance(my_pos, p))
        else:
            # Patrol
            if not self.patrol_points:
                return random.choice(legal_actions)
            
            target = self.patrol_points[self.patrol_index % len(self.patrol_points)]
            
            if self.get_maze_distance(my_pos, target) <= 1:
                self.patrol_index += 1
                target = self.patrol_points[self.patrol_index % len(self.patrol_points)]
        
        # Move toward target
        best_action = min(legal_actions,
                         key=lambda a: self.get_maze_distance(
                             game_state.generate_successor(self.index, a).get_agent_position(self.index),
                             target))
        
        return best_action


class RandomAgent(CaptureAgent):
    """
    Completely random agent - for baseline comparison
    """
    
    def choose_action(self, game_state):
        """Choose random legal action"""
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP
        
        return random.choice(legal_actions)


class GreedyAgent(CaptureAgent):
    """
    Simple greedy agent - always goes for closest food
    No strategic thinking, useful for training
    """
    
    def choose_action(self, game_state):
        """Go to closest food or home if carrying"""
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP
        
        if Directions.STOP in legal_actions and len(legal_actions) > 1:
            legal_actions.remove(Directions.STOP)
        
        # If carrying 3+ food, go home
        if my_state.num_carrying >= 3:
            walls = game_state.get_walls()
            width = walls.width
            mid_x = width // 2
            
            if self.red:
                target_x = mid_x - 1
            else:
                target_x = mid_x
            
            target = (target_x, my_pos[1])
        else:
            # Go to closest food
            food_list = self.get_food(game_state).as_list()
            if not food_list:
                return random.choice(legal_actions)
            
            target = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        
        # Move toward target
        best_action = min(legal_actions,
                         key=lambda a: self.get_maze_distance(
                             game_state.generate_successor(self.index, a).get_agent_position(self.index),
                             target))
        
        return best_action


class MirrorAgent(CaptureAgent):
    """
    Mirrors opponent behavior - useful for testing adaptation
    """
    
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.opponent_history = {}
    
    def choose_action(self, game_state):
        """Try to mirror what opponent did in similar situation"""
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP
        
        # Track opponent positions
        for opponent in self.get_opponents(game_state):
            pos = game_state.get_agent_position(opponent)
            if pos:
                self.opponent_history[opponent] = pos
        
        # Random action with slight bias
        if Directions.STOP in legal_actions and len(legal_actions) > 1:
            legal_actions.remove(Directions.STOP)
        
        return random.choice(legal_actions)
