"""
Learning-enabled Pacman Agent
Integrates Q-Learning for continuous improvement
"""

import random
import time
from typing import List, Tuple, Dict, Optional

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point, manhattan_distance

# Import our training framework
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_framework import QLearningAgent, DeepQLearningAgent, FeatureExtractor, OpponentModel


# Global learners (shared across games)
offensive_learner = None
defensive_learner = None
opponent_model = OpponentModel()

def create_team(first_index, second_index, is_red,
               first='LearningOffensiveAgent', second='LearningDefensiveAgent', num_training=0):
    """
    Creates a team with learning-enabled agents
    num_training: number of training games (vs evaluation games)
    """
    global offensive_learner, defensive_learner
    
    # Initialize learners if not already done
    if offensive_learner is None:
        offensive_learner = QLearningAgent(alpha=0.2, gamma=0.9, epsilon=0.3)
        # Try to load existing knowledge
        offensive_learner.load('offensive_qlearning.pkl')
    
    if defensive_learner is None:
        defensive_learner = QLearningAgent(alpha=0.2, gamma=0.9, epsilon=0.3)
        defensive_learner.load('defensive_qlearning.pkl')
    
    # Set training mode
    is_training = num_training > 0
    offensive_learner.training = is_training
    defensive_learner.training = is_training
    
    return [eval(first)(first_index), eval(second)(second_index)]


class LearningOffensiveAgent(CaptureAgent):
    """
    Offensive agent that learns from experience using Q-Learning
    """
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.learner = offensive_learner
        
        # Track previous state for learning
        self.previous_state_features = None
        self.previous_action = None
        self.previous_score = 0
        self.previous_carrying = 0
        
        # Precomputed data
        self.border_positions = []
        
    def register_initial_state(self, game_state):
        """Initialize agent"""
        CaptureAgent.register_initial_state(self, game_state)
        
        # Find border positions
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2
        
        if self.red:
            border_x = mid_x - 1
        else:
            border_x = mid_x
        
        self.border_positions = []
        for y in range(height):
            if not walls[border_x][y]:
                self.border_positions.append((border_x, y))
        
        # Reset episode tracking
        self.previous_score = 0
        self.previous_carrying = 0
    
    def choose_action(self, game_state):
        """Choose action using learned policy"""
        start_time = time.time()
        
        try:
            # Get legal actions
            legal_actions = game_state.get_legal_actions(self.index)
            if not legal_actions:
                return Directions.STOP
            
            # Remove STOP unless necessary
            if Directions.STOP in legal_actions and len(legal_actions) > 1:
                legal_actions.remove(Directions.STOP)
            
            # Learn from previous experience
            if self.previous_state_features is not None and self.previous_action is not None:
                reward = self._calculate_reward(game_state)
                current_features = FeatureExtractor.extract_offensive_features(game_state, self)
                
                self.learner.update(
                    self.previous_state_features,
                    self.previous_action,
                    current_features,
                    reward,
                    legal_actions
                )
            
            # Choose action
            action = self.learner.get_action(game_state, self, legal_actions)
            
            # Fallback to safe action if computation takes too long
            if time.time() - start_time > 0.8:
                action = self._get_safe_action(game_state, legal_actions)
            
            # Store current state for next update
            self.previous_state_features = FeatureExtractor.extract_offensive_features(game_state, self)
            self.previous_action = action
            self.previous_score = self.get_score(game_state)
            self.previous_carrying = game_state.get_agent_state(self.index).num_carrying
            
            return action
            
        except Exception as e:
            print(f"Error in LearningOffensiveAgent: {e}")
            return random.choice(legal_actions) if legal_actions else Directions.STOP
    
    def _calculate_reward(self, game_state) -> float:
        """Calculate reward for the previous action"""
        reward = 0.0
        
        current_score = self.get_score(game_state)
        my_state = game_state.get_agent_state(self.index)
        current_carrying = my_state.num_carrying
        
        # Reward for scoring points
        score_delta = current_score - self.previous_score
        reward += score_delta * 10.0
        
        # Reward for picking up food
        carrying_delta = current_carrying - self.previous_carrying
        if carrying_delta > 0:
            reward += 2.0
        
        # Penalty for losing food (getting caught)
        if carrying_delta < 0 and current_carrying == 0:
            reward -= 5.0 * abs(carrying_delta)
        
        # Reward for returning food
        if self.previous_carrying > 0 and current_carrying == 0 and score_delta > 0:
            reward += 5.0
        
        # Small penalty for time (encourages efficiency)
        reward -= 0.01
        
        # Penalty for dying
        my_pos = game_state.get_agent_position(self.index)
        if my_pos == game_state.get_initial_agent_position(self.index):
            reward -= 10.0
        
        return reward
    
    def _get_safe_action(self, game_state, legal_actions: List[str]) -> str:
        """Fallback safe action when time is running out"""
        my_pos = game_state.get_agent_position(self.index)
        
        # Move toward nearest border
        if self.border_positions:
            target = min(self.border_positions, key=lambda b: self.get_maze_distance(my_pos, b))
            
            best_action = legal_actions[0]
            best_dist = float('inf')
            
            for action in legal_actions:
                successor = self.get_successor(game_state, action)
                next_pos = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(next_pos, target)
                
                if dist < best_dist:
                    best_dist = dist
                    best_action = action
            
            return best_action
        
        return random.choice(legal_actions)
    
    def get_successor(self, game_state, action):
        """Get successor state"""
        return game_state.generate_successor(self.index, action)
    
    def _is_reverse_action(self, game_state, action) -> bool:
        """Check if action reverses previous direction"""
        # Implementation for feature extraction
        return False
    
    def final(self, game_state):
        """Called at game end"""
        # Final update
        if self.previous_state_features is not None:
            final_reward = self.get_score(game_state) * 5.0  # Big reward/penalty for final score
            self.learner.update(
                self.previous_state_features,
                self.previous_action,
                FeatureExtractor.extract_offensive_features(game_state, self),
                final_reward,
                []
            )
        
        # Episode end
        final_score = self.get_score(game_state)
        self.learner.episode_end(final_score)
        
        # Save after every game for distributed training
        self.learner.save('offensive_qlearning.pkl')


class LearningDefensiveAgent(CaptureAgent):
    """
    Defensive agent that learns from experience
    """
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.learner = defensive_learner
        
        # Track previous state
        self.previous_state_features = None
        self.previous_action = None
        self.previous_invaders = 0
        self.previous_food_defending = 0
        
        # Precomputed data
        self.border_positions = []
        
    def register_initial_state(self, game_state):
        """Initialize agent"""
        CaptureAgent.register_initial_state(self, game_state)
        
        # Find border positions
        walls = game_state.get_walls()
        width, height = walls.width, walls.height
        mid_x = width // 2
        
        if self.red:
            border_x = mid_x - 1
        else:
            border_x = mid_x
        
        self.border_positions = []
        for y in range(height):
            if not walls[border_x][y]:
                self.border_positions.append((border_x, y))
        
        # Reset tracking
        self.previous_invaders = 0
        self.previous_food_defending = len(self.get_food_you_are_defending(game_state).as_list())
    
    def choose_action(self, game_state):
        """Choose action using learned policy"""
        try:
            legal_actions = game_state.get_legal_actions(self.index)
            if not legal_actions:
                return Directions.STOP
            
            # Remove STOP
            if Directions.STOP in legal_actions and len(legal_actions) > 1:
                legal_actions.remove(Directions.STOP)
            
            # Learn from previous experience
            if self.previous_state_features is not None:
                reward = self._calculate_reward(game_state)
                current_features = FeatureExtractor.extract_offensive_features(game_state, self)
                
                self.learner.update(
                    self.previous_state_features,
                    self.previous_action,
                    current_features,
                    reward,
                    legal_actions
                )
            
            # Choose action
            action = self.learner.get_action(game_state, self, legal_actions)
            
            # Store state
            self.previous_state_features = FeatureExtractor.extract_offensive_features(game_state, self)
            self.previous_action = action
            self.previous_invaders = len([o for o in self.get_opponents(game_state) 
                                         if game_state.get_agent_state(o).is_pacman])
            self.previous_food_defending = len(self.get_food_you_are_defending(game_state).as_list())
            
            return action
            
        except Exception as e:
            print(f"Error in LearningDefensiveAgent: {e}")
            return random.choice(legal_actions) if legal_actions else Directions.STOP
    
    def _calculate_reward(self, game_state) -> float:
        """Calculate reward for defensive play"""
        reward = 0.0
        
        # Count current invaders
        current_invaders = len([o for o in self.get_opponents(game_state)
                               if game_state.get_agent_state(o).is_pacman])
        
        # Reward for reducing invaders (eating them)
        if current_invaders < self.previous_invaders:
            reward += 10.0
        
        # Penalty for food being eaten
        current_food = len(self.get_food_you_are_defending(game_state).as_list())
        food_delta = current_food - self.previous_food_defending
        if food_delta < 0:
            reward -= 5.0 * abs(food_delta)
        
        # Small reward for being near invaders
        if current_invaders > 0:
            my_pos = game_state.get_agent_position(self.index)
            invader_positions = [game_state.get_agent_position(o) 
                                for o in self.get_opponents(game_state)
                                if game_state.get_agent_state(o).is_pacman]
            
            if invader_positions:
                min_dist = min(self.get_maze_distance(my_pos, pos) 
                              for pos in invader_positions if pos)
                if min_dist <= 3:
                    reward += 1.0
        
        # Small penalty for time
        reward -= 0.01
        
        return reward
    
    def get_successor(self, game_state, action):
        """Get successor state"""
        return game_state.generate_successor(self.index, action)
    
    def _is_reverse_action(self, game_state, action) -> bool:
        """Check if action reverses previous direction"""
        return False
    
    def final(self, game_state):
        """Called at game end"""
        if self.previous_state_features is not None:
            final_reward = -self.get_score(game_state) * 3.0  # Defensive perspective
            self.learner.update(
                self.previous_state_features,
                self.previous_action,
                FeatureExtractor.extract_offensive_features(game_state, self),
                final_reward,
                []
            )
        
        final_score = self.get_score(game_state)
        self.learner.episode_end(final_score)
        
        # Save after every game for distributed training
        self.learner.save('defensive_qlearning.pkl')
