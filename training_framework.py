"""
Training Framework for Pacman Capture Agent
Uses Q-Learning and Deep Q-Learning to train agents through self-play
"""

import json
import pickle
import random
import time
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any
import numpy as np

class FeatureExtractor:
    """
    Extracts features from game state for learning
    """
    
    @staticmethod
    def extract_offensive_features(game_state, agent) -> Dict[str, float]:
        """Extract features for offensive play"""
        features = {}
        
        my_state = game_state.get_agent_state(agent.index)
        my_pos = game_state.get_agent_position(agent.index)
        
        # Basic state features
        features['is_pacman'] = 1.0 if my_state.is_pacman else 0.0
        features['carrying'] = float(my_state.num_carrying)
        features['score'] = float(agent.get_score(game_state))
        
        # Food features
        food_list = agent.get_food(game_state).as_list()
        features['food_remaining'] = float(len(food_list))
        
        if food_list:
            food_distances = [agent.get_maze_distance(my_pos, food) for food in food_list[:10]]
            features['min_food_distance'] = float(min(food_distances))
            features['avg_food_distance'] = float(sum(food_distances) / len(food_distances))
        else:
            features['min_food_distance'] = 0.0
            features['avg_food_distance'] = 0.0
        
        # Border distance
        if hasattr(agent, 'border_positions') and agent.border_positions:
            border_dists = [agent.get_maze_distance(my_pos, b) for b in agent.border_positions[:5]]
            features['min_border_distance'] = float(min(border_dists))
        else:
            features['min_border_distance'] = 0.0
        
        # Enemy features
        enemies = agent.get_opponents(game_state)
        ghost_distances = []
        pacman_distances = []
        
        for enemy in enemies:
            enemy_state = game_state.get_agent_state(enemy)
            enemy_pos = game_state.get_agent_position(enemy)
            
            if enemy_pos:
                dist = agent.get_maze_distance(my_pos, enemy_pos)
                if enemy_state.is_pacman:
                    pacman_distances.append(dist)
                else:
                    if enemy_state.scared_timer > 0:
                        features['scared_ghost_nearby'] = 1.0
                    ghost_distances.append(dist)
        
        features['min_ghost_distance'] = float(min(ghost_distances)) if ghost_distances else 99.0
        features['min_invader_distance'] = float(min(pacman_distances)) if pacman_distances else 99.0
        features['num_visible_ghosts'] = float(len(ghost_distances))
        features['num_invaders'] = float(len(pacman_distances))
        
        # Capsule features
        capsules = agent.get_capsules(game_state)
        if capsules:
            capsule_distances = [agent.get_maze_distance(my_pos, cap) for cap in capsules]
            features['min_capsule_distance'] = float(min(capsule_distances))
        else:
            features['min_capsule_distance'] = 0.0
        
        # Danger level
        if ghost_distances:
            min_ghost_dist = min(ghost_distances)
            if min_ghost_dist <= 3:
                features['in_danger'] = 1.0
            else:
                features['in_danger'] = 0.0
        else:
            features['in_danger'] = 0.0
        
        return features
    
    @staticmethod
    def extract_action_features(game_state, agent, action) -> Dict[str, float]:
        """Extract features for a specific action"""
        features = FeatureExtractor.extract_offensive_features(game_state, agent)
        
        # Add action-specific features
        successor = agent.get_successor(game_state, action)
        succ_state = successor.get_agent_state(agent.index)
        succ_pos = successor.get_agent_position(agent.index)
        
        # Distance changes
        food_list = agent.get_food(game_state).as_list()
        if food_list:
            current_min = min(agent.get_maze_distance(game_state.get_agent_position(agent.index), f) 
                            for f in food_list[:5])
            next_min = min(agent.get_maze_distance(succ_pos, f) for f in food_list[:5])
            features['food_distance_delta'] = float(current_min - next_min)
        
        # Movement features
        features['action_is_stop'] = 1.0 if action == 'Stop' else 0.0
        features['action_is_reverse'] = 1.0 if agent._is_reverse_action(game_state, action) else 0.0
        
        # Successor carrying
        features['successor_carrying'] = float(succ_state.num_carrying)
        
        return features


class QLearningAgent:
    """
    Q-Learning agent that learns from experience
    """
    
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.1):
        """
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate
        """
        self.q_values = defaultdict(float)  # (state_hash, action) -> Q-value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = True
        
        # Training statistics
        self.episodes = 0
        self.total_reward = 0
        self.episode_rewards = []
        
    def get_q_value(self, state_features: Dict[str, float], action: str) -> float:
        """Get Q-value for state-action pair"""
        state_hash = self._hash_features(state_features)
        return self.q_values.get((state_hash, action), 0.0)
    
    def get_action(self, game_state, agent, legal_actions: List[str]) -> str:
        """Choose action using epsilon-greedy policy"""
        if not legal_actions:
            return 'Stop'
        
        # Exploration
        if self.training and random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        # Exploitation - choose best action
        features = FeatureExtractor.extract_offensive_features(game_state, agent)
        q_values = [(self.get_q_value(features, action), action) for action in legal_actions]
        
        max_q = max(q_values, key=lambda x: x[0])[0]
        best_actions = [action for q, action in q_values if q == max_q]
        
        return random.choice(best_actions)
    
    def update(self, state_features: Dict[str, float], action: str, 
               next_state_features: Dict[str, float], reward: float, 
               legal_next_actions: List[str]):
        """Update Q-value using Q-learning update rule"""
        if not self.training:
            return
        
        current_q = self.get_q_value(state_features, action)
        
        # Get max Q-value for next state
        if legal_next_actions:
            next_q_values = [self.get_q_value(next_state_features, next_action) 
                           for next_action in legal_next_actions]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0.0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        state_hash = self._hash_features(state_features)
        self.q_values[(state_hash, action)] = new_q
        
        self.total_reward += reward
    
    def _hash_features(self, features: Dict[str, float]) -> str:
        """Create hashable representation of features"""
        # Discretize continuous features for better generalization
        discretized = {}
        for key, value in features.items():
            if 'distance' in key:
                # Discretize distances into buckets
                if value < 2:
                    discretized[key] = 0
                elif value < 5:
                    discretized[key] = 1
                elif value < 10:
                    discretized[key] = 2
                else:
                    discretized[key] = 3
            elif key in ['carrying', 'num_visible_ghosts', 'num_invaders']:
                discretized[key] = int(value)
            else:
                discretized[key] = 1 if value > 0.5 else 0
        
        return str(sorted(discretized.items()))
    
    def save(self, filepath: str, metadata: dict = None):
        """Save Q-values to file with metadata"""
        import time
        data = {
            'q_values': dict(self.q_values),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'episode_rewards': self.episode_rewards,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved Q-learning agent to {filepath}")
        print(f"  Episodes: {self.episodes}, Q-values: {len(self.q_values)}, Epsilon: {self.epsilon:.3f}")
    
    def load(self, filepath: str):
        """Load Q-values from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.q_values = defaultdict(float, data['q_values'])
            self.alpha = data.get('alpha', self.alpha)
            self.gamma = data.get('gamma', self.gamma)
            self.epsilon = data.get('epsilon', self.epsilon)
            self.episodes = data.get('episodes', 0)
            self.episode_rewards = data.get('episode_rewards', [])
            timestamp = data.get('timestamp', 0)
            
            print(f"✓ Loaded Q-learning agent from {filepath}")
            print(f"  Episodes trained: {self.episodes}")
            print(f"  Q-values learned: {len(self.q_values)}")
            print(f"  Current epsilon: {self.epsilon:.3f}")
            if self.episode_rewards:
                recent_avg = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                print(f"  Recent avg reward: {recent_avg:.2f}")
            if timestamp:
                import time
                hours_ago = (time.time() - timestamp) / 3600
                print(f"  Last saved: {hours_ago:.1f} hours ago")
            print(f"  → Training will resume from episode {self.episodes + 1}")
            return True
        except FileNotFoundError:
            print(f"ℹ No saved agent found at {filepath}, starting fresh training")
            return False
    
    def episode_end(self, final_score: float):
        """Called at end of episode"""
        self.episodes += 1
        self.episode_rewards.append(self.total_reward)
        
        # Decay epsilon over time
        if self.training and self.episodes % 10 == 0:
            self.epsilon = max(0.01, self.epsilon * 0.99)
        
        # Print progress
        if self.episodes % 10 == 0:
            avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
            print(f"Episode {self.episodes}: Avg Reward (last 10): {avg_reward:.2f}, "
                  f"Epsilon: {self.epsilon:.3f}, Q-values: {len(self.q_values)}")
        
        self.total_reward = 0
    
    def merge(self, other_agent):
        """Merge Q-values from another agent (for distributed training)"""
        merged_count = 0
        for key, value in other_agent.q_values.items():
            if key in self.q_values:
                # Average the Q-values
                self.q_values[key] = (self.q_values[key] + value) / 2
            else:
                # Add new Q-value
                self.q_values[key] = value
                merged_count += 1
        
        # Merge episode statistics
        self.episodes += other_agent.episodes
        self.episode_rewards.extend(other_agent.episode_rewards)
        
        print(f"✓ Merged agent: +{merged_count} new Q-values, total episodes: {self.episodes}")
        return merged_count


class DeepQLearningAgent:
    """
    Deep Q-Learning agent using neural network approximation
    Requires numpy (already available)
    """
    
    def __init__(self, feature_size=20, hidden_size=64, alpha=0.001, gamma=0.9, epsilon=0.1):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = True
        
        # Simple neural network weights (2-layer)
        self.w1 = np.random.randn(feature_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros(1)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        
        # Training statistics
        self.episodes = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []
    
    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to fixed-size vector"""
        # Predefined feature order
        feature_keys = [
            'is_pacman', 'carrying', 'score', 'food_remaining',
            'min_food_distance', 'avg_food_distance', 'min_border_distance',
            'min_ghost_distance', 'min_invader_distance', 'num_visible_ghosts',
            'num_invaders', 'min_capsule_distance', 'in_danger',
            'food_distance_delta', 'action_is_stop', 'action_is_reverse',
            'successor_carrying', 'scared_ghost_nearby'
        ]
        
        vector = np.zeros(self.feature_size)
        for i, key in enumerate(feature_keys[:self.feature_size]):
            vector[i] = features.get(key, 0.0)
        
        # Normalize features
        vector = np.clip(vector / 100.0, -1, 1)
        return vector
    
    def _forward(self, x: np.ndarray) -> float:
        """Forward pass through network"""
        # Hidden layer with ReLU
        hidden = np.maximum(0, np.dot(x, self.w1) + self.b1)
        # Output layer
        output = np.dot(hidden, self.w2) + self.b2
        return output[0]
    
    def get_q_value(self, features: Dict[str, float]) -> float:
        """Get Q-value for state-action pair"""
        x = self._features_to_vector(features)
        return self._forward(x)
    
    def get_action(self, game_state, agent, legal_actions: List[str]) -> str:
        """Choose action using epsilon-greedy policy"""
        if not legal_actions:
            return 'Stop'
        
        # Exploration
        if self.training and random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        # Exploitation
        best_action = None
        best_q = float('-inf')
        
        for action in legal_actions:
            features = FeatureExtractor.extract_action_features(game_state, agent, action)
            q = self.get_q_value(features)
            if q > best_q:
                best_q = q
                best_action = action
        
        return best_action if best_action else legal_actions[0]
    
    def update(self, state_features: Dict[str, float], action: str,
               next_state_features: Dict[str, float], reward: float,
               legal_next_actions: List[str]):
        """Store experience and train if buffer is large enough"""
        if not self.training:
            return
        
        # Store experience
        self.replay_buffer.append((state_features, reward, next_state_features, legal_next_actions))
        self.total_reward += reward
        
        # Train on batch if buffer is large enough
        if len(self.replay_buffer) >= self.batch_size:
            self._train_on_batch()
    
    def _train_on_batch(self):
        """Train on random batch from replay buffer"""
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        total_loss = 0
        for state_features, reward, next_state_features, legal_next_actions in batch:
            # Current prediction
            x = self._features_to_vector(state_features)
            q_pred = self._forward(x)
            
            # Target
            if legal_next_actions:
                # Get best next Q-value
                max_next_q = max(self.get_q_value(
                    FeatureExtractor.extract_action_features(None, None, next_action)
                ) for next_action in legal_next_actions if next_action != 'Stop')
            else:
                max_next_q = 0.0
            
            q_target = reward + self.gamma * max_next_q
            
            # Gradient descent
            loss = (q_pred - q_target) ** 2
            total_loss += loss
            
            # Backprop (simple gradient)
            grad_output = 2 * (q_pred - q_target)
            
            # Update weights (simplified backprop)
            hidden = np.maximum(0, np.dot(x, self.w1) + self.b1)
            
            # Output layer gradients
            grad_w2 = hidden.reshape(-1, 1) * grad_output
            grad_b2 = grad_output
            
            # Hidden layer gradients
            grad_hidden = grad_output * self.w2.flatten()
            grad_hidden[hidden <= 0] = 0  # ReLU derivative
            grad_w1 = x.reshape(-1, 1) * grad_hidden
            grad_b1 = grad_hidden
            
            # Update with learning rate
            self.w2 -= self.alpha * grad_w2
            self.b2 -= self.alpha * grad_b2
            self.w1 -= self.alpha * grad_w1
            self.b1 -= self.alpha * grad_b1
        
        self.losses.append(total_loss / self.batch_size)
    
    def save(self, filepath: str, metadata: dict = None):
        """Save network weights with metadata"""
        import time
        data = {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        avg_loss = np.mean(self.losses[-100:]) if len(self.losses) >= 100 else 0
        print(f"✓ Saved DQN agent to {filepath}")
        print(f"  Episodes: {self.episodes}, Epsilon: {self.epsilon:.3f}, Avg Loss: {avg_loss:.4f}")
    
    def load(self, filepath: str):
        """Load network weights"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.w1 = data['w1']
            self.b1 = data['b1']
            self.w2 = data['w2']
            self.b2 = data['b2']
            self.alpha = data.get('alpha', self.alpha)
            self.gamma = data.get('gamma', self.gamma)
            self.epsilon = data.get('epsilon', self.epsilon)
            self.episodes = data.get('episodes', 0)
            self.episode_rewards = data.get('episode_rewards', [])
            self.losses = data.get('losses', [])
            timestamp = data.get('timestamp', 0)
            
            print(f"✓ Loaded DQN agent from {filepath}")
            print(f"  Episodes trained: {self.episodes}")
            avg_loss = np.mean(self.losses[-100:]) if len(self.losses) >= 100 else 0
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Current epsilon: {self.epsilon:.3f}")
            if timestamp:
                import time
                hours_ago = (time.time() - timestamp) / 3600
                print(f"  Last saved: {hours_ago:.1f} hours ago")
            print(f"  → Training will resume from episode {self.episodes + 1}")
            return True
        except FileNotFoundError:
            print(f"ℹ No saved DQN found at {filepath}, starting fresh training")
            return False
    
    def episode_end(self, final_score: float):
        """Called at end of episode"""
        self.episodes += 1
        self.episode_rewards.append(self.total_reward)
        
        # Decay epsilon
        if self.training and self.episodes % 10 == 0:
            self.epsilon = max(0.01, self.epsilon * 0.99)
        
        # Print progress
        if self.episodes % 10 == 0:
            avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
            avg_loss = np.mean(self.losses[-100:]) if len(self.losses) >= 100 else 0
            print(f"Episode {self.episodes}: Avg Reward: {avg_reward:.2f}, "
                  f"Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}")
        
        self.total_reward = 0


class OpponentModel:
    """
    Models opponent behavior to predict their actions
    """
    
    def __init__(self):
        self.position_history = defaultdict(list)  # {agent_index: [positions]}
        self.action_patterns = defaultdict(lambda: defaultdict(int))  # {agent_index: {pattern: count}}
        self.food_targeting = defaultdict(list)  # {agent_index: [targeted_food]}
        
    def observe(self, game_state, opponent_index: int):
        """Observe opponent behavior"""
        opponent_pos = game_state.get_agent_position(opponent_index)
        if opponent_pos:
            self.position_history[opponent_index].append(opponent_pos)
            
            # Keep only recent history
            if len(self.position_history[opponent_index]) > 20:
                self.position_history[opponent_index] = self.position_history[opponent_index][-20:]
    
    def predict_target(self, game_state, opponent_index: int) -> Tuple[int, int]:
        """Predict where opponent is heading"""
        history = self.position_history.get(opponent_index, [])
        
        if len(history) < 2:
            return None
        
        # Calculate direction vector from recent positions
        recent = history[-5:]
        if len(recent) >= 2:
            dx = recent[-1][0] - recent[0][0]
            dy = recent[-1][1] - recent[0][1]
            
            # Predict next position
            predicted = (recent[-1][0] + dx, recent[-1][1] + dy)
            return predicted
        
        return None
    
    def get_tendency(self, opponent_index: int) -> str:
        """Get opponent's play style tendency"""
        history = self.position_history.get(opponent_index, [])
        
        if len(history) < 10:
            return 'unknown'
        
        # Analyze position patterns
        recent = history[-10:]
        x_positions = [pos[0] for pos in recent]
        y_positions = [pos[1] for pos in recent]
        
        x_variance = np.var(x_positions) if len(x_positions) > 1 else 0
        y_variance = np.var(y_positions) if len(y_positions) > 1 else 0
        
        if x_variance < 2 and y_variance < 2:
            return 'defensive'  # Stays in one area (defending)
        elif x_variance > 10 or y_variance > 10:
            return 'aggressive'  # Moves around a lot (attacking)
        else:
            return 'balanced'
    
    def save(self, filepath: str):
        """Save opponent model"""
        data = {
            'position_history': dict(self.position_history),
            'action_patterns': dict(self.action_patterns),
            'food_targeting': dict(self.food_targeting)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Load opponent model"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.position_history = defaultdict(list, data.get('position_history', {}))
            self.action_patterns = defaultdict(lambda: defaultdict(int), data.get('action_patterns', {}))
            self.food_targeting = defaultdict(list, data.get('food_targeting', {}))
            print(f"Loaded opponent model from {filepath}")
        except FileNotFoundError:
            print(f"No opponent model found at {filepath}")
