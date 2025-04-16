# import numpy as np
# import tensorflow as tf
# import collections
# import random
# import math
# import os
# import networkx as nx
# from typing import Dict, List, NamedTuple, Optional, Tuple, Set

# print("Imported all files")

# if not os.path.exists('checkpoint'):
#     os.makedirs('checkpoint')
    
# print("Created checkpoint directory if it didn't exist")

# class GraphEnvironment:
    
#     def __init__(self, max_nodes=10):
#         self.max_nodes = max_nodes
#         self.reset()

#     # Generate a random graph and reset the environment state
#     def reset(self):
        
#         # Generate random number of nodes
#         self.num_nodes = random.randint(5, self.max_nodes)

#         # Create adjacency matrix (random graph)
#         G = nx.complete_graph(self.num_nodes)
#         self.adjacency_matrix = nx.to_numpy_array(G, dtype=int)

#         # Set start node
#         self.start_node = random.randint(0, self.num_nodes - 1)

#         # DFS state
#         self.current_node = self.start_node
#         self.visited = set([self.start_node])
#         self.stack = [self.start_node]
#         self.steps = 0
#         self.done = False

#         return self.get_observation()

#     # Get the current observation
#     def get_observation(self):
#         # One-hot encoding of current node
#         current_node_encoding = np.zeros(self.max_nodes)
#         current_node_encoding[self.current_node] = 1

#         # One-hot encoding of visited nodes
#         visited_encoding = np.zeros(self.max_nodes)
#         for node in self.visited:
#             visited_encoding[node] = 1

#         # Stack representation
#         stack_encoding = np.zeros(self.max_nodes)
#         for node in self.stack:
#             stack_encoding[node] = 1

#         # Adjacency matrix padded to max_nodes
#         padded_adjacency = np.zeros((self.max_nodes, self.max_nodes))
#         padded_adjacency[:self.num_nodes, :self.num_nodes] = self.adjacency_matrix

#         return {
#             'current_node': current_node_encoding,
#             'visited': visited_encoding,
#             'stack': stack_encoding,
#             'adjacency': padded_adjacency.flatten(),
#             'num_nodes': self.num_nodes
#         }

#     # Get the legal actions from the current state
#     def get_legal_actions(self):
#         if not self.stack:
#             return []  # No actions if stack is empty

#         current = self.stack[-1]
#         legal_actions = []

#         # Option 1: Pop from stack (always legal if stack not empty)
#         legal_actions.append(self.max_nodes)  # Pop action

#         # Option 2: Visit unvisited neighbors
#         for neighbor in range(self.num_nodes):
#             if (self.adjacency_matrix[current, neighbor] == 1 and
#                 neighbor not in self.visited):
#                 legal_actions.append(neighbor)

#         return legal_actions

#     # Take action in the environment
#     def step(self, action):
#         self.steps += 1
#         reward = -0.05  # Slightly higher negative reward to encourage efficiency
        
#         if action == self.max_nodes:  # Pop action
#             if self.stack:
#                 self.stack.pop()
#                 if not self.stack:  # If stack empty, we're done
#                     self.done = True
#                 else:
#                     self.current_node = self.stack[-1]
#         else:  # Visit a new node
#             if action not in self.visited:
#                 self.visited.add(action)
#                 self.stack.append(action)
#                 self.current_node = action
#                 # Increased reward for visiting new nodes
#                 reward += 2.0  
                
#                 # Additional reward as we get closer to visiting all nodes
#                 completion_percentage = len(self.visited) / self.num_nodes
#                 reward += completion_percentage * 0.5

#         # Check if DFS is complete (all nodes visited)
#         if len(self.visited) == self.num_nodes:
#             # Dramatically increased reward for full completion
#             reward += 15.0 
#             # We only call it done if we've visited all nodes
#             if not self.stack:
#                 self.done = True
                
#         # Penalty for exceeding max steps, increased to discourage wandering
#         if self.steps > 10 * self.num_nodes:
#             reward -= 10.0
#             self.done = True

#         return self.get_observation(), reward, self.done
    
# print("GraphEnvironment initialized")
    
# class DFSNetwork:
#     def __init__(self, max_nodes=10):
#         self.max_nodes = max_nodes
#         self.action_space_size = max_nodes + 1  # Nodes + pop action
#         self.input_dim = 3 * max_nodes + max_nodes * max_nodes

#         # Create the model
#         self.model = self._build_model()
#         self.target_model = self._build_model()
#         self.update_target_network()

#     def _build_model(self):
#         """Build the neural network model with GPU support."""
#         # Ensure TensorFlow uses GPU if available
#         physical_devices = tf.config.list_physical_devices('GPU')
#         if physical_devices:
#             try:
#                 tf.config.experimental.set_memory_growth(physical_devices[0], True)
#                 print("Using GPU:", physical_devices[0])
#             except RuntimeError as e:
#                 print("Error setting GPU memory growth:", e)
#         else:
#             print("No GPU found, using CPU.")

#         input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        
#         # Wider and deeper network for better representation
#         x1 = tf.keras.layers.Dense(256, activation='relu')(input_layer)
#         x1 = tf.keras.layers.BatchNormalization()(x1)
#         x2 = tf.keras.layers.Dense(256, activation='relu')(x1)
#         x2 = tf.keras.layers.BatchNormalization()(x2)
#         x3 = tf.keras.layers.Dense(128, activation='relu')(x2)

#         # Policy head (action probabilities)
#         policy_head = tf.keras.layers.Dense(128, activation='relu')(x3)
#         policy_output = tf.keras.layers.Dense(self.action_space_size, activation='softmax', name='policy')(policy_head)

#         # Value head (state value estimation)
#         value_head = tf.keras.layers.Dense(128, activation='relu')(x3)
#         value_output = tf.keras.layers.Dense(1, name='value')(value_head)

#         model = tf.keras.Model(inputs=input_layer, outputs=[policy_output, value_output])
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Lower learning rate for stability
#             loss={
#                 'policy': 'categorical_crossentropy',
#                 'value': 'mse'
#             }
#         )

#         return model

#     def update_target_network(self):
#         """Update target network with current weights."""
#         self.target_model.set_weights(self.model.get_weights())

#     def predict(self, state):
#         """Make a prediction based on the state."""
#         # Prepare the input
#         current_node = state['current_node']
#         visited = state['visited']
#         stack = state['stack']
#         adjacency = state['adjacency']

#         # Concatenate all features
#         input_features = np.concatenate([current_node, visited, stack, adjacency])
#         input_features = np.expand_dims(input_features, axis=0)

#         # Get prediction
#         policy, value = self.model.predict(input_features, verbose=0)

#         return policy[0], value[0][0]

#     def target_predict(self, state):
#         """Make a prediction with the target network."""
#         # Prepare the input
#         current_node = state['current_node']
#         visited = state['visited']
#         stack = state['stack']
#         adjacency = state['adjacency']

#         # Concatenate all features
#         input_features = np.concatenate([current_node, visited, stack, adjacency])
#         input_features = np.expand_dims(input_features, axis=0)

#         # Get prediction
#         policy, value = self.target_model.predict(input_features, verbose=0)

#         return policy[0], value[0][0]

#     def train(self, states, actions, rewards, next_states, dones):
#         """Train the network with a batch of experiences."""
#         # Prepare inputs
#         inputs = []
#         for state in states:
#             input_features = np.concatenate([
#                 state['current_node'],
#                 state['visited'],
#                 state['stack'],
#                 state['adjacency']
#             ])
#             inputs.append(input_features)
#         inputs = np.array(inputs)

#         # Get target values using target network
#         next_values = []
#         for next_state in next_states:
#             if next_state is None:  # Terminal state
#                 next_values.append(0)
#             else:
#                 _, value = self.target_predict(next_state)
#                 next_values.append(value)

#         # Calculate target values
#         target_values = []
#         for i in range(len(rewards)):
#             if dones[i]:
#                 target_values.append(rewards[i])
#             else:
#                 target_values.append(rewards[i] + 0.99 * next_values[i])

#         # One-hot encode actions
#         action_indices = np.array(actions)
#         action_targets = np.zeros((len(actions), self.action_space_size))
#         for i, action in enumerate(action_indices):
#             action_targets[i, action] = 1

#         # Train the model
#         self.model.fit(
#             inputs,
#             [action_targets, np.array(target_values).reshape(-1, 1)],
#             verbose=0,
#             batch_size=32
#         )

#     def save(self, filepath):
#         """Save the model to disk."""
#         if not filepath.endswith('.weights.h5'):
#             filepath = filepath + '.weights.h5'
#         self.model.save_weights(filepath)

#     def load(self, filepath):
#         """Load the model from disk."""
#         if not filepath.endswith('.weights.h5'):
#             filepath = filepath + '.weights.h5'
#         self.model.load_weights(filepath)
#         self.update_target_network()
      
# print("DFSNetwork initialized")

# class Node:
#     """Node in the MCTS tree."""

#     def __init__(self, prior=0.0):
#         self.visit_count = 0
#         self.prior = prior
#         self.value_sum = 0
#         self.children = {}
#         self.state = None
#         self.reward = 0
#         self.done = False

#     def expanded(self):
#         return bool(self.children)

#     def value(self):
#         if self.visit_count == 0:
#             return 0
#         return self.value_sum / self.visit_count

# class MCTS:
#     """Monte Carlo Tree Search implementation for DFS."""

#     def __init__(self, network, num_simulations=50, c_puct=2.0):  # Increased exploration parameter
#         self.network = network
#         self.num_simulations = num_simulations
#         self.c_puct = c_puct

#     def run(self, env, temperature=1.0):
#         """Run MCTS from the current environment state."""
#         # Create root node
#         root = Node(0)
#         root.state = env.get_observation()

#         # Get network prediction for root
#         policy, value = self.network.predict(root.state)

#         # Initialize root with legal actions
#         legal_actions = env.get_legal_actions()
#         for action in legal_actions:
#             root.children[action] = Node(prior=policy[action])

#         # Run simulations
#         for _ in range(self.num_simulations):
#             self._simulate(root, env.max_nodes, env)

#         # Select action based on visit count
#         action_visits = [(action, child.visit_count) for action, child in root.children.items()]
#         if not action_visits:
#             return None, root  # No legal actions

#         # Temperature-based action selection
#         if temperature == 0:  # Deterministic selection
#             action = max(action_visits, key=lambda x: x[1])[0]
#         else:  # Sample based on visit count distribution
#             visits = np.array([x[1] for x in action_visits])
#             # Add epsilon to avoid division by zero
#             visits = np.power(visits + 1e-8, 1.0 / temperature)
#             visits = visits / np.sum(visits)
#             action = np.random.choice([x[0] for x in action_visits], p=visits)

#         return action, root

#     def _simulate(self, node, max_nodes, real_env):
#         """Run a single MCTS simulation."""
#         # If node not expanded, expand it
#         if not node.expanded():
#             # Create a copy of the environment to simulate
#             env = GraphEnvironment(max_nodes)
#             env.num_nodes = real_env.num_nodes
#             env.adjacency_matrix = real_env.adjacency_matrix.copy()
#             env.start_node = real_env.start_node
#             env.current_node = real_env.current_node
#             env.visited = real_env.visited.copy()
#             env.stack = real_env.stack.copy()
#             env.steps = real_env.steps
#             env.done = real_env.done

#             # Get network prediction
#             policy, value = self.network.predict(node.state)

#             # Initialize children with legal actions
#             legal_actions = real_env.get_legal_actions()
#             for action in legal_actions:
#                 node.children[action] = Node(prior=policy[action])

#             return value

#         # Select child with highest UCB score
#         action, child = self._select_child(node)

#         # Create a copy of the environment to simulate
#         env = GraphEnvironment(max_nodes)
#         env.num_nodes = real_env.num_nodes
#         env.adjacency_matrix = real_env.adjacency_matrix.copy()
#         env.start_node = real_env.start_node
#         env.current_node = real_env.current_node
#         env.visited = real_env.visited.copy()
#         env.stack = real_env.stack.copy()
#         env.steps = real_env.steps
#         env.done = real_env.done

#         # Take action in the simulation
#         next_state, reward, done = env.step(action)

#         # Store the state in the child node
#         child.state = next_state
#         child.reward = reward
#         child.done = done

#         # Get value from child
#         if done:
#             value = 0
#         else:
#             value = self._simulate(child, max_nodes, env)

#         # Update statistics
#         child.value_sum += value + reward
#         child.visit_count += 1

#         return value + reward

#     def _select_child(self, node):
#         """Select child with highest UCB score."""
#         sqrt_sum = math.sqrt(sum(child.visit_count for child in node.children.values()))

#         def ucb_score(action, child):
#             # Exploitation term
#             if child.visit_count > 0:
#                 q_value = child.value_sum / child.visit_count
#             else:
#                 q_value = 0

#             # Exploration term, increased weight
#             u_value = self.c_puct * child.prior * sqrt_sum / (1 + child.visit_count)

#             return q_value + u_value

#         return max(node.children.items(), key=lambda x: ucb_score(*x))
    
# print("MCTS initialized")

# # Replay Memory for experience replay
# class ReplayMemory:
#     """Replay memory for storing experiences."""

#     def __init__(self, capacity=50000):  # Increased capacity for better learning
#         self.memory = collections.deque(maxlen=capacity)

#     def add(self, state, action, reward, next_state, done):
#         """Add an experience to memory."""
#         self.memory.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         """Sample a batch of experiences."""
#         if len(self.memory) < batch_size:
#             batch_size = len(self.memory)

#         batch = random.sample(self.memory, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         return states, actions, rewards, next_states, dones

#     def __len__(self):
#         return len(self.memory)
    
# # Training loop with checkpointing and improved exploration
# def train_dfs_agent(episodes=2000, max_nodes=10, checkpoint_dir='checkpoint'):
#     # Create checkpoint directory if it doesn't exist
#     os.makedirs(checkpoint_dir, exist_ok=True)

#     # Initialize environment and agent
#     env = GraphEnvironment(max_nodes)
#     network = DFSNetwork(max_nodes)
#     mcts = MCTS(network, num_simulations=50)  # Increased simulations for better planning
#     memory = ReplayMemory()

#     # Training statistics
#     rewards_history = []
#     nodes_visited_history = []
#     completion_rate_history = []

#     # Check if there's a checkpoint to resume from
#     latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

#     start_episode = 1
#     if latest_checkpoint:
#         print(f"Restoring from checkpoint: {latest_checkpoint}")
#         network.load(latest_checkpoint)

#         # Extract episode number from checkpoint name
#         import re
#         match = re.search(r'episode-(\d+)', latest_checkpoint)
#         if match:
#             start_episode = int(match.group(1)) + 1
#             print(f"Resuming training from episode {start_episode}")

#     # Epsilon for exploration (gradually decreases)
#     epsilon_start = 1
#     epsilon_end = 0.05
#     epsilon_decay = episodes * 0.7 

#     for episode in range(start_episode, episodes+1):
#         state = env.reset()
#         done = False
#         episode_reward = 0
        
#         # Calculate epsilon for this episode (linear decay)
#         epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * 
#                       min(1.0, episode / epsilon_decay))
        
#         # Temperature for MCTS exploration
#         temperature = max(0.2, 1.0 - (episode / (episodes*0.8))) 
        

#         while not done:
#             # Sometimes take random action for exploration
#             if random.random() < epsilon:
#                 legal_actions = env.get_legal_actions()
#                 if not legal_actions:
#                     break
#                 action = random.choice(legal_actions)
#                 # Create a dummy root for consistency
#                 root = Node(0)
#             else:
#                 # Run MCTS to get action
#                 action, root = mcts.run(env, temperature=temperature)
                
#                 if action is None:
#                     break

#             # Take action in environment
#             next_state, reward, done = env.step(action)
#             episode_reward += reward

#             # Store experience
#             memory.add(state, action, reward, next_state if not done else None, done)

#             # Move to next state
#             state = next_state

#             # Train network more frequently
#             if len(memory) >= 256:
#                 # Train multiple batches per step for better learning
#                 for _ in range(3):
#                     states, actions, rewards, next_states, dones = memory.sample(256)
#                     network.train(states, actions, rewards, next_states, dones)

#         # Update target network periodically
#         if episode % 5 == 0:  # More frequent updates
#             network.update_target_network()

#         # Track statistics
#         rewards_history.append(episode_reward)
#         nodes_visited_history.append(len(env.visited))
#         completion_rate = len(env.visited) / env.num_nodes
#         completion_rate_history.append(completion_rate)

#         # Save checkpoint periodically
#         if episode % 50 == 0 or episode == episodes:
#             save_path = os.path.join(checkpoint_dir, f"dfs_model_episode-{episode}")
#             network.save(save_path)
#             print(f"Checkpoint saved at episode {episode} to {save_path}.weights.h5")

#         # Print progress with more detailed statistics
#         if episode % 20 == 0:  # More frequent feedback
#             recent_rewards = rewards_history[-20:]
#             recent_nodes = nodes_visited_history[-20:] 
#             recent_completion = completion_rate_history[-20:]
            
#             avg_reward = sum(recent_rewards) / len(recent_rewards)
#             avg_nodes = sum(recent_nodes) / len(recent_nodes)
#             avg_completion = sum(recent_completion) / len(recent_completion) * 100
            
#             print(f"Episode {episode}/{episodes} - Avg Reward: {avg_reward:.2f}, Avg Nodes: {avg_nodes:.2f}, Completion: {avg_completion:.1f}%")
#             print(f"  Epsilon: {epsilon:.3f}, Temperature: {temperature:.3f}")

#     return network, rewards_history, nodes_visited_history

# print("Training function defined")

# # Function to evaluate the model
# def evaluate_model(network, num_tests=10, max_nodes=10, verbose=True):
#     env = GraphEnvironment(max_nodes)
#     completion_rates = []
#     rewards = []
    
#     for test in range(num_tests):
#         state = env.reset()
#         done = False
#         total_reward = 0
        
#         if verbose and test == 0:
#             print(f"\nDetailed evaluation on test graph:")
#             print(f"Number of nodes: {env.num_nodes}")
#             print(f"Starting node: {env.start_node}")
#             visited_order = [env.start_node]
        
#         step_count = 0
#         while not done and step_count < 3 * env.num_nodes:
#             step_count += 1
            
#             # Use the network directly for evaluation
#             policy, _ = network.predict(state)
            
#             # Get legal actions
#             legal_actions = env.get_legal_actions()
#             if not legal_actions:
#                 break
                
#             # Filter policy to legal actions and add noise for exploration
#             legal_policy = np.array([policy[action] for action in legal_actions])
#             legal_policy = legal_policy / legal_policy.sum()  # Normalize
            
#             # Select action (slightly stochastic for evaluation)
#             action_idx = np.random.choice(len(legal_actions), p=legal_policy)
#             action = legal_actions[action_idx]
            
#             # Take action
#             state, reward, done = env.step(action)
#             total_reward += reward
            
#             if verbose and test == 0 and action != env.max_nodes:
#                 visited_order.append(action)
        
#         completion_rate = len(env.visited) / env.num_nodes
#         completion_rates.append(completion_rate)
#         rewards.append(total_reward)
        
#         if verbose and test == 0:
#             print(f"DFS traversal order: {visited_order}")
#             print(f"Total nodes visited: {len(env.visited)}/{env.num_nodes} ({completion_rate*100:.1f}%)")
#             print(f"Total reward: {total_reward:.2f}")
    
#     avg_completion = sum(completion_rates) / len(completion_rates) * 100
#     avg_reward = sum(rewards) / len(rewards)
    
#     print(f"\nEvaluation over {num_tests} tests:")
#     print(f"Average completion rate: {avg_completion:.1f}%")
#     print(f"Average reward: {avg_reward:.2f}")
    
#     return avg_completion, avg_reward

# print("Starting training...")
# trained_network, rewards, nodes = train_dfs_agent(episodes=2000, max_nodes=10)

# # Evaluate the trained model
# evaluate_model(trained_network, num_tests=20)


import numpy as np
import tensorflow as tf
import collections
import random
import math
import os
import networkx as nx
from typing import Dict, List, NamedTuple, Optional, Tuple, Set
import copy

print("Imported all files")

if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')
    
print("Created checkpoint directory if it didn't exist")

class GraphEnvironment:
    
    def __init__(self, max_nodes=10):
        self.max_nodes = max_nodes
        self.reset()

    # Generate a random graph and reset the environment state
    def reset(self):
        
        # Generate random number of nodes
        self.num_nodes = random.randint(5, self.max_nodes)

        # Create adjacency matrix (random graph)
        G = nx.complete_graph(self.num_nodes)
        self.adjacency_matrix = nx.to_numpy_array(G, dtype=int)

        # Set start node
        self.start_node = random.randint(0, self.num_nodes - 1)

        # DFS state
        self.current_node = self.start_node
        self.visited = set([self.start_node])
        self.stack = [self.start_node]
        self.steps = 0
        self.done = False

        return self.get_observation()
    
    # Create a deep copy of the environment
    def clone(self):
        env_copy = GraphEnvironment(self.max_nodes)
        env_copy.num_nodes = self.num_nodes
        env_copy.adjacency_matrix = self.adjacency_matrix.copy()
        env_copy.start_node = self.start_node
        env_copy.current_node = self.current_node
        env_copy.visited = self.visited.copy()
        env_copy.stack = self.stack.copy()
        env_copy.steps = self.steps
        env_copy.done = self.done
        return env_copy

    # Get the current observation
    def get_observation(self):
        # One-hot encoding of current node
        current_node_encoding = np.zeros(self.max_nodes)
        current_node_encoding[self.current_node] = 1

        # One-hot encoding of visited nodes
        visited_encoding = np.zeros(self.max_nodes)
        for node in self.visited:
            visited_encoding[node] = 1

        # Stack representation
        stack_encoding = np.zeros(self.max_nodes)
        for node in self.stack:
            stack_encoding[node] = 1

        # Adjacency matrix padded to max_nodes
        padded_adjacency = np.zeros((self.max_nodes, self.max_nodes))
        padded_adjacency[:self.num_nodes, :self.num_nodes] = self.adjacency_matrix

        return {
            'current_node': current_node_encoding,
            'visited': visited_encoding,
            'stack': stack_encoding,
            'adjacency': padded_adjacency.flatten(),
            'num_nodes': self.num_nodes
        }

    # Get the legal actions from the current state
    def get_legal_actions(self):
        if not self.stack:
            return []  # No actions if stack is empty

        current = self.stack[-1]
        legal_actions = []

        # Option 1: Pop from stack (always legal if stack not empty)
        legal_actions.append(self.max_nodes)  # Pop action

        # Option 2: Visit unvisited neighbors
        for neighbor in range(self.num_nodes):
            if (self.adjacency_matrix[current, neighbor] == 1 and
                neighbor not in self.visited):
                legal_actions.append(neighbor)

        return legal_actions

    # Take action in the environment
    def step(self, action):
        self.steps += 1
        reward = -0.05  # Slightly higher negative reward to encourage efficiency
        
        if action == self.max_nodes:  # Pop action
            if self.stack:
                self.stack.pop()
                if not self.stack:  # If stack empty, we're done
                    self.done = True
                else:
                    self.current_node = self.stack[-1]
        else:  # Visit a new node
            if action not in self.visited:
                self.visited.add(action)
                self.stack.append(action)
                self.current_node = action
                # Increased reward for visiting new nodes
                reward += 2.0  
                
                # Additional reward as we get closer to visiting all nodes
                completion_percentage = len(self.visited) / self.num_nodes
                reward += completion_percentage * 0.5

        # Check if DFS is complete (all nodes visited)
        if len(self.visited) == self.num_nodes:
            # Dramatically increased reward for full completion
            reward += 15.0 
            # We only call it done if we've visited all nodes
            if not self.stack:
                self.done = True
                
        # Penalty for exceeding max steps, increased to discourage wandering
        if self.steps > 3 * self.num_nodes:  # Reduced max steps to prevent infinite loops
            reward -= 10.0
            self.done = True

        return self.get_observation(), reward, self.done
    
print("GraphEnvironment initialized")
    
class DFSNetwork:
    def __init__(self, max_nodes=10):
        self.max_nodes = max_nodes
        self.action_space_size = max_nodes + 1  # Nodes + pop action
        self.input_dim = 3 * max_nodes + max_nodes * max_nodes

        # Create the model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def _build_model(self):
        """Build the neural network model with GPU support."""
        # Ensure TensorFlow uses GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("Using GPU:", physical_devices[0])
            except RuntimeError as e:
                print("Error setting GPU memory growth:", e)
        else:
            print("No GPU found, using CPU.")

        input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        
        # Smaller network for faster training
        x1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        x2 = tf.keras.layers.Dense(128, activation='relu')(x1)

        # Policy head (action probabilities)
        policy_output = tf.keras.layers.Dense(self.action_space_size, activation='softmax', name='policy')(x2)

        # Value head (state value estimation)
        value_output = tf.keras.layers.Dense(1, name='value')(x2)

        model = tf.keras.Model(inputs=input_layer, outputs=[policy_output, value_output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mse'
            }
        )

        return model

    def update_target_network(self, tau=0.01):
        """Update target network using soft update with a loss function.
        
        Args:
            tau: Float between 0 and 1 controlling the update rate.
                The closer to 0, the slower the update.
        """
        # Get current weights for both networks
        current_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        # Compute new target weights using loss-based update
        updated_weights = []
        for cw, tw in zip(current_weights, target_weights):
            # Apply soft update formula: target = (1-tau)*target + tau*current
            updated_weight = (1 - tau) * tw + tau * cw
            updated_weights.append(updated_weight)
        
        # Set the updated weights
        self.target_model.set_weights(updated_weights)
        
        # Calculate and return the update loss (MSE between the networks)
        loss = 0
        for cw, tw in zip(current_weights, updated_weights):
            loss += np.mean((cw - tw) ** 2)
        
        return loss

    def predict(self, state):
        """Make a prediction based on the state."""
        # Prepare the input
        current_node = state['current_node']
        visited = state['visited']
        stack = state['stack']
        adjacency = state['adjacency']

        # Concatenate all features
        input_features = np.concatenate([current_node, visited, stack, adjacency])
        input_features = np.expand_dims(input_features, axis=0)

        # Get prediction
        policy, value = self.model.predict(input_features, verbose=0)

        return policy[0], value[0][0]

    def target_predict(self, state):
        """Make a prediction with the target network."""
        # Prepare the input
        current_node = state['current_node']
        visited = state['visited']
        stack = state['stack']
        adjacency = state['adjacency']

        # Concatenate all features
        input_features = np.concatenate([current_node, visited, stack, adjacency])
        input_features = np.expand_dims(input_features, axis=0)

        # Get prediction
        policy, value = self.target_model.predict(input_features, verbose=0)

        return policy[0], value[0][0]

    def train(self, states, actions, rewards, next_states, dones):
        """Train the network with a batch of experiences."""
        # Prepare inputs
        inputs = []
        for state in states:
            input_features = np.concatenate([
                state['current_node'],
                state['visited'],
                state['stack'],
                state['adjacency']
            ])
            inputs.append(input_features)
        inputs = np.array(inputs)

        # Get target values using target network
        next_values = []
        for next_state in next_states:
            if next_state is None:  # Terminal state
                next_values.append(0)
            else:
                _, value = self.target_predict(next_state)
                next_values.append(value)

        # Calculate target values
        target_values = []
        for i in range(len(rewards)):
            if dones[i]:
                target_values.append(rewards[i])
            else:
                target_values.append(rewards[i] + 0.99 * next_values[i])

        # One-hot encode actions
        action_indices = np.array(actions)
        action_targets = np.zeros((len(actions), self.action_space_size))
        for i, action in enumerate(action_indices):
            action_targets[i, action] = 1

        # Train the model
        self.model.fit(
            inputs,
            [action_targets, np.array(target_values).reshape(-1, 1)],
            verbose=0,
            batch_size=32
        )

    def save(self, filepath):
        """Save the model to disk."""
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        self.model.save_weights(filepath)

    def load(self, filepath):
        """Load the model from disk."""
        if not filepath.endswith('.weights.h5'):
            filepath = filepath + '.weights.h5'
        self.model.load_weights(filepath)
        self.update_target_network()
      
print("DFSNetwork initialized")

class Node:
    """Node in the MCTS tree."""

    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.reward = 0
        self.done = False

    def expanded(self):
        return bool(self.children)

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    """Monte Carlo Tree Search implementation for DFS."""

    def __init__(self, network, num_simulations=20):  # Reduced simulations for faster execution
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = 2.0

    def run(self, env, temperature=1.0):
        """Run MCTS from the current environment state."""
        # Clone the environment
        root_env = env.clone()
        
        # Create root node
        root = Node(0)
        root.state = root_env.get_observation()

        # Get network prediction for root
        policy, value = self.network.predict(root.state)

        # Initialize root with legal actions
        legal_actions = root_env.get_legal_actions()
        for action in legal_actions:
            root.children[action] = Node(prior=policy[action])

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root, root_env.clone())

        # Select action based on visit count
        action_visits = [(action, child.visit_count) for action, child in root.children.items()]
        if not action_visits:
            return None, root  # No legal actions

        # Temperature-based action selection
        if temperature == 0:  # Deterministic selection
            action = max(action_visits, key=lambda x: x[1])[0]
        else:  # Sample based on visit count distribution
            visits = np.array([x[1] for x in action_visits])
            # Add epsilon to avoid division by zero
            visits = np.power(visits + 1e-8, 1.0 / temperature)
            visits = visits / np.sum(visits)
            action = np.random.choice([x[0] for x in action_visits], p=visits)

        return action, root

    def _simulate(self, node, env):
        """Run a single MCTS simulation."""
        # If node not expanded, expand it
        if not node.expanded():
            # Get network prediction
            policy, value = self.network.predict(node.state)

            # Initialize children with legal actions
            legal_actions = env.get_legal_actions()
            for action in legal_actions:
                node.children[action] = Node(prior=policy[action])

            return value

        # Select child with highest UCB score
        action, child = self._select_child(node)

        # Take action in the simulation
        next_state, reward, done = env.step(action)

        # Store the state in the child node
        child.state = next_state
        child.reward = reward
        child.done = done

        # Get value from child
        if done:
            value = 0
        else:
            value = self._simulate(child, env)

        # Update statistics
        child.value_sum += value + reward
        child.visit_count += 1

        return value + reward

    def _select_child(self, node):
        """Select child with highest UCB score."""
        sqrt_sum = math.sqrt(sum(child.visit_count for child in node.children.values()) + 1e-8)

        def ucb_score(action, child):
            # Exploitation term
            if child.visit_count > 0:
                q_value = child.value_sum / child.visit_count
            else:
                q_value = 0

            # Exploration term, increased weight
            u_value = self.c_puct * child.prior * sqrt_sum / (1 + child.visit_count)

            return q_value + u_value

        return max(node.children.items(), key=lambda x: ucb_score(*x))
    
print("MCTS initialized")

# Replay Memory for experience replay
class ReplayMemory:
    """Replay memory for storing experiences."""

    def __init__(self, capacity=10000):  # Reduced memory size for faster processing
        self.memory = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add an experience to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
    
# Training loop with checkpointing and improved exploration
def train_dfs_agent(episodes=1000, max_nodes=10, checkpoint_dir='checkpoint'):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize environment and agent
    env = GraphEnvironment(max_nodes)
    network = DFSNetwork(max_nodes)
    mcts = MCTS(network, num_simulations=20)  # Reduced simulations for faster execution
    memory = ReplayMemory()

    # Training statistics
    rewards_history = []
    nodes_visited_history = []
    completion_rate_history = []
    
    # Added timeouts to prevent infinite loops
    max_steps_per_episode = 100  # New maximum steps per episode

    # Check if there's a checkpoint to resume from
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    start_episode = 1
    if latest_checkpoint:
        print(f"Restoring from checkpoint: {latest_checkpoint}")
        network.load(latest_checkpoint)

        # Extract episode number from checkpoint name
        import re
        match = re.search(r'episode-(\d+)', latest_checkpoint)
        if match:
            start_episode = int(match.group(1)) + 1
            print(f"Resuming training from episode {start_episode}")

    # Epsilon for exploration (gradually decreases)
    epsilon_start = 1
    epsilon_end = 0.05
    epsilon_decay = episodes * 0.5  # Faster decay

    # Training frequency control
    train_frequency = 4  # Only train every N steps
    update_target_frequency = 10  # Update target network every N episodes

    for episode in range(start_episode, episodes+1):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Calculate epsilon for this episode (linear decay)
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * 
                      min(1.0, episode / epsilon_decay))
        
        # Temperature for MCTS exploration
        temperature = max(0.2, 1.0 - (episode / (episodes*0.6)))  # Faster decay
        
        while not done and steps < max_steps_per_episode:
            steps += 1
            
            # Sometimes take random action for exploration
            if random.random() < epsilon:
                legal_actions = env.get_legal_actions()
                if not legal_actions:
                    break
                action = random.choice(legal_actions)
                # Create a dummy root for consistency
                root = Node(0)
            else:
                # Run MCTS to get action
                action, root = mcts.run(env, temperature=temperature)
                
                if action is None:
                    break

            # Take action in environment
            next_state, reward, done = env.step(action)
            episode_reward += reward

            # Store experience
            memory.add(state, action, reward, next_state if not done else None, done)

            # Move to next state
            state = next_state

            # Train network less frequently for better efficiency
            if episode % train_frequency == 0 and len(memory) >= 128:
                states, actions, rewards, next_states, dones = memory.sample(128)
                network.train(states, actions, rewards, next_states, dones)

        # Update target network periodically
        if episode % update_target_frequency == 0:
            network.update_target_network()

        # Track statistics
        rewards_history.append(episode_reward)
        nodes_visited_history.append(len(env.visited))
        completion_rate = len(env.visited) / env.num_nodes
        completion_rate_history.append(completion_rate)
        
        # Check if we hit the step limit
        if steps >= max_steps_per_episode:
            print(f"Episode {episode} reached step limit of {max_steps_per_episode}, terminating early")

        # Save checkpoint periodically
        if episode % 100 == 0 or episode == episodes:
            save_path = os.path.join(checkpoint_dir, f"dfs_model_episode-{episode}")
            network.save(save_path)
            print(f"Checkpoint saved at episode {episode} to {save_path}.weights.h5")

        # Print progress with more detailed statistics
        if episode % 10 == 0:  # More frequent feedback
            recent_rewards = rewards_history[-min(10, len(rewards_history)):]
            recent_nodes = nodes_visited_history[-min(10, len(nodes_visited_history)):] 
            recent_completion = completion_rate_history[-min(10, len(completion_rate_history)):]
            
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_nodes = sum(recent_nodes) / len(recent_nodes)
            avg_completion = sum(recent_completion) / len(recent_completion) * 100
            
            print(f"Episode {episode}/{episodes} - Avg Reward: {avg_reward:.2f}, Avg Nodes: {avg_nodes:.2f}, Completion: {avg_completion:.1f}%")
            print(f"  Epsilon: {epsilon:.3f}, Temperature: {temperature:.3f}, Steps: {steps}")

    return network, rewards_history, nodes_visited_history

print("Training function defined")

# Function to evaluate the model
def evaluate_model(network, num_tests=10, max_nodes=10, verbose=True):
    env = GraphEnvironment(max_nodes)
    completion_rates = []
    rewards = []
    
    for test in range(num_tests):
        state = env.reset()
        done = False
        total_reward = 0
        
        if verbose and test == 0:
            print(f"\nDetailed evaluation on test graph:")
            print(f"Number of nodes: {env.num_nodes}")
            print(f"Starting node: {env.start_node}")
            visited_order = [env.start_node]
        
        step_count = 0
        while not done and step_count < 2 * env.num_nodes:  # Reduced step limit for evaluation
            step_count += 1
            
            # Use the network directly for evaluation
            policy, _ = network.predict(state)
            
            # Get legal actions
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
                
            # Filter policy to legal actions and add noise for exploration
            legal_policy = np.array([policy[action] for action in legal_actions])
            legal_policy = legal_policy / legal_policy.sum()  # Normalize
            
            # Select action (slightly stochastic for evaluation)
            action_idx = np.random.choice(len(legal_actions), p=legal_policy)
            action = legal_actions[action_idx]
            
            # Take action
            state, reward, done = env.step(action)
            total_reward += reward
            
            if verbose and test == 0 and action != env.max_nodes:
                visited_order.append(action)
        
        completion_rate = len(env.visited) / env.num_nodes
        completion_rates.append(completion_rate)
        rewards.append(total_reward)
        
        if verbose and test == 0:
            print(f"DFS traversal order: {visited_order}")
            print(f"Total nodes visited: {len(env.visited)}/{env.num_nodes} ({completion_rate*100:.1f}%)")
            print(f"Total reward: {total_reward:.2f}")
    
    avg_completion = sum(completion_rates) / len(completion_rates) * 100
    avg_reward = sum(rewards) / len(rewards)
    
    print(f"\nEvaluation over {num_tests} tests:")
    print(f"Average completion rate: {avg_completion:.1f}%")
    print(f"Average reward: {avg_reward:.2f}")
    
    return avg_completion, avg_reward

print("Starting training...")
# Reduced episode count for faster training
trained_network, rewards, nodes = train_dfs_agent(episodes=1000, max_nodes=10)

# Evaluate the trained model
evaluate_model(trained_network, num_tests=20)