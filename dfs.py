import numpy as np
import tensorflow as tf
import collections
import random
import math
import os
from typing import Dict, List, NamedTuple, Optional, Tuple, Set

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
        self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if random.random() < 0.3:  # 30% chance of edge
                    self.adjacency_matrix[i, j] = 1
                    self.adjacency_matrix[j, i] = 1  # Undirected graph

        # Set start node
        self.start_node = random.randint(0, self.num_nodes - 1)

        # DFS state
        self.current_node = self.start_node
        self.visited = set([self.start_node])
        self.stack = [self.start_node]
        self.steps = 0
        self.done = False

        return self.get_observation()

    # Reset the environment to a new random graph
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
        reward = -0.01  # Small negative reward for each step

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
                reward += 1.0  # Reward for visiting new node

        # Check if DFS is complete (all nodes visited)
        if len(self.visited) == self.num_nodes:
            reward += 5.0  # Bonus for visiting all nodes
            self.done = True

        # Penalty for exceeding max steps
        if self.steps > 3 * self.num_nodes:
            reward -= 5.0
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
        x1 = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        x2 = tf.keras.layers.Dense(128, activation='relu')(x1)

        # Policy head (action probabilities)
        policy_head = tf.keras.layers.Dense(128, activation='relu')(x2)
        policy_output = tf.keras.layers.Dense(self.action_space_size, activation='softmax', name='policy')(policy_head)

        # Value head (state value estimation)
        value_head = tf.keras.layers.Dense(128, activation='relu')(x2)
        value_output = tf.keras.layers.Dense(1, name='value')(value_head)

        model = tf.keras.Model(inputs=input_layer, outputs=[policy_output, value_output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
            'policy': 'categorical_crossentropy',
            'value': 'mse'
            }
        )

        return model

    def update_target_network(self):
        """Update target network with current weights."""
        self.target_model.set_weights(self.model.get_weights())

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

    # In the save method of DFSNetwork class
    def save(self, filepath):
      """Save the model to disk."""
      if not filepath.endswith('.weights.h5'):
         filepath = filepath + '.weights.h5'
      self.model.save_weights(filepath)

    # And in the load method as well
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

    def __init__(self, network, num_simulations=50, c_puct=1.0):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def run(self, env, temperature=1.0):
        """Run MCTS from the current environment state."""
        # Create root node
        root = Node(0)
        root.state = env.get_observation()

        # Get network prediction for root
        policy, value = self.network.predict(root.state)

        # Initialize root with legal actions
        legal_actions = env.get_legal_actions()
        for action in legal_actions:
            root.children[action] = Node(prior=policy[action])

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root, env.max_nodes, env)

        # Select action based on visit count
        action_visits = [(action, child.visit_count) for action, child in root.children.items()]
        if not action_visits:
            return None, root  # No legal actions

        if temperature == 0:  # Deterministic selection
            action = max(action_visits, key=lambda x: x[1])[0]
        else:  # Sample based on visit count distribution
            visits = np.array([x[1] for x in action_visits])
            visits = visits ** (1 / temperature)
            visits = visits / np.sum(visits)
            action = np.random.choice([x[0] for x in action_visits], p=visits)

        return action, root

    def _simulate(self, node, max_nodes, real_env):
        """Run a single MCTS simulation."""
        # If node not expanded, expand it
        if not node.expanded():
            # Create a copy of the environment to simulate
            env = GraphEnvironment(max_nodes)
            env.num_nodes = real_env.num_nodes
            env.adjacency_matrix = real_env.adjacency_matrix.copy()
            env.start_node = real_env.start_node
            env.current_node = real_env.current_node
            env.visited = real_env.visited.copy()
            env.stack = real_env.stack.copy()
            env.steps = real_env.steps
            env.done = real_env.done

            # Get network prediction
            policy, value = self.network.predict(node.state)

            # Initialize children with legal actions
            legal_actions = real_env.get_legal_actions()
            for action in legal_actions:
                node.children[action] = Node(prior=policy[action])

            return value

        # Select child with highest UCB score
        action, child = self._select_child(node)

        # Create a copy of the environment to simulate
        env = GraphEnvironment(max_nodes)
        env.num_nodes = real_env.num_nodes
        env.adjacency_matrix = real_env.adjacency_matrix.copy()
        env.start_node = real_env.start_node
        env.current_node = real_env.current_node
        env.visited = real_env.visited.copy()
        env.stack = real_env.stack.copy()
        env.steps = real_env.steps
        env.done = real_env.done

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
            value = self._simulate(child, max_nodes, env)

        # Update statistics
        child.value_sum += value + reward
        child.visit_count += 1

        return value + reward

    def _select_child(self, node):
        """Select child with highest UCB score."""
        sqrt_sum = math.sqrt(sum(child.visit_count for child in node.children.values()))

        def ucb_score(action, child):
            # Exploitation term
            if child.visit_count > 0:
                q_value = child.value_sum / child.visit_count
            else:
                q_value = 0

            # Exploration term
            u_value = self.c_puct * child.prior * sqrt_sum / (1 + child.visit_count)

            return q_value + u_value

        return max(node.children.items(), key=lambda x: ucb_score(*x))
    
print("MCTS initialized")

# Replay Memory for experience replay
class ReplayMemory:
    """Replay memory for storing experiences."""

    def __init__(self, capacity=10000):
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
    
# Training loop with checkpointing
def train_dfs_agent(episodes=1000, max_nodes=10, checkpoint_dir='checkpoint'):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize environment and agent
    env = GraphEnvironment(max_nodes)
    network = DFSNetwork(max_nodes)
    mcts = MCTS(network, num_simulations=20)
    memory = ReplayMemory()

    # Training statistics
    rewards_history = []
    nodes_visited_history = []

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

    for episode in range(start_episode, episodes+1):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Run MCTS to get action
            action, root = mcts.run(env, temperature=1.0 if episode < episodes//2 else 0.5)

            if action is None:
                break

            # Take action in environment
            next_state, reward, done = env.step(action)
            episode_reward += reward

            # Store experience
            memory.add(state, action, reward, next_state if not done else None, done)

            # Move to next state
            state = next_state

            # Train network
            if len(memory) >= 128:
                states, actions, rewards, next_states, dones = memory.sample(128)
                network.train(states, actions, rewards, next_states, dones)

        # Update target network periodically
        if episode % 10 == 0:
            network.update_target_network()

        # Track statistics
        rewards_history.append(episode_reward)
        nodes_visited_history.append(len(env.visited))

        # Save checkpoint periodically
        if episode % 50 == 0 or episode == episodes:
            save_path = os.path.join(checkpoint_dir, f"dfs_model_episode-{episode}")
            network.save(save_path)
            print(f"Checkpoint saved at episode {episode} to {save_path}.weights.h5")

        # Print progress
        if episode % 100 == 0:
            avg_reward = sum(rewards_history[-100:]) / min(100, len(rewards_history))
            avg_nodes = sum(nodes_visited_history[-100:]) / min(100, len(nodes_visited_history))
            print(f"Episode {episode}/{episodes} - Avg Reward: {avg_reward:.2f}, Avg Nodes Visited: {avg_nodes:.2f}/{env.num_nodes}")

    return network, rewards_history, nodes_visited_history

print("Training function defined")

print("Starting training...")
trained_network, rewards, nodes = train_dfs_agent(episodes=1000, max_nodes=10)

# Test the trained agent
env = GraphEnvironment(max_nodes=10)
state = env.reset()
done = False
total_reward = 0

print("\nTesting trained agent on a new graph:")
print(f"Number of nodes: {env.num_nodes}")
print(f"Starting node: {env.start_node}")

visited_order = [env.start_node]


while not done:
    # Use the network directly without MCTS for inference
    policy, _ = trained_network.predict(state)

    # Get legal actions
    legal_actions = env.get_legal_actions()
    if not legal_actions:
        break

    # Filter policy to legal actions
    legal_policy = {action: policy[action] for action in legal_actions}
    action = max(legal_policy.items(), key=lambda x: x[1])[0]

    # Take action
    state, reward, done = env.step(action)
    total_reward += reward

    if action != env.max_nodes:  # If not a pop action
        visited_order.append(action)

print(f"DFS traversal order: {visited_order}")
print(f"Total nodes visited: {len(env.visited)}/{env.num_nodes}")
print(f"Total reward: {total_reward}")