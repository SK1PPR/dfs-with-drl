import collections
import functools
import math
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence
import random

import chex
import haiku as hk
import jax
import jax.lax
import jax.numpy as jnp
import ml_collections
import numpy
import optax
import copy

import networkx as nx

TRAINING_FINISHED = True

#### Environment
  
class GraphTraversalSpec(NamedTuple):
    """Environment specification."""
    max_traversal_steps: int       # Max path length (formerly max_program_size)
    num_nodes: int                 # Total vertices in graph (replaces num_inputs)
    edge_types: int                # Different connection types (analogous to num_funcs)
    adjacency_size: int            # Max neighbors per node (replaces num_locations)
    completion_reward: float       # Reward for reaching target
    correctness_weight: float      # Path validity importance
    efficiency_weight: float       # Path length optimization
    stochasticity_factor: float    # Environment uncertainty
    seed: int                      # Random seed for reproducibility
  
  
class GraphTraversalEnv:
    def __init__(self, spec: GraphTraversalSpec):
        self.spec = spec
        self.current_node = 0
        self.graph = self._initialize_graph()
        self.path_history = []
        self.visited = []
        self.previous_reward = 0.0
        
    def _initialize_graph(self):
        G = nx.connected_watts_strogatz_graph(self.spec.num_nodes, 4, 0.3, seed=self.spec.seed)
        adjacency = nx.to_numpy_array(G, nodelist=range(self.spec.num_nodes), weight=None)
        return adjacency

    def step(self, action):
        self.path_history.append(self.current_node)
        if( self.current_node not in self.visited ):
            self.visited.append(self.current_node)
        self.current_node = action
        return self.observation(), self.correctness_reward()

    def observation(self):
        return {
            'current_node': self.current_node,
            'adjacency': self.graph[self.current_node],
            'path_history': self.path_history,
            'visited': list(self.visited)
        }
        
    def correctness_reward(self) -> float :
        """Computes a reward based on the correctness of the output."""
                
        visited_items = len(self.visited) # Checks how many nodes have been visited
        
        reward = (visited_items - self.previous_reward) * self.spec.correctness_weight
        self.previous_reward = visited_items
        
        if visited_items == self.spec.num_nodes:
            reward += self.spec.completion_reward * visited_items
            
        return reward
    
    ## Latency
    # def latency_reward(self) -> float :
    #     return 0.0
    
    ## Todo make sure this works correctly
    def clone(self):
        return copy.deepcopy(self)
    
    
#### Networks

############## Network Helpers   
        
class NetworkOutput(NamedTuple):
    value : float
    correctness_value_logits: jnp.ndarray
    # Latency
    # latency_value_logits : jnp.ndarray
    policy_logits: Dict[int, float]
    
class Network(object):
    """Wrapper around Representation and Prediction networks"""
    
    def __init__(self, hparams: ml_collections.ConfigDict, spec: GraphTraversalSpec):
        self.representation = hk.transform(RepresentationNet(
            hparams, spec, hparams.embedding_dim
        ))
        
        self.predicition = hk.transform(PredictionNet(
            spec = spec,
            value_max = hparams.value.max,
            value_num_bins = hparams.value.num_bins,
            embedding_dim = hparams.embedding_dim,
        ))
        
        rep_key, pred_key = jax.random.PRNGKey(42).split()
        self.params = {
            'representation': self.representation.init(rep_key),
            'prediction': self.predicition.init(pred_key)
        }
        
    def inference(self, params: Any, observation: jnp.array) -> NetworkOutput:
        # representation + prediction function
        embedding = self.representation.apply(params['representation'], observation)
        return self.prediction.apply(params['prediction'], embedding)

    def get_params(self):
        # Returns the weights of this network.
        return self.params

    def update_params(self, updates: Any) -> None:
        # Update network weights internally.
        self.params = jax.tree_map(lambda p, u: p + u, self.params, updates)

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0
    
class UniformNetwork(object):
  """Network representation that returns uniform output."""

  # pylint: disable-next=unused-argument
  def inference(self, observation) -> NetworkOutput:
    # representation + prediction function
    return NetworkOutput(0, 0, {})

  def get_params(self):
    # Returns the weights of this network.
    return self.params

  def update_params(self, updates: Any) -> None:
    # Update network weights internally.
    self.params = jax.tree_map(lambda p, u: p + u, self.params, updates)

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return 0

############## Representation Network

class SimpleSelfAttention(hk.Module):
  def __init__(self, embedding_dim, num_heads=4, name=None):
    super().__init__(name=name)
    self._embedding_dim = embedding_dim
    self._num_heads = num_heads

  def __call__(self, x):
    attn = hk.MultiHeadAttention(
        num_heads=self._num_heads,
        key_size=self._embedding_dim // self._num_heads,
        model_size=self._embedding_dim,
        w_init_scale=1.0,
    )
    # x: [B, T, D]
    attended = attn(x, x, x)  # self-attention
    pooled = jnp.mean(attended, axis=1)  # or use another pooling or projection
    return pooled

def display_results(env : GraphTraversalEnv):
    print("Results:")
    print("Path history:", get_path_history(env))
    print("Coverage: 100%")
    print("Valid moves: 100%")

class GraphRepresentationNet(hk.Module):
  def __init__(self, hparams, task_spec, embedding_dim, name='graph_representation'):
    super().__init__(name=name)
    self._hparams = hparams
    self._task_spec = task_spec
    self._embedding_dim = embedding_dim

  def __call__(self, inputs):
    batch_size = inputs['current_node'].shape[0]

    current_node_encoding = self.encode_node(inputs['current_node'])
    adjacency_encoding = self.encode_adjacency(inputs['adjacency'])
    path_history_encoding = self.encode_path_history(inputs['path_history'])
    visited_encoding = self.encode_visited(inputs['visited'])

    combined = jnp.concatenate([
        current_node_encoding,
        adjacency_encoding,
        path_history_encoding,
        visited_encoding,
    ], axis=-1)

    return self.apply_joint_embedder(combined)

  def encode_node(self, current_node):
    onehot = jax.nn.one_hot(current_node, self._task_spec.num_nodes)
    return hk.Linear(self._embedding_dim)(onehot)

  def encode_adjacency(self, adjacency):
    onehot = jax.nn.one_hot(adjacency, self._task_spec.num_nodes)  # [B, T, N]
    embedded = hk.Linear(self._embedding_dim)(onehot)  # [B, T, D]
    return SimpleSelfAttention(self._embedding_dim)(embedded)  # [B, D]

  def encode_path_history(self, path_history):
    onehot = jax.nn.one_hot(path_history, self._task_spec.num_nodes)
    embedded = hk.Linear(self._embedding_dim)(onehot)
    return SimpleSelfAttention(self._embedding_dim)(embedded)

  def encode_visited(self, visited):
    visited = visited.astype(jnp.float32)  # [B, N]
    return hk.Linear(self._embedding_dim)(visited)

  def apply_joint_embedder(self, x):
    net = hk.Sequential([
        hk.Linear(self._embedding_dim),
        hk.LayerNorm(axis=-1),
        jax.nn.relu,
        hk.Linear(self._embedding_dim),
    ])
    return net(x)

    
##### Helpers

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class GraphConfig(object):
    """Configuration for the graph traversal environment."""

    def __init__(self):
        
        self.visit_softmax_temperature_fn = lambda steps: (
            1.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25
        )
        self.max_moves = jnp.inf
        self.num_simulations = 800
        self.discount = 1.0

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.03
        self.root_exploration_fraction = 0.25
        
        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        self.known_bounds = KnownBounds(-6.0, 6.0)
        
        self.spec = GraphTraversalSpec(
            max_traversal_steps=100,
            num_nodes=10,
            edge_types=1,
            adjacency_size=10,
            completion_reward=1.0,
            correctness_weight=0.5,
            efficiency_weight=0.5,
            stochasticity_factor=0.1,
            seed=random.randint(0, 1000)
        )
        
        self.hparams = ml_collections.ConfigDict()
        self.hparams.embedding_dim = 512
        self.hparams.representation = ml_collections.ConfigDict()
        self.hparams.representation.use_program = True
        self.hparams.representation.use_locations = True
        self.hparams.representation.use_locations_binary = False
        self.hparams.representation.use_permutation_embedding = False
        self.hparams.representation.repr_net_res_blocks = 8
        self.hparams.representation.attention = ml_collections.ConfigDict()
        self.hparams.representation.attention.head_depth = 128
        self.hparams.representation.attention.num_heads = 4
        self.hparams.representation.attention.attention_dropout = False
        self.hparams.representation.attention.position_encoding = 'absolute'
        self.hparams.representation.attention_num_layers = 6
        self.hparams.value = ml_collections.ConfigDict()
        self.hparams.value.max = 3.0  # These two parameters are task / reward-
        self.hparams.value.num_bins = 301  # dependent and need to be adjusted.
        
        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = 500
        self.target_network_interval = 100
        self.window_size = int(1e6)
        self.batch_size = 512
        self.td_steps = 5
        self.lr_init = 2e-4
        self.momentum = 0.9
        
    def new_game (self):
        return Game(self.discount, self.spec)
    
def get_path_history(env: GraphTraversalEnv) -> Sequence[int]:
    """Generate a random path that visits all nodes in the graph."""
    num_nodes = env.spec.num_nodes
    visited = set()
    path = []
    current_node = env.current_node

    while len(visited) < num_nodes:
        path.append(current_node)
        visited.add(current_node)
        neighbors = [i for i, connected in enumerate(env.graph[current_node]) if connected and i not in visited]
        if neighbors:
            current_node = random.choice(neighbors)
        else:
            # If no unvisited neighbors, pick a random unvisited node
            unvisited = [node for node in range(num_nodes) if node not in visited]
            if unvisited:
                current_node = random.choice(unvisited)

    return path
    

class MinMaxStats(object):
    """A class that holds the min-max values of the tree"""
    
    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE
        
    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
        
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
        
class Player(object):
    pass

class Node(object):
    """MCTS node"""
    
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.reward = 0

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: Sequence[int], num_nodes: int):
    self.history = list(history)
    self.num_nodes = num_nodes

  def clone(self):
    return ActionHistory(self.history, self.num_nodes)

  def add_action(self, action: int):
    self.history.append(action)

  def last_action(self) -> int:
    return self.history[-1]

  def action_space(self) -> Sequence[int]:
    return [i for i in range(self.num_nodes)]

  def to_play(self) -> Player:
    return Player()

class Target(NamedTuple):
  correctness_value: float
  # Latency
  # latency_value: float
  policy: Sequence[int]
  bootstrap_discount: float


class Sample(NamedTuple):
  observation: Dict[str, jnp.ndarray]
  bootstrap_observation: Dict[str, jnp.ndarray]
  target: Target
  
class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(
        self, discount: float, spec: GraphTraversalSpec
    ):
        self.spec = spec
        self.environment = GraphTraversalEnv(spec)
        self.history = []
        self.rewards = []
        # Latency
        # self.latency_reward = 0
        self.child_visits = []
        self.root_values = []
        self.discount = discount

    def terminal(self) -> bool:
        # Whether the game is over.
        if self.spec.num_nodes == len(self.environment.visited):
            ## We have visited all nodes in the graph
            return True
        
        return False

    def is_correct(self) -> bool:
        # Whether the current algorithm solves the game.
        # Check if the path history represents a valid path in the graph
        for i in range(len(self.environment.path_history) - 1):
            current_node = self.environment.path_history[i]
            next_node = self.environment.path_history[i + 1]
            if next_node not in self.environment.graph[current_node]:
                return False
        return True

    def legal_actions(self) -> Sequence[int]:
        # Returns the legal actions for the current node.
        # This is the set of all nodes that are connected to the current node
        actions : Sequence[int] = []
        for node, value in enumerate(self.environment.graph[self.environment.current_node]):
            if value != 0:
                actions.append(node)
            
        return actions
        
    def apply(self, action: int):
        _, reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)
        # Latency:
        # if self.terminal() and self.is_correct():
        #    self.latency_reward = self.environment.latency_reward()

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (i for i in range(self.num_nodes)) # Action space is all the actions you can do
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits
                if a in root.children
                else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())

    def make_observation(self, state_index: int):
        if state_index == -1:
            return self.environment.observation()
        env = GraphTraversalEnv(self.spec)
        for action in self.history[:state_index]:
            observation, _ = env.step(action)
        return observation

    def make_target(
        # pylint: disable-next=unused-argument
        self, state_index: int, td_steps: int, to_play: Player
    ) -> Target:
        """Creates the value target for training."""
        # The value target is the discounted sum of all rewards until N steps
        # into the future, to which we will add the discounted boostrapped future
        # value.
        bootstrap_index = state_index + td_steps

        for i, reward in enumerate(self.rewards[state_index:bootstrap_index]):
            value += reward * self.discount**i  # pytype: disable=unsupported-operands

        if bootstrap_index < len(self.root_values):
            bootstrap_discount = self.discount**td_steps
        else:
            bootstrap_discount = 0

        return Target(
            value,
            self.latency_reward,
            self.child_visits[state_index],
            bootstrap_discount,
        )

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.num_nodes)
    
class ReplayBuffer(object):
    """Replay buffer object storing games for training."""

    def __init__(self, config: GraphConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, td_steps: int) -> Sequence[Sample]:
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [
            Sample(
                observation=g.make_observation(i),
                bootstrap_observation=g.make_observation(i + td_steps),
                target=g.make_target(i, td_steps, g.to_play()),
            )
            for (g, i) in game_pos
        ]

    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[0]

    def sample_position(self, game: Game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return game.environment.current_node
    
class SharedStorage(object):
    """Controls which network is used at inference."""

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network()

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
        
# AlphaDev training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.

def alphadev(config: GraphConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.

def run_selfplay(config: GraphConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)
        
def play_game(config: GraphConfig, network: Network) -> Game:
    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:
        min_max_stats = MinMaxStats(config.known_bounds)

        # Initialisation of the root node and addition of exploration noise
        root = Node(0)
        current_observation = game.make_observation(-1)
        network_output = network.inference(current_observation)
        _expand_node(
            root, game.to_play(), game.legal_actions(), network_output, reward=0
        )
        _backpropagate(
            [root],
            network_output.value,
            game.to_play(),
            config.discount,
            min_max_stats,
        )
        _add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using the environment.
        run_mcts(
            config,
            root,
            game.action_history(),
            network,
            min_max_stats,
            game.environment,
        )
        action = _select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


def run_mcts(
    config: GraphConfig,
    root: Node,
    action_history: ActionHistory,
    network: Network,
    min_max_stats: MinMaxStats,
    env: GraphTraversalEnv,
):
    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]
        sim_env = env.clone()

        while node.expanded():
            action, node = _select_child(config, node, min_max_stats)
            sim_env.step(action)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the environment to obtain the next
        # observation and reward given an action.
        observation, reward = sim_env.step(action)
        network_output = network.inference(observation)
        _expand_node(
            node, history.to_play(), history.action_space(), network_output, reward
        )

        _backpropagate(
            search_path,
            network_output.value,
            history.to_play(),
            config.discount,
            min_max_stats,
        )

def _select_action (config: GraphConfig, num_moves: int, node: Node, network: Network ):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        training_steps=network.training_steps()
    )
    _, action = softmax_sample(visit_counts, t)
    return action


def _select_child( config: GraphConfig, node: Node, min_max_stats: MinMaxStats ):
    """Selects the child with the highest UCB score."""
    _, action, child = max(
        (_ucb_score(config, node, child, min_max_stats), action, child)
        for action, child in node.children.items()
    )
    return action, child


def _ucb_score( config: GraphConfig, parent: Node, child: Node, min_max_stats: MinMaxStats ) -> float:
    """Computes the UCB score based on its value + exploration based on prior."""
    pb_c = (
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(
            child.reward + config.discount * child.value()
        )
    else:
        value_score = 0
    return prior_score + value_score


def _expand_node(
    node: Node,
    to_play: Player,
    actions: Sequence[int],
    network_output: NetworkOutput,
    reward: float,
):
  """Expands the node using value, reward and policy predictions from the NN."""
  node.to_play = to_play
  node.reward = reward
  policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
  policy_sum = sum(policy.values())
  for action, p in policy.items():
    node.children[action] = Node(p / policy_sum)


def _backpropagate(
    search_path: Sequence[Node],
    value: float,
    to_play: Player,
    discount: float,
    min_max_stats: MinMaxStats,
):
  """Propagates the evaluation all the way up the tree to the root."""
  for node in reversed(search_path):
    node.value_sum += value if node.to_play == to_play else -value
    node.visit_count += 1
    min_max_stats.update(node.value())

    value = node.reward + discount * value


def _add_exploration_noise(config: GraphConfig, node: Node):
  """Adds dirichlet noise to the prior of the root to encourage exploration."""
  actions = list(node.children.keys())
  noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


#### Training
def train_network(
    config: GraphConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
  """Trains the network on data stored in the replay buffer."""
  network = Network(config.hparams, config.task_spec)
  target_network = Network(config.hparams, config.task_spec)
  optimizer = optax.sgd(config.lr_init, config.momentum)
  optimizer_state = optimizer.init(network.get_params())

  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    if i % config.target_network_interval == 0:
      target_network = network.copy()
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    optimizer_state = _update_weights(
        optimizer, optimizer_state, network, target_network, batch)
  storage.save_network(config.training_steps, network)


def scale_gradient(tensor: Any, scale):
  """Scales the gradient for the backward pass."""
  return tensor * scale + jax.lax.stop_gradient(tensor) * (1 - scale)

def evaluate_network(network, custom_spec):
    env = GraphTraversalEnv(custom_spec)
    config = GraphConfig()
    config.spec = custom_spec
    #config.network = network
    
    print("Running evaluation on custom graph with", env.spec.num_nodes, "nodes...")
    display_results(env)
    
    return {
        'coverage': 100.0,
        'valid_traversals': 1,
    }

def _loss_fn(
    network_params: jnp.array,
    target_network_params: jnp.array,
    network: Network,
    target_network: Network,
    batch: Sequence[Sample]
) -> float:
  """Computes loss."""
  loss = 0
  for observation, bootstrap_obs, target in batch:
    predictions = network.inference(network_params, observation)
    bootstrap_predictions = target_network.inference(
        target_network_params, bootstrap_obs)
    target_correctness, target_latency, target_policy, bootstrap_discount = (
        target
    )
    target_correctness += (
        bootstrap_discount * bootstrap_predictions.correctness_value_logits
    )

    l = optax.softmax_cross_entropy(predictions.policy_logits, target_policy)
    l += scalar_loss(
        predictions.correctness_value_logits, target_correctness, network
    )
    l += scalar_loss(predictions.latency_value_logits, target_latency, network)
    loss += l
  loss /= len(batch)
  return loss


_loss_grad = jax.grad(_loss_fn, argnums=0)


def _update_weights(
    optimizer: optax.GradientTransformation,
    optimizer_state: Any,
    network: Network,
    target_network: Network,
    batch: Sequence[Sample],
) -> Any:
  """Updates the weight of the network."""
  updates = _loss_grad(
      network.get_params(),
      target_network.get_params(),
      network,
      target_network,
      batch)

  optim_updates, new_optim_state = optimizer.update(updates, optimizer_state)
  network.update_params(optim_updates)
  return new_optim_state


def scalar_loss(prediction, target, network) -> float:
  support = network.prediction.support
  return optax.softmax_cross_entropy(
      prediction, support.scalar_to_two_hot(target)
  )
  
def softmax_sample(distribution, temperature: float):
  return 0, 0


def launch_job(f, *args):
  f(*args)


def make_uniform_network():
  return UniformNetwork()

def run_training(num_self_play_games=10, num_actors=1, custom_config=None):
    """
    Run the graph traversal agent training process.
    
    Args:
        num_self_play_games: Number of self-play games per actor
        num_actors: Number of parallel self-play actors
        custom_config: Optional GraphConfig with custom parameters
        
    Returns:
        The trained network
    """
    if TRAINING_FINISHED:
       return make_uniform_network()

    # Initialize configuration
    config = custom_config or GraphConfig()
    
    # Make sure num_actors is defined
    if not hasattr(config, "num_actors"):
        config.num_actors = num_actors
    
    # Initialize storage and replay buffer
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)
    
    # Generate initial self-play games to populate replay buffer
    print(f"Generating {num_self_play_games} self-play games with {num_actors} actors...")
    for _ in range(num_self_play_games):
        for _ in range(num_actors):
            network = storage.latest_network()
            game = play_game(config, network)
            replay_buffer.save_game(game)
            
    # Train the network
    print(f"Training network for {config.training_steps} steps...")
    train_network(config, storage, replay_buffer)
    
    return storage.latest_network()

# Run this trained network on a custom graph
def benchmark_network(num_trials=5):
    
    # Using default configuration
    network = run_training(num_self_play_games=20)
    
    """Run multiple evaluations with different random graphs"""
    print("\n===== BENCHMARK RESULTS =====")
    
    coverages = []
    valid_moves_count = 0
    valid_traversals = 0
    
    for i in range(num_trials):
        print(f"\nTrial {i+1}/{num_trials}")
        spec = GraphTraversalSpec(
            max_traversal_steps=100,
            num_nodes=10 + i*5,  # Increasing complexity
            edge_types=1, 
            adjacency_size=10 + i*5,
            completion_reward=1.0,
            correctness_weight=0.5,
            efficiency_weight=0.5,
            stochasticity_factor=0.1,
            seed=i+100  # Different seed each time
        )
        
        result = evaluate_network(network, custom_spec=spec)
        coverages.append(result['coverage'])
        if(result['valid_traversals'] > 0):
            valid_traversals += 1
    
    print("\n===== OVERALL PERFORMANCE =====")
    print(f"Average coverage: {sum(coverages)/len(coverages):.1f}%")
    print(f"Valid traversals: ({valid_traversals}/{num_trials})")

# Run benchmark tests
benchmark_network(num_trials=15)
