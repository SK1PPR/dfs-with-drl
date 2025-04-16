import collections
import functools
import math
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence

import chex
import haiku as hk
import jax
import jax.lax
import jax.numpy as jnp
import ml_collections
import numpy
import optax
import copy

#### Environment
  
class GraphTraversalSpec(NamedTuple):
  max_traversal_steps: int       # Max path length (formerly max_program_size)
  num_nodes: int                 # Total vertices in graph (replaces num_inputs)
  edge_types: int                # Different connection types (analogous to num_funcs)
  adjacency_size: int            # Max neighbors per node (replaces num_locations)
  action_space: int              # Possible traversal decisions (replaces num_actions)
  completion_reward: float       # Reward for reaching target
  correctness_weight: float      # Path validity importance
  efficiency_weight: float       # Path length optimization
  stochasticity_factor: float    # Environment uncertainty
  
  
class GraphTraversalEnv:
    ## TODO
    def __init__(self, spec: GraphTraversalSpec):
        self.spec = spec
        self.graph = self._initialize_graph() ## Todo : use the specs to initialize a random graph
        self.current_node = None
        self.path_history = []
        self.visited = []
        self.previous_reward = 0.0

    ## TODO
    def step(self, action):
        # Modified assembly instruction application to graph navigation
        next_node = self._apply_traversal_action(action)
        reward = self._calculate_reward()
        return self.observation(), reward

    ## TODO
    def _apply_traversal_action(self, action):
        # Convert assembly-like actions to graph operations
        if action == MOVE_FORWARD:
            return self._explore_adjacent()
        elif action == BACKTRACK:
            return self._step_back()
        elif action == SHORTCUT:
            return self._attempt_shortcut()

    def observation(self):
        return {
            'current_node': self.current_node,
            'adjacency': self.graph[self.current_node],
            'path_history': self.path_history,
            'visited': list(self.visited)
        }
        
    def correctness_reward(self) -> float :
        """Computes a reward based on the correctness of the output."""
        expected_outputs = get_expected_outputs() ## TODO : Implement this function to get the correct expected path
        
        # Weighted sum of correctly travelled path
        correct_items = 0
        for i in range(len(self.path_history)):
            if self.path_history[i] == expected_outputs[i]:
                correct_items += 1
                
        reward = (correct_items - self.previous_reward) * self.spec.correctness_weight
        self.previous_reward = correct_items
        
        # Bonus for fully correct path
        if correct_items == len(expected_outputs):
            reward += self.spec.completion_reward * correct_items
            
        return reward
    
    ## Todo
    # def latency_reward(self) -> float :
    #     return 0.0
    
    ## Todo make sure this works correctly
    def clone(self):
        return copy.deepcopy(self)
    
    
#### Networks

############## Network Helpers

class Action(object):
    """Action representation."""
    
    def __init__(self, index: int):
        self.index = index
        
    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index      
        
class NetworkOutput(NamedTuple):
    value : float
    correctness_value_logits: jnp.ndarray
    # TODO
    # latency_value_logits : jnp.ndarray
    policy_logits: Dict[Action, float]
    
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

class MultiQueryAttentionBlock:
    """
    Attention with multiple query heads and a single shared key and value head.

    Implementation of "Fast Transformer Decoding: One Write-Head is All You Need", see https://arxiv.org/abs/1911.02150.
    """
    
class ResBlockV2:
     """Layer-normed variant of the block from https://arxiv.org/abs/1603.05027."""


## TODO fix this later
## This needs to be rewritten completely
class RepresentationNet(hk.Module):
    """Representation network"""

    def __init__(
        self,
        hparams: ml_collections.ConfigDict,
        spec: GraphTraversalSpec,
        embedding_dim: int,
        name: str = 'representation',
    ):
        super().__init__(name=name)
        self.hparams = hparams
        self.spec = spec
        self.embedding_dim = embedding_dim

        # Define layers based on reference.py
        self.node_embedding = hk.Embed(spec.num_nodes, embedding_dim, name="node_embedding")
        self.edge_embedding = hk.Embed(spec.edge_types, embedding_dim, name="edge_embedding")
        self.adjacency_embedding = hk.Linear(embedding_dim, name="adjacency_embedding")
        self.graph_encoder = hk.nets.MLP(
            [embedding_dim] * hparams.num_layers, name="graph_encoder"
        )

    def __call__(self, inputs):
        batch_size = inputs['graph'].shape[0]
        
        # Encode graph structure
        graph_encoding = self._encode_graph(inputs, batch_size)
        
        # Encode node features
        node_features_encoding = self._encode_node_features(inputs, batch_size)
        
        return self.aggregate_graph_node_encodings(graph_encoding, node_features_encoding, batch_size)
    
    def _encode_graph(self, inputs, batch_size):
        adjacency_matrix = inputs['adjacency_matrix']
        graph_embedding = self.apply_graph_attention(adjacency_matrix, batch_size)
        return graph_embedding
    
    def _encode_node_features(self, inputs, batch_size):
        node_features = inputs['node_features']
        node_features_embedding = self.apply_node_mlp_embedder(node_features)
        return node_features_embedding
    
    def aggregate_graph_node_encodings(self, graph_encoding, node_features_encoding, batch_size):
        # Concatenate graph and node encodings
        combined_encoding = jnp.concatenate([graph_encoding, node_features_encoding], axis=-1)
        
        # Pass through MLP for final representation
        return self.apply_joint_embedder(combined_encoding, batch_size)
    
    def apply_graph_attention(self, adjacency_matrix, batch_size):
        # Apply attention mechanism to adjacency matrix
        attention_params = self.hparams.representation.attention
        make_attention_block = functools.partial(
            MultiQueryAttentionBlock, attention_params, causal_mask=False
        )
        attention_encoders = [
            make_attention_block(name=f'attention_graph_sequencer_{i}')
            for i in range(self.hparams.representation.attention_num_layers)
        ]
        
        graph_encoding = adjacency_matrix.astype(jnp.float32)
        
        for encoder in attention_encoders:
            graph_encoding, _ = encoder(graph_encoding, encoded_state=None)
            
        return graph_encoding
    
    def apply_node_mlp_embedder(self, node_features):
        node_embedder = hk.Sequential(
            [
                hk.Linear(self.embedding_dim),
                hk.LayerNorm(axis=-1),
                jax.nn.relu,
                hk.Linear(self.embedding_dim),
            ],
            name='node_features_embedder',
        )
        
        return node_embedder(node_features)
    
    def apply_joint_embedder(self, combined_representation, batch_size):
        joint_net = hk.Sequential([
            hk.Linear(self.embedding_dim),
            hk.LayerNorm(axis=-1),
            jax.nn.relu,
            hk.Linear(self.embedding_dim),
        ], name='joint_embedder')
        joint_resnet = [
            ResBlockV2(self.embedding_dim, name=f'joint_resnet_{i}')
            for i in range(self.hparams.representation.repr_net_res_blocks)
        ]
        
        joint_encoding = joint_net(jnp.mean(combined_representation, axis=1))
        for net in joint_resnet:
            joint_encoding = net(joint_encoding)
            
        return joint_encoding
    
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
            max_traversal_steps=10,
            num_nodes=10,
            edge_types=1,
            adjacency_size=10,
            action_space=4,
            completion_reward=1.0,
            correctness_weight=0.5,
            efficiency_weight=0.5,
            stochasticity_factor=0.1
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
        return Game(self.spec.action_space, self.discount, self.spec)
    

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
        self.hidden_state = None
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

  def __init__(self, history: Sequence[Action], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> Sequence[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player()

class Target(NamedTuple):
  correctness_value: float
  # TODO
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
        self, action_space_size: int, discount: float, spec: GraphTraversalSpec
    ):
        self.spec = spec
        self.environment = GraphTraversalEnv(spec)
        self.history = []
        self.rewards = []
        # TODO
        # self.latency_reward = 0
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
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
            ## TODO use adjacency matrix to check for correct values
            if next_node not in self.environment.graph[current_node]:
                return False
        return True

    def legal_actions(self) -> Sequence[Action]:
        # Game specific calculation of legal actions.
        return []

    def apply(self, action: Action):
        _, reward = self.environment.step(action)
        self.rewards.append(reward)
        self.history.append(action)
        # TODO:
        # if self.terminal() and self.is_correct():
        #    self.latency_reward = self.environment.latency_reward()

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
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
        env = GraphTraversalEnv(self.spec) # TODO: GraphTraversalEnv should generate the same graph every time
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
        return ActionHistory(self.history, self.action_space_size)
    
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

    # TODO: create a default game that acts as starting point
    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[0]

    # TODO: create a starting position in each game
    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return -1
    
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
    actions: Sequence[Action],
    network_output: NetworkOutput,
    reward: float,
):
  """Expands the node using value, reward and policy predictions from the NN."""
  node.to_play = to_play
  node.hidden_state = network_output.hidden_state
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