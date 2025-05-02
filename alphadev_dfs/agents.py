import random
import time
from dfs import GraphTraversalEnv, GraphTraversalSpec

# --- RandomAgent ---
class RandomAgent:
    def select_action(self, env: GraphTraversalEnv) -> int:
        legal = [i for i, x in enumerate(env.graph[env.current_node]) if x]
        return random.choice(legal) if legal else env.current_node

# --- Agents ---
class DFSAgent:
    def __init__(self):
        self.stack = []

    def select_action(self, env: GraphTraversalEnv) -> int:
        if not self.stack:
            self.stack.append(env.current_node)
        visited = set(env.visited)
        while self.stack:
            node = self.stack.pop()
            for neighbor, connected in enumerate(env.graph[node]):
                if connected and neighbor not in visited:
                    self.stack.append(neighbor)
                    return neighbor
        return env.current_node

# --- Agents ---
class MinimaxAgent:
    def __init__(self, depth=2):
        self.depth = depth

    def minimax(self, env, depth, maximizing):
        if depth == 0 or len(env.visited) == env.spec.num_nodes:
            return len(env.visited), None
        legal_actions = [i for i, x in enumerate(env.graph[env.current_node]) if x]
        best_score = -float('inf') if maximizing else float('inf')
        best_action = None
        for action in legal_actions:
            sim_env = env.clone()
            sim_env.step(action)
            score, _ = self.minimax(sim_env, depth - 1, not maximizing)
            if maximizing and score > best_score:
                best_score, best_action = score, action
            elif not maximizing and score < best_score:
                best_score, best_action = score, action
        return best_score, best_action

    def select_action(self, env: GraphTraversalEnv) -> int:
        _, action = self.minimax(env, self.depth, True)
        return action or env.current_node


class MCTSAgent:
    def __init__(self, simulations=50, discount=1.0):
        self.simulations = simulations
        self.discount = discount

    def select_action(self, env: GraphTraversalEnv) -> int:
        root = self._run_mcts(env)
        best_action = max(root['children'], key=lambda a: root['children'][a]['visits'])
        return best_action

    def _run_mcts(self, root_env: GraphTraversalEnv):
        root = {
            'visits': 0,
            'value_sum': 0,
            'children': {},
        }

        legal_actions = [i for i, x in enumerate(root_env.graph[root_env.current_node]) if x]
        for action in legal_actions:
            root['children'][action] = {'visits': 0, 'value_sum': 0}

        for _ in range(self.simulations):
            for action in legal_actions:
                sim_env = root_env.clone()
                sim_env.step(action)
                reward = len(sim_env.visited)
                value = self._simulate(sim_env, depth=5)
                total_value = reward + self.discount * value
                child = root['children'][action]
                child['visits'] += 1
                child['value_sum'] += total_value

        return root

    def _simulate(self, env: GraphTraversalEnv, depth: int) -> float:
        if depth == 0 or len(env.visited) == env.spec.num_nodes:
            return len(env.visited)
        legal_actions = [i for i, x in enumerate(env.graph[env.current_node]) if x]
        if not legal_actions:
            return len(env.visited)
        action = random.choice(legal_actions)
        env.step(action)
        return self._simulate(env, depth - 1)
    
# --- Benchmark Runner ---
def benchmark_agent(agent_class, trials=5):
    print(f"\n🔍 Benchmarking {agent_class.__name__}")
    
    total_coverage = 0.0
    total_time = 0.0
    valid_traversals = 0

    for trial in range(trials):
        spec = GraphTraversalSpec(
            100,                  # max_traversal_steps
            10 + trial * 5,       # num_nodes
            1,                    # edge_types
            10 + trial * 5,       # adjacency_size
            1.0,                  # completion_reward
            0.5,                  # correctness_weight
            0.5,                  # efficiency_weight
            0.1,                  # stochasticity_factor
            trial + 42            # seed
        )
        
        env = GraphTraversalEnv(spec)
        agent = agent_class()
        
        steps = 0
        start_time = time.time()
        while steps < spec.max_traversal_steps and len(env.visited) < spec.num_nodes:
            action = agent.select_action(env)
            env.step(action)
            steps += 1
        end_time = time.time()

        visited_nodes = len(env.visited)
        coverage = (visited_nodes / spec.num_nodes) * 100
        duration = end_time - start_time
        total_coverage += coverage
        total_time += duration

        if visited_nodes == spec.num_nodes:
            valid_traversals += 1

        print(f"Trial {trial+1:02}: Visited {visited_nodes}/{spec.num_nodes} "
              f"({coverage:.2f}%), Time: {duration:.2f}s")

    avg_coverage = total_coverage / trials
    avg_time = total_time / trials

    print(f"\n📊 Summary for {agent_class.__name__}")
    print(f"✔ Success Rate     : {valid_traversals}/{trials}")
    print(f"📈 Avg. Coverage   : {avg_coverage:.2f}%")
    print(f"⏱  Avg. Time/Trial : {avg_time:.2f} seconds")



# --- Run Benchmarks ---
benchmark_agent(MCTSAgent)
benchmark_agent(RandomAgent)
benchmark_agent(DFSAgent)
benchmark_agent(MinimaxAgent)