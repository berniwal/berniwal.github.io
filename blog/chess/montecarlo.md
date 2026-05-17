In the previous post, we explored the [minimax algorithm](/#/blog/mini-max) enhanced with [alpha-beta pruning](/#/blog/alpha-beta) and saw how we could cut off branches that would never influence the final decision. Now, we shift gears to a different paradigm for game tree search—Monte Carlo Tree Search (MCTS).

Unlike minimax, which evaluates every branch (or prunes many), MCTS is a simulation-based search algorithm. It uses random sampling (or rollouts) to estimate the value of moves. This approach is especially powerful in games like Go where the sheer size of the search tree makes exhaustive evaluation impractical.

## How does MCTS work?
MCTS builds the search tree incrementally and relies on four distinct steps: selection, expansion, simulation, and backpropagation. 
But before we dive into these steps, let's first define what information we store in each node of the tree. 
To determine which moves are promising, each node keeps track of how many times it has been visited 
and how many wins it has accumulated in the simulation. Always from the perspective of the player who is making the move.

 ![Game Tree](/blog/images/montecarlo-nodes.png)

Now given this state of a game tree, we can execute another iteration of the MCTS algorithm,
which involves the following steps:

### Selection:
Starting at the root, the algorithm recursively selects child nodes according to a policy. A common choice is the Upper Confidence Bound (UCB1) formula, which balances the trade-off between exploring new moves and exploiting moves known to be promising.

![Game Tree](/blog/images/montecarlo-selection.png)

### Expansion:
When a leaf node is reached and it is not terminal, one or more child nodes are added to the tree. This expands the search into previously unexplored areas.

![Game Tree](/blog/images/montecarlo-expansion.png)

### Simulation (Rollout):
From the newly added node, a simulation is run. The algorithm plays out random moves (or uses a simpler policy) until it reaches a terminal state or a defined depth. The outcome of this simulation is used as an estimate of the node's value.

![Game Tree](/blog/images/montecarlo-rollout.png)

### Backpropagation:
The result from the simulation is then propagated back up the tree, updating statistics (like visit count and cumulative reward) for each node along the path. This helps the algorithm learn which moves are promising over many iterations.

![Game Tree](/blog/images/montecarlo-backpropagation.png)

By repeating these steps, MCTS gradually builds a more accurate picture of which moves lead to better outcomes, without needing to evaluate the entire game tree.
In the end we can choose the best child from the root node based on the number of visits. Note it is not necessary to choose the child
with the highest reward, as the number of visits is a good indicator of the quality of the move and the one with the highest reward might be an outlier
which was not visited often enough to give a good estimate.
## Code Implementation.


```python
import math
import random

MAX_DEPTH = 10  # Maximum depth for our simulation

class MCTSNode:
    def __init__(self, state, parent=None):
        # The state is represented as a tuple: (depth, dummy_value)
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        # For simplicity, each node has 2 possible moves unless at max depth.
        self.untried_moves = [0, 1] if self.state[0] < MAX_DEPTH else []

    def is_terminal(self):
        # A node is terminal if we've reached our depth limit.
        return self.state[0] >= MAX_DEPTH

    def expand(self):
        # Remove a move from untried moves and create a new child node.
        move = self.untried_moves.pop()
        new_depth = self.state[0] + 1
        # Generate a new dummy value to simulate state change.
        new_value = random.random()
        new_state = (new_depth, new_value)
        child = MCTSNode(new_state, parent=self)
        self.children.append(child)
        return child

    def rollout(self):
        # Run a simulation from this node until a terminal state is reached.
        current_depth = self.state[0]
        total_reward = 0.0
        while current_depth < MAX_DEPTH:
            total_reward += random.random()
            current_depth += 1
        return total_reward

def uct_select_child(node):
    # Use the Upper Confidence Bound for Trees (UCT) formula.
    # UCT = (child's average reward) + sqrt(2 * log(parent visits) / child visits)
    log_parent_visits = math.log(node.visits)
    def uct(child):
        return (child.reward / child.visits) + math.sqrt(2 * log_parent_visits / child.visits)
    return max(node.children, key=uct)

def mcts(root, iterations):
    for _ in range(iterations):
        node = root
        # SELECTION: Traverse the tree until reaching a node that can be expanded.
        while node.untried_moves == [] and node.children != []:
            node = uct_select_child(node)
        # EXPANSION: If there are moves to try, expand the tree.
        if node.untried_moves:
            node = node.expand()
        # SIMULATION: Run a rollout from the new node.
        reward = node.rollout()
        # BACKPROPAGATION: Update the node statistics up to the root.
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent
    # After running all iterations, choose the best child from the root.
    return max(root.children, key=lambda c: c.visits)

# Main Function
if __name__ == "__main__":
    # Start from an initial state (depth 0, dummy value 0.0)
    root = MCTSNode((0, 0.0))
    iterations = 1000  # Number of iterations for MCTS
    best_child = mcts(root, iterations)
    
    print("Best Child State:", best_child.state)
    print("Visits:", best_child.visits)
    print("Average Reward:", best_child.reward / best_child.visits)
```